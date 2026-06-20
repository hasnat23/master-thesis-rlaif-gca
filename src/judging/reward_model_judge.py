"""
Reward/factuality model judge for faithfulness scoring.

Supports two deterministic judge backends:
  1. AlignScore (primary): factual-consistency metric from yzha/AlignScore.
  2. Sequence classification fallback: e.g. CogComp/bart-faithful-summary-detector.

No OpenAI API, no generative LLM calls.

Score object format:
    {
        "score":       float,
        "label":       "faithful" | "hallucinated",
        "model_name":  str,
        "granularity": "full_summary" | "sentence",
        "metadata":    dict
    }
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import transformers as _transformers

# transformers>=5.0 removed AdamW; alignscore still expects it at import time
if not hasattr(_transformers, "AdamW"):
    _transformers.AdamW = torch.optim.AdamW

logger = logging.getLogger(__name__)

_DEFAULT_ALIGNSCORE_REPO = "yzha/AlignScore"
_DEFAULT_ALIGNSCORE_BACKBONE = "FacebookAI/roberta-base"
_DEFAULT_ALIGNSCORE_FILENAME = "AlignScore-base.ckpt"
_DEFAULT_ALIGNSCORE_EVAL_MODE = "nli_sp"

_FAITHFUL_NAMES = {"faithful", "entailment", "label_1", "1", "yes", "supported", "true"}
_UNFAITHFUL_NAMES = {
    "hallucinated", "unfaithful", "contradiction", "label_0",
    "0", "no", "not_supported", "false", "incorrect",
}


def _resolve_faithful_idx(id2label: dict) -> int:
    """Determine which output index corresponds to 'faithful'."""
    label_map = {int(key): str(value).lower() for key, value in id2label.items()}

    for idx, name in label_map.items():
        if name in _FAITHFUL_NAMES:
            logger.info("Label %d ('%s') resolved as faithful.", idx, name)
            return idx

    for idx, name in label_map.items():
        if name in _UNFAITHFUL_NAMES:
            other_idx = max(key for key in label_map if key != idx)
            logger.info(
                "Label %d ('%s') resolved as unfaithful; using %d as faithful.",
                idx,
                name,
                other_idx,
            )
            return other_idx

    fallback = max(label_map.keys())
    logger.warning(
        "Could not map labels %s to a known faithful name. Using highest index %d as fallback.",
        id2label,
        fallback,
    )
    return fallback


class RewardModelJudge:
    """Deterministic factuality judge with AlignScore primary and classifier fallback."""

    def __init__(
        self,
        model_name: str = _DEFAULT_ALIGNSCORE_REPO,
        device: Optional[str] = None,
        max_length: int = 1024,
        backend: str = "auto",
        alignscore_backbone: str = _DEFAULT_ALIGNSCORE_BACKBONE,
        alignscore_ckpt: Optional[str] = None,
        alignscore_filename: str = _DEFAULT_ALIGNSCORE_FILENAME,
        alignscore_evaluation_mode: str = _DEFAULT_ALIGNSCORE_EVAL_MODE,
        alignscore_batch_size: int = 8,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.alignscore_backbone = alignscore_backbone
        self.alignscore_ckpt = alignscore_ckpt
        self.alignscore_filename = alignscore_filename
        self.alignscore_evaluation_mode = alignscore_evaluation_mode
        self.alignscore_batch_size = alignscore_batch_size

        self._model = None
        self._tokenizer = None
        self._alignscore = None
        self._faithful_idx: Optional[int] = None

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.backend = self._resolve_backend(backend)
        if self.backend == "alignscore":
            self.judge_name = f"{self.model_name} [{self.alignscore_backbone}]"
        else:
            self.judge_name = self.model_name

    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend

        lowered = str(self.model_name).lower()
        if self.alignscore_ckpt is not None or "alignscore" in lowered:
            return "alignscore"

        return "sequence_classification"

    def _resolve_alignscore_device(self) -> str:
        if self.device == "cuda":
            return "cuda:0"
        return "cpu"

    def _load_sequence_classification(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading sequence-classification judge: %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        self._faithful_idx = _resolve_faithful_idx(self._model.config.id2label)
        logger.info(
            "Judge ready. id2label=%s faithful_idx=%d",
            dict(self._model.config.id2label),
            self._faithful_idx,
        )

    def _resolve_alignscore_ckpt_path(self) -> str:
        if self.alignscore_ckpt:
            ckpt_path = Path(self.alignscore_ckpt).expanduser()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"AlignScore checkpoint not found: {ckpt_path}")
            return str(ckpt_path)

        from huggingface_hub import hf_hub_download

        logger.info(
            "Downloading AlignScore checkpoint from %s (%s)",
            self.model_name,
            self.alignscore_filename,
        )
        return hf_hub_download(repo_id=self.model_name, filename=self.alignscore_filename)

    def _load_alignscore(self) -> None:
        if self._alignscore is not None:
            return

        try:
            from alignscore import AlignScore
        except ImportError as exc:
            raise RuntimeError(
                "AlignScore backend requested, but the 'alignscore' package is not installed. "
                "Install it from https://github.com/yuh-zha/AlignScore and ensure its runtime "
                "dependencies (pytorch-lightning, spacy, nltk, scikit-learn) are available."
            ) from exc

        ckpt_path = self._resolve_alignscore_ckpt_path()
        alignscore_device = self._resolve_alignscore_device()

        logger.info(
            "Loading AlignScore: repo=%s backbone=%s ckpt=%s device=%s mode=%s",
            self.model_name,
            self.alignscore_backbone,
            ckpt_path,
            alignscore_device,
            self.alignscore_evaluation_mode,
        )

        self._alignscore = AlignScore(
            model=self.alignscore_backbone,
            batch_size=self.alignscore_batch_size,
            device=alignscore_device,
            ckpt_path=ckpt_path,
            evaluation_mode=self.alignscore_evaluation_mode,
            verbose=False,
        )

    def _load(self) -> None:
        if self.backend == "alignscore":
            self._load_alignscore()
        else:
            self._load_sequence_classification()

    def _infer_sequence_classification(self, article: str, text: str, granularity: str) -> dict:
        self._load_sequence_classification()

        enc = self._tokenizer(
            article,
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc = {key: value.to(self.device) for key, value in enc.items()}

        with torch.no_grad():
            logits = self._model(**enc).logits

        probs = torch.softmax(logits, dim=-1)[0].cpu()
        faithful_score = float(probs[self._faithful_idx].item())
        label = "faithful" if faithful_score >= 0.5 else "hallucinated"

        return {
            "score": round(faithful_score, 6),
            "label": label,
            "model_name": self.judge_name,
            "granularity": granularity,
            "metadata": {
                "backend": "sequence_classification",
                "all_probs": [round(float(prob), 6) for prob in probs],
                "faithful_label_id": self._faithful_idx,
                "device": self.device,
            },
        }

    def _infer_alignscore(self, article: str, text: str, granularity: str) -> dict:
        self._load_alignscore()

        raw_score = float(self._alignscore.score(contexts=[article], claims=[text])[0])
        faithful_score = max(0.0, min(1.0, raw_score))
        label = "faithful" if faithful_score >= 0.5 else "hallucinated"

        return {
            "score": round(faithful_score, 6),
            "label": label,
            "model_name": self.judge_name,
            "granularity": granularity,
            "metadata": {
                "backend": "alignscore",
                "raw_score": round(raw_score, 6),
                "device": self._resolve_alignscore_device(),
                "evaluation_mode": self.alignscore_evaluation_mode,
                "checkpoint": self.alignscore_ckpt or self.alignscore_filename,
                "backbone": self.alignscore_backbone,
            },
        }

    def _infer(self, article: str, text: str, granularity: str) -> dict:
        self._load()
        if self.backend == "alignscore":
            return self._infer_alignscore(article, text, granularity)
        return self._infer_sequence_classification(article, text, granularity)

    def score(self, article: str, summary: str) -> dict:
        return self._infer(article, summary, granularity="full_summary")

    def score_sentence(self, article: str, sentence: str) -> dict:
        return self._infer(article, sentence, granularity="sentence")

    def score_sentences(self, article: str, sentences: list[str]) -> list[dict]:
        return [self.score_sentence(article, sentence) for sentence in sentences]
