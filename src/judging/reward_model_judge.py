"""
Reward/factuality model judge for faithfulness scoring.

Uses a fixed sequence-classification model (default: CogComp/bart-faithful-summary-detector)
to score article-summary pairs deterministically.

No OpenAI API, no generative LLM calls.

Score object format:
    {
        "score":       float,                         # 0.0–1.0 faithful probability
        "label":       "faithful" | "hallucinated",
        "model_name":  str,
        "granularity": "full_summary" | "sentence",
        "metadata":    dict
    }
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "CogComp/bart-faithful-summary-detector"

# Normalised names that map to the "faithful" class (checked lowercase)
_FAITHFUL_NAMES = {"faithful", "entailment", "label_1", "1", "yes", "supported", "true"}
# Normalised names that map to the "not faithful" class
_UNFAITHFUL_NAMES = {
    "hallucinated", "unfaithful", "contradiction", "label_0",
    "0", "no", "not_supported", "false", "incorrect",
}


def _resolve_faithful_idx(id2label: dict) -> int:
    """
    Determine which output index corresponds to 'faithful'.

    Strategy:
      1. Check for a known faithful label name.
      2. Check for a known unfaithful label name and use the OTHER index.
      3. Fall back to the highest index with a warning.
    """
    label_map = {int(k): str(v).lower() for k, v in id2label.items()}

    for idx, name in label_map.items():
        if name in _FAITHFUL_NAMES:
            logger.info("Label %d ('%s') resolved as faithful.", idx, name)
            return idx

    for idx, name in label_map.items():
        if name in _UNFAITHFUL_NAMES:
            other_idx = max(k for k in label_map if k != idx)
            logger.info(
                "Label %d ('%s') resolved as unfaithful; using %d as faithful.",
                idx, name, other_idx,
            )
            return other_idx

    fallback = max(label_map.keys())
    logger.warning(
        "Could not map labels %s to a known faithful name. "
        "Using highest index %d as fallback — check output sanity.",
        id2label, fallback,
    )
    return fallback


class RewardModelJudge:
    """
    Deterministic factuality judge backed by a sequence-classification model.

    The model is loaded lazily on the first .score() call and reused for all
    subsequent calls.  Thread-safe for sequential use.

    Typical usage::

        judge = RewardModelJudge()
        result = judge.score(article, summary)
        # result["score"] is the faithful probability (0.0–1.0)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 1024,
    ):
        """
        Args:
            model_name: HuggingFace model ID (or local path).
            device:     'cpu', 'cuda', 'mps', or None (auto-detect).
            max_length: Maximum tokeniser length. Inputs are truncated to this.
        """
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._faithful_idx: Optional[int] = None

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load tokeniser and model on first use."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info("Loading judge model: %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        self._faithful_idx = _resolve_faithful_idx(self._model.config.id2label)
        logger.info(
            "Judge ready.  id2label=%s  faithful_idx=%d",
            dict(self._model.config.id2label),
            self._faithful_idx,
        )

    def _infer(self, article: str, text: str, granularity: str) -> dict:
        """Run one (article, text) inference pass."""
        self._load()

        enc = self._tokenizer(
            article,
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self._model(**enc).logits

        probs = torch.softmax(logits, dim=-1)[0].cpu()
        faithful_score = float(probs[self._faithful_idx].item())
        label = "faithful" if faithful_score >= 0.5 else "hallucinated"

        return {
            "score": round(faithful_score, 6),
            "label": label,
            "model_name": self.model_name,
            "granularity": granularity,
            "metadata": {
                "all_probs": [round(float(p), 6) for p in probs],
                "faithful_label_id": self._faithful_idx,
                "device": self.device,
            },
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, article: str, summary: str) -> dict:
        """Score a full summary against the article."""
        return self._infer(article, summary, granularity="full_summary")

    def score_sentence(self, article: str, sentence: str) -> dict:
        """Score a single sentence against the article."""
        return self._infer(article, sentence, granularity="sentence")

    def score_sentences(self, article: str, sentences: list[str]) -> list[dict]:
        """Score each sentence in a list against the article (sequential)."""
        return [self.score_sentence(article, s) for s in sentences]
