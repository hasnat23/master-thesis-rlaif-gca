"""
Data schemas for the thesis pipeline.

Dataclasses defining the contract between modules:
- SubsetSample: a single CNN/DM article in the working subset
- CandidatePair: two candidate summaries for one article
- Judgment: holistic A/B preference judgment
- SentenceJudgment: per-sentence factual judgment
- PreferencePair: DPO training format (prompt, chosen, rejected)
"""

from dataclasses import dataclass, field, asdict
from typing import Literal
import json


@dataclass
class SubsetSample:
    """A single article from the CNN/DailyMail subset."""
    sample_id: str
    article: str
    reference_summary: str
    split: str  # "train", "validation", "test"


@dataclass
class CandidatePair:
    """Two candidate summaries generated for one article."""
    sample_id: str
    article: str
    reference_summary: str
    summary_a: str
    summary_b: str
    generation_params_a: dict = field(default_factory=dict)
    generation_params_b: dict = field(default_factory=dict)


@dataclass
class Judgment:
    """Holistic A/B preference judgment from the AI judge."""
    pair_id: str
    winner: Literal["A", "B", "tie"]
    confidence: float  # 0.0 to 1.0
    rationale: str
    order_swapped: bool = False  # True if A/B were presented in swapped order


@dataclass
class SentenceJudgment:
    """Per-sentence factual judgment."""
    pair_id: str
    summary_side: Literal["A", "B"]
    sentence_index: int
    sentence_text: str
    factual_score: float  # 0.0 to 1.0
    rationale: str


@dataclass
class GCAAggregation:
    """Summary-level preference derived from sentence-level judgments."""
    pair_id: str
    summary_a_score: float
    summary_b_score: float
    winner: Literal["A", "B", "tie"]
    sentence_judgments_a: list[dict] = field(default_factory=list)
    sentence_judgments_b: list[dict] = field(default_factory=list)


@dataclass
class PreferencePair:
    """DPO training format."""
    prompt: str
    chosen: str
    rejected: str
    source: Literal["holistic", "gca"]
    pair_id: str
    confidence: float = 0.0


# --- Serialization helpers ---

def to_jsonl_line(obj) -> str:
    """Serialize a dataclass instance to a single JSON line."""
    return json.dumps(asdict(obj), ensure_ascii=False)


def save_jsonl(items: list, filepath: str) -> int:
    """Save a list of dataclass instances to a JSONL file. Returns count written."""
    from pathlib import Path
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for item in items:
            f.write(to_jsonl_line(item) + "\n")
    return len(items)


def load_jsonl(filepath: str) -> list[dict]:
    """Load a JSONL file and return list of dicts."""
    items = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
