"""Tweet sentiment scoring with a Hugging Face transformers pipeline."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .config import DEFAULT_MODEL, ConfigError

LABELS = ("negative", "neutral", "positive")
# Models without named labels (e.g. cardiffnlp/twitter-roberta-base-sentiment) use LABEL_n.
_INDEXED_LABELS = {f"label_{i}": name for i, name in enumerate(LABELS)}

_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+")


def preprocess(text: str) -> str:
    """Normalise a tweet the way the cardiffnlp Twitter models were trained."""
    return _URL.sub("http", _MENTION.sub("@user", text)).strip()


@dataclass(frozen=True)
class SentimentResult:
    negative: float
    neutral: float
    positive: float

    @property
    def label(self) -> str:
        return max(LABELS, key=lambda name: getattr(self, name))

    @property
    def score(self) -> float:
        """Net sentiment in [-1, 1]: P(positive) - P(negative)."""
        return self.positive - self.negative

    @classmethod
    def from_pipeline_output(cls, output: Sequence[dict]) -> SentimentResult:
        probs = dict.fromkeys(LABELS, 0.0)
        for item in output:
            name = item["label"].lower()
            name = _INDEXED_LABELS.get(name, name)
            if name not in probs:
                raise ValueError(f"Unexpected sentiment label {item['label']!r}")
            probs[name] = float(item["score"])
        return cls(**probs)


def require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError:
        raise ConfigError(
            'Scoring needs the sentiment extra: pip install -e ".[sentiment]"'
        ) from None


class SentimentScorer:
    """Batch scorer. The transformers pipeline is loaded lazily on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        pipeline: Callable | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._pipeline = pipeline

    def _get_pipeline(self) -> Callable:
        if self._pipeline is None:
            require_transformers()
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=None,  # return probabilities for every label
            )
        return self._pipeline

    def score(self, texts: Sequence[str]) -> list[SentimentResult]:
        if not texts:
            return []
        outputs = self._get_pipeline()(
            [preprocess(t) for t in texts],
            batch_size=self.batch_size,
            truncation=True,
            max_length=128,
        )
        return [SentimentResult.from_pipeline_output(o) for o in outputs]
