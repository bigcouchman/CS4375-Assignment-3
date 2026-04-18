from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class TweetRecord:
    tweet_id: str
    timestamp: str
    raw_text: str
    normalized_text: str
    tokens: Tuple[str, ...]
    token_set: FrozenSet[str]
