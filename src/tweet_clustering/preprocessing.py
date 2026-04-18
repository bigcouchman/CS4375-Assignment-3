from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .types import TweetRecord


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


@dataclass
class LoadResult:
    tweets: List[TweetRecord]
    raw_line_count: int
    dropped_tweet_count: int


def parse_tweet_line(line: str) -> Tuple[str, str, str]:
    """Parse one dataset line: tweet_id|timestamp|tweet_text.

    If a line is malformed, the whole line is treated as tweet text.
    """
    parts = line.rstrip("\n").split("|", 2)
    if len(parts) == 3:
        tweet_id, timestamp, text = parts
        return tweet_id.strip(), timestamp.strip(), text.strip()
    return "", "", line.strip()


def preprocess_tweet_text(text: str) -> List[str]:
    """Apply preprocessing to tweet text.

    - remove URLs
    - lowercase
    - remove words that start with @
    - convert hashtag token '#word' to 'word'
    """
    no_urls = URL_RE.sub(" ", text)
    tokens: List[str] = []

    for raw_token in no_urls.split():
        token = raw_token.lower().strip()

        # Remove mention tokens before punctuation stripping so '@name' is dropped.
        if token.startswith("@"):
            continue

        token = token.strip(string.punctuation)
        if not token:
            continue

        if token.startswith("#"):
            token = token[1:].strip(string.punctuation)
            if not token:
                continue

        tokens.append(token)

    return tokens


def load_and_preprocess_tweets(file_path: Path, drop_empty: bool = True) -> LoadResult:
    tweets: List[TweetRecord] = []
    raw_line_count = 0
    dropped_tweet_count = 0

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            raw_line_count += 1
            tweet_id, timestamp, raw_text = parse_tweet_line(line)
            tokens = preprocess_tweet_text(raw_text)

            if drop_empty and not tokens:
                dropped_tweet_count += 1
                continue

            if not tweet_id:
                tweet_id = f"line_{line_no}"

            record = TweetRecord(
                tweet_id=tweet_id,
                timestamp=timestamp,
                raw_text=raw_text,
                normalized_text=" ".join(tokens),
                tokens=tuple(tokens),
                token_set=frozenset(tokens),
            )
            tweets.append(record)

    return LoadResult(
        tweets=tweets,
        raw_line_count=raw_line_count,
        dropped_tweet_count=dropped_tweet_count,
    )
