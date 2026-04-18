from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tweet_clustering.preprocessing import (
    load_and_preprocess_tweets,
    parse_tweet_line,
    preprocess_tweet_text,
)


class TestPreprocessing(unittest.TestCase):
    def test_parse_valid_dataset_line(self) -> None:
        tweet_id, timestamp, text = parse_tweet_line(
            "123|2015-01-01 08:00:00|Health #News is here"
        )
        self.assertEqual(tweet_id, "123")
        self.assertEqual(timestamp, "2015-01-01 08:00:00")
        self.assertEqual(text, "Health #News is here")

    def test_parse_malformed_line(self) -> None:
        tweet_id, timestamp, text = parse_tweet_line("just raw tweet text")
        self.assertEqual(tweet_id, "")
        self.assertEqual(timestamp, "")
        self.assertEqual(text, "just raw tweet text")

    def test_assignment_preprocessing_rules(self) -> None:
        raw = "Check @AnnaMedaris #Depression news at https://t.co/abc123 NOW"
        tokens = preprocess_tweet_text(raw)
        self.assertEqual(tokens, ["check", "depression", "news", "at", "now"])

    def test_load_and_drop_empty(self) -> None:
        content = "\n".join(
            [
                "1|2020-01-01|@user1 @user2 https://x.com",
                "2|2020-01-02|#Health updates are useful",
            ]
        )

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            f.write(content)
            tmp_path = Path(f.name)

        try:
            result = load_and_preprocess_tweets(tmp_path, drop_empty=True)
            self.assertEqual(result.raw_line_count, 2)
            self.assertEqual(result.dropped_tweet_count, 1)
            self.assertEqual(len(result.tweets), 1)
            self.assertEqual(result.tweets[0].tokens, ("health", "updates", "are", "useful"))
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
