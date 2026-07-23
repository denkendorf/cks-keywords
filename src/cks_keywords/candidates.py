from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .canonical import normalize_keyword

_HAS_DIGIT = re.compile(r"\d")


@dataclass(frozen=True)
class Candidate:
    display: str
    exact: str
    lemma: str
    ngram_length: int
    count: int
    structural_quality: float


class CandidateGenerator:
    """spaCy-based implementation of the frozen V3 candidate rule.

    Exact Paper-1 candidate regeneration requires spaCy and the same
    ``en_core_web_sm`` model family. The scoring API can also consume a
    precomputed feature frame and does not require spaCy.
    """

    def __init__(self, rule: dict, spacy_model: str = "en_core_web_sm") -> None:
        self.rule = rule
        self.spacy_model = spacy_model
        self._nlp = None

    @classmethod
    def from_json(cls, path: str | Path, spacy_model: str = "en_core_web_sm") -> "CandidateGenerator":
        return cls(json.loads(Path(path).read_text(encoding="utf-8-sig")), spacy_model)

    def _load_nlp(self):
        if self._nlp is None:
            try:
                import spacy
            except ImportError as exc:
                raise ImportError("Raw-text candidate generation requires: pip install 'cks-keywords[nlp]'") from exc
            try:
                self._nlp = spacy.load(self.spacy_model, disable=["ner", "parser"])
            except OSError as exc:
                raise OSError(
                    f"spaCy model {self.spacy_model!r} is unavailable. Run: "
                    f"python -m spacy download {self.spacy_model}"
                ) from exc
        return self._nlp

    def _is_admissible(self, tokens, exact: str, lemma: str) -> bool:
        n = len(tokens)
        if n < 1 or n > int(self.rule.get("max_ngram", 4)):
            return False
        malformed = set(self.rule.get("malformed_stem_forms", []))
        number_words = set(self.rule.get("number_word_unigrams", []))
        phrase_only = set(self.rule.get("phrase_only_unigrams", []))
        protected_exact = set(self.rule.get("protected_unigram_exact", []))
        protected_lemma = set(self.rule.get("protected_unigram_lemma", []))
        protected_phrase_exact = set(self.rule.get("protected_phrase_exact", []))
        protected_phrase_lemma = set(self.rule.get("protected_phrase_lemma", []))
        content_pos = set(self.rule.get("content_pos", self.rule.get("allowed_content_unigram_pos", [])))

        if n == 1:
            token = tokens[0]
            if exact in malformed or lemma in malformed:
                return False
            if exact in number_words or lemma in number_words:
                return False
            if exact in phrase_only or lemma in phrase_only:
                return False
            if exact in protected_exact or lemma in protected_lemma:
                return True
            if len(exact) < 2 or token.like_num or token.pos_ in set(self.rule.get("function_unigram_pos", [])):
                return False
            return token.pos_ in content_pos

        if exact in protected_phrase_exact or lemma in protected_phrase_lemma:
            return True
        forbidden = set(self.rule.get("multiword_forbidden_anywhere_pos", []))
        boundary = set(self.rule.get("multiword_boundary_disallowed_pos", []))
        linking = set(self.rule.get("internal_linking_lemmas", []))
        if any(t.pos_ in forbidden for t in tokens):
            return False
        if tokens[0].pos_ in boundary or tokens[-1].pos_ in boundary:
            return False
        for token in tokens[1:-1]:
            token_lemma = normalize_keyword(token.lemma_)
            if token.pos_ not in content_pos and token_lemma not in linking and token.pos_ != "NUM":
                return False
        return any(t.pos_ in content_pos for t in tokens)

    @staticmethod
    def _structural_quality(tokens, exact: str, rule: dict) -> float:
        # Preferred lexical tokens receive 1.0. Numeric contamination gives
        # a half-token contribution, reproducing the frozen values 0.50,
        # 0.75, 0.8333, and 0.875 for one- through four-token candidates.
        if not tokens:
            return 0.0
        weak_generic = {"evidence", "finding"}
        if len(tokens) == 1 and exact in weak_generic:
            return 0.45
        linking = set(rule.get("internal_linking_lemmas", []))
        values: list[float] = []
        for token in tokens:
            lemma = normalize_keyword(token.lemma_)
            if token.like_num or _HAS_DIGIT.search(token.text):
                values.append(0.5)
            elif token.is_stop and lemma not in linking:
                values.append(0.35)
            else:
                values.append(1.0)
        score = sum(values) / len(values)
        if len(tokens) == 1 and tokens[0].is_stop:
            score *= 0.35
        return max(0.0, min(1.0, float(score)))

    def generate(self, text: str) -> list[Candidate]:
        nlp = self._load_nlp()
        doc = nlp("" if text is None else str(text))
        usable = [t for t in doc if not t.is_space]
        counts: Counter[tuple[str, str, int, float]] = Counter()
        displays: dict[tuple[str, str, int, float], str] = {}
        max_ngram = int(self.rule.get("max_ngram", 4))
        for start in range(len(usable)):
            for size in range(1, max_ngram + 1):
                end = start + size
                if end > len(usable):
                    break
                tokens = usable[start:end]
                if any(t.is_punct for t in tokens):
                    break
                exact = normalize_keyword(" ".join(t.lower_ for t in tokens))
                lemma = normalize_keyword(" ".join(t.lemma_.lower() for t in tokens))
                if not exact or not self._is_admissible(tokens, exact, lemma):
                    continue
                quality = self._structural_quality(tokens, exact, self.rule)
                key = (exact, lemma, size, quality)
                counts[key] += 1
                displays.setdefault(key, " ".join(t.text for t in tokens).strip())
        return [
            Candidate(displays[key], key[0], key[1], key[2], count, key[3])
            for key, count in counts.items()
        ]
