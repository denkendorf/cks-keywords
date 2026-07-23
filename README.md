# cks-keywords

`cks-keywords` is a transparent Python implementation of Composite Keyword
Scoring (CKS) for reconstructing missing author keywords. CKS combines
TF--IDF, document frequency, dispersion, domain focus derived from observed
metadata, and phrase quality.

## Status

Version `0.1.1` is a research-preview build generated from the
frozen B.-R. Ryu & S. Lee (2026) pipeline. It does not change the published/test results.

## Installation

```bash
python -m pip install dist/cks_keywords-0.1.1-py3-none-any.whl
python -m pip install "cks-keywords[nlp]"
python -m spacy download en_core_web_sm
```

## Frozen Paper-1 profile

```python
from cks_keywords import FrozenCKS

cks = FrozenCKS.from_profile("paper1-w050-s40")
assert cks.verify_manifest()

result = cks.extract_keywords(
    abstract="This study investigates phonological structure in sign language ...",
    title="Phonological structure in sign language",
    top_n=10,
)
print(result[["candidate_display", "candidate_score", "candidate_rank"]])
```

The packaged frozen configuration is `W050_S40`, with minimum score `0.40`.

## Exact reproduction boundary

The frozen weighted score, threshold, deterministic tie-breaking,
canonical de-duplication, and ranking are exactly reproduced from the
frozen Stage-08 candidate-feature frame. Regenerating candidates from raw
text additionally depends on the same spaCy model and the frozen V3
admissibility rule. Run `benchmarks/run_stage10_regression.py` against the
original project root before publishing a new release.

## CLI

```bash
cks-keywords verify-profile --profile paper1-w050-s40
cks-keywords extract input.csv keywords.csv \
  --id-column record_id --text-column abstract --title-column title
```

## General-domain fitting

`CKSKeywordExtractor.fit()` fits document frequency, inverse document
frequency, entropy-based dispersion, optional author-keyword domain focus,
and a keyword-length prior on a new corpus. A profile trained in one field
should not be assumed to transfer unchanged to another field.

## Scientific use

Manual inspection or later software development must not be used to retune
the frozen Paper-1 test result. New tuning constitutes a new experiment and
requires a new development/validation/test design.
