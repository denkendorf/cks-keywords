from __future__ import annotations

import json
from importlib.resources import files

import pandas as pd

from .canonical import CanonicalMapper, normalize_keyword
from .candidates import CandidateGenerator
from .config import CKSConfig, CKSWeights
from .extractor import CKSKeywordExtractor
from .features import FittedState
from .scoring import rank_feature_frame

_PROFILES = ["paper1-w050-s40"]


def available_profiles() -> list[str]:
    return list(_PROFILES)


def _resource_root(profile_name: str):
    if profile_name not in _PROFILES:
        raise KeyError(f"Unknown frozen profile {profile_name!r}; available={_PROFILES}")
    return files("cks_keywords").joinpath("resources", profile_name)


def _read_json_resource(root, name: str) -> dict:
    resource = root.joinpath(name)
    return json.loads(resource.read_text(encoding="utf-8-sig"))


def _read_mapping_resource(root) -> CanonicalMapper:
    # Do not call importlib.resources.as_file() on the profile directory.
    # Directory materialization is unsupported for zip-imported wheels on
    # Python 3.11.  Reading each Traversable resource directly works both
    # from an installed directory and directly from a .whl/.zip file.
    with root.joinpath("canonical_mapping.csv").open("rb") as handle:
        frame = pd.read_csv(handle)

    target_candidates = ["canonical_label", "canonical_keyword", "canonical_key"]
    source_candidates = ["variant_key", "canonical_key", "mapping_base_key", "variant", "keyword"]
    target_cols = [column for column in target_candidates if column in frame.columns]
    source_cols = [column for column in source_candidates if column in frame.columns]
    if not target_cols or not source_cols:
        raise ValueError(f"Unsupported canonical mapping schema: {frame.columns.tolist()}")

    mapping: dict[str, str] = {}
    for _, row in frame.iterrows():
        target = next(
            (normalize_keyword(row[column]) for column in target_cols if normalize_keyword(row[column])),
            "",
        )
        if not target:
            continue
        mapping[target] = target
        for column in source_cols:
            source = normalize_keyword(row[column])
            if not source:
                continue
            prior = mapping.get(source)
            if prior is not None and prior != target:
                raise ValueError(
                    f"Canonical mapping conflict for {source!r}: {prior!r} vs {target!r}"
                )
            mapping[source] = target
    return CanonicalMapper(mapping)


def _read_term_statistics_resource(root) -> pd.DataFrame:
    with root.joinpath("term_statistics.csv.gz").open("rb") as handle:
        return pd.read_csv(handle, compression="gzip")


class FrozenCKS:
    def __init__(self, profile_name: str, extractor: CKSKeywordExtractor, manifest: dict) -> None:
        self.profile_name = profile_name
        self.extractor = extractor
        self.manifest = manifest

    @classmethod
    def from_profile(cls, profile_name: str = "paper1-w050-s40") -> "FrozenCKS":
        root = _resource_root(profile_name)
        manifest = _read_json_resource(root, "manifest.json")
        configuration = _read_json_resource(root, "configuration.json")
        rule = _read_json_resource(root, "candidate_rule.json")
        fitted = _read_json_resource(root, "fitted_resources.json")
        mapper = _read_mapping_resource(root)
        stats = _read_term_statistics_resource(root)

        maximum = max(int(value) for value in fitted["gold_length_counts"].values())
        length_prior = {
            int(key): int(value) / maximum
            for key, value in fitted["gold_length_counts"].items()
        }
        state = FittedState(int(fitted["fit_record_count"]), stats, length_prior, mapper)
        config = CKSConfig(
            weights=CKSWeights.from_mapping(configuration["weights"]),
            minimum_score=float(configuration["minimum_cks_score"]),
            top_n=10,
            max_ngram=int(rule.get("max_ngram", 4)),
            profile_name=profile_name,
            configuration_id=configuration["configuration_id"],
        )
        extractor = CKSKeywordExtractor(
            config,
            CandidateGenerator(rule, config.spacy_model),
            mapper,
            state,
        )
        return cls(profile_name, extractor, manifest)

    def verify_manifest(self) -> bool:
        return (
            self.manifest.get("configuration_id") == "W050_S40"
            and abs(float(self.manifest.get("minimum_cks_score")) - 0.40) < 1e-12
            and self.manifest.get("configuration_sha256")
            == "0f59ca5856ad132a6516e63e3d61650878ef1ffb8a446ae11defb679642dbb3f"
        )

    def extract_keywords(self, abstract: str, **kwargs):
        return self.extractor.extract_keywords(abstract, **kwargs)

    def extract_keywords_batch(self, documents: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.extractor.extract_keywords_batch(documents, **kwargs)

    def score_feature_frame(self, frame: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
        return rank_feature_frame(
            frame,
            self.extractor.config.weights.to_dict(),
            minimum_score=self.extractor.config.minimum_score,
            top_n=top_n,
        )
