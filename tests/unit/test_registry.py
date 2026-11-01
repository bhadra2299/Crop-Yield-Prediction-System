"""Versioned model registry tests.

What this protects against
--------------------------
A registry that serves an unreproducible artefact, or deletes the version that is
currently promoted. The manifest is the audit trail: if the data digest, config
digest or metrics stop being recorded, an incident can no longer be reconstructed
from the artefact alone. Every test runs inside a temporary directory and never
touches models/registry.
"""

from __future__ import annotations

import pytest

from agri.common.errors import ModelNotTrainedError, RegistryError
from agri.common.registry import LATEST_POINTER, ModelRegistry

pytestmark = pytest.mark.unit


def test_save_and_load_round_trip_preserves_the_artifact(tmp_registry: ModelRegistry) -> None:
    artifact = {"weights": [1, 2, 3], "bias": 0.5}
    entry = tmp_registry.save("demo", "1.0.0", artifact)
    assert entry.exists() is True
    assert entry.load() == artifact
    assert tmp_registry.exists("demo", "1.0.0") is True


def test_manifest_records_the_full_audit_trail(tmp_registry: ModelRegistry) -> None:
    entry = tmp_registry.save(
        "demo",
        "1.0.0",
        {"weights": [1]},
        metrics={"accuracy": 0.9},
        config_digest="cfg-digest",
        data_digest="data-digest",
        dataset="demo.csv",
        params={"n_estimators": 10},
    )
    manifest = entry.manifest
    assert manifest["artifact_sha256"]
    assert manifest["artifact_bytes"] > 0
    assert manifest["trained_at"]
    assert manifest["libraries"]
    assert manifest["metrics"] == {"accuracy": 0.9}
    assert manifest["params"] == {"n_estimators": 10}
    assert manifest["config_digest"] == "cfg-digest"
    assert manifest["data_digest"] == "data-digest"
    assert manifest["dataset"] == "demo.csv"


def test_promoted_version_wins_over_a_newer_directory(tmp_registry: ModelRegistry) -> None:
    tmp_registry.save("demo", "2.0.0", {"weights": [2]}, promote=False)
    tmp_registry.save("demo", "1.0.0", {"weights": [1]}, promote=False)
    tmp_registry.promote("demo", "1.0.0")

    pointer = tmp_registry.module_dir("demo") / LATEST_POINTER
    assert pointer.read_text(encoding="utf-8").strip() == "1.0.0"
    assert tmp_registry.latest_version("demo") == "1.0.0"
    assert tmp_registry.get("demo").version == "1.0.0"


def test_versions_lists_every_directory_in_order(tmp_registry: ModelRegistry) -> None:
    tmp_registry.save("demo", "2.0.0", {"weights": [2]}, promote=False)
    tmp_registry.save("demo", "1.0.0", {"weights": [1]}, promote=False)
    assert tmp_registry.versions("demo") == ["1.0.0", "2.0.0"]
    assert tmp_registry.versions("never_trained") == []


def test_get_raises_when_nothing_is_registered(tmp_registry: ModelRegistry) -> None:
    with pytest.raises(ModelNotTrainedError):
        tmp_registry.get("demo")
    assert tmp_registry.exists("demo") is False
    assert tmp_registry.latest_version("demo") is None


def test_get_raises_when_the_artifact_file_is_missing(tmp_registry: ModelRegistry) -> None:
    entry = tmp_registry.save("demo", "1.0.0", {"weights": [1]})
    entry.artifact_path.unlink()
    with pytest.raises(ModelNotTrainedError):
        tmp_registry.get("demo")
    assert tmp_registry.exists("demo") is False


def test_delete_refuses_the_promoted_version(tmp_registry: ModelRegistry) -> None:
    tmp_registry.save("demo", "1.0.0", {"weights": [1]})
    tmp_registry.save("demo", "2.0.0", {"weights": [2]}, promote=False)
    with pytest.raises(RegistryError):
        tmp_registry.delete("demo", "1.0.0")
    assert tmp_registry.exists("demo", "1.0.0") is True


def test_delete_removes_an_unpromoted_version(tmp_registry: ModelRegistry) -> None:
    tmp_registry.save("demo", "1.0.0", {"weights": [1]})
    tmp_registry.save("demo", "2.0.0", {"weights": [2]}, promote=False)
    tmp_registry.delete("demo", "2.0.0")
    assert tmp_registry.versions("demo") == ["1.0.0"]


def test_promote_rejects_an_unknown_version(tmp_registry: ModelRegistry) -> None:
    tmp_registry.save("demo", "1.0.0", {"weights": [1]})
    with pytest.raises(RegistryError):
        tmp_registry.promote("demo", "9.9.9")


def test_list_entries_returns_every_registered_module(tmp_registry: ModelRegistry) -> None:
    tmp_registry.save("alpha", "1.0.0", {"weights": [1]})
    tmp_registry.save("beta", "1.0.0", {"weights": [2]})
    assert {entry.name for entry in tmp_registry.list_entries()} == {"alpha", "beta"}
