"""CI metric-floor regression gate.

What this protects against
--------------------------
A model that is registered with metrics below the floor its own configuration
declares. Training enforces the gate at registration time, but a later config
edit, a hand-copied artefact or a trainer that forgets to check the gate could
still let a substandard model serve. This test re-evaluates the promoted
manifest of every registered module against the CURRENT config and fails if a
recorded metric no longer satisfies its gate. It skips cleanly when nothing is
registered, which is the normal state of a fresh checkout and of unit-only CI.

The registry root is resolved from the environment at call time rather than from
agri.common.registry.MODELS_DIR. The integration fixtures redirect that module
attribute for the whole session, so reading it here would audit their throwaway
training registry whenever the integration tests ran first. A CI gate must not
depend on collection order.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agri.common.config import load_module_config
from agri.common.errors import ConfigError, ModelNotTrainedError, RegistryError
from agri.common.metrics import evaluate_gate
from agri.common.paths import PROJECT_ROOT
from agri.common.registry import ModelRegistry, RegistryEntry

pytestmark = pytest.mark.regression


def production_registry() -> ModelRegistry:
    """The registry CI is asked about, immune to other tests' monkeypatching."""
    override = os.getenv("AGRI_MODEL_REGISTRY")
    return ModelRegistry(Path(override) if override else PROJECT_ROOT / "models" / "registry")


def promoted_versions(registry: ModelRegistry) -> list[tuple[str, str]]:
    """Every module directory that has a promoted version, in name order."""
    root = registry.root
    if not root.is_dir():
        return []
    pairs: list[tuple[str, str]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        version = registry.latest_version(path.name)
        if version is not None:
            pairs.append((path.name, version))
    return pairs


def test_every_promoted_model_still_satisfies_its_configured_gate() -> None:
    registry = production_registry()
    pairs = promoted_versions(registry)
    if not pairs:
        pytest.skip(
            "no promoted model is registered; the regression gate is only meaningful after a training run"
        )

    failures: list[str] = []
    audited = 0
    for module_name, version in pairs:
        try:
            config = load_module_config(module_name)
        except ConfigError:
            continue

        try:
            entry: RegistryEntry = registry.get(module_name, version)
        except (ModelNotTrainedError, RegistryError) as exc:
            failures.append(f"{module_name} v{version}: promoted entry is unreadable ({exc})")
            continue

        audited += 1
        if not entry.metrics:
            failures.append(f"{module_name} v{version}: manifest records no metrics")
            continue

        gate = evaluate_gate(entry.metrics, config.gates)
        if not gate.passed:
            failures.append(
                f"{module_name} v{version}: {gate.primary_metric}={gate.primary_value} "
                f"failed {gate.failed_checks}"
            )

    if audited == 0 and not failures:
        pytest.skip("registered modules do not match any current module configuration")

    assert not failures, "recorded metrics fell below the configured gate: " + "; ".join(failures)
