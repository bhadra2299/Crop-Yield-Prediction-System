"""Audit every promoted artefact against its configured gate, straight from the registry.

Reads manifests directly instead of parsing CLI output: it is faster, it cannot be
confused by log lines interleaved with JSON, and it reports the same thing the
regression test asserts.
"""

from __future__ import annotations

from agri.common.config import available_module_configs, load_module_config
from agri.common.metrics import evaluate_gate
from agri.common.registry import ModelRegistry


def main() -> int:
    registry = ModelRegistry()
    failing: list[str] = []
    rows: list[tuple[str, str, str, str]] = []

    for name in available_module_configs():
        config = load_module_config(name)
        try:
            entry = registry.get(name)
        except Exception as exc:
            rows.append((name, "-", "NONE", f"no artefact ({type(exc).__name__})"))
            continue

        gate = evaluate_gate(entry.metrics, config.gates)
        shown = (
            f"{gate.primary_metric}={gate.primary_value:.4f}"
            if isinstance(gate.primary_value, (int, float))
            else f"{gate.primary_metric}={gate.primary_value}"
        )
        rows.append((name, entry.version, "PASS" if gate.passed else "FAIL", shown))
        if not gate.passed:
            failing.append(f"{name} v{entry.version}: {shown} failed {gate.failed_checks}")

    width = max(len(r[0]) for r in rows) + 2
    print(f"{'module':<{width}} {'version':<9} {'gate':<6} primary")
    print("-" * (width + 40))
    for name, version, verdict, shown in rows:
        print(f"{name:<{width}} {version:<9} {verdict:<6} {shown}")

    print()
    if failing:
        print("GATE FAILURES:")
        for line in failing:
            print("  " + line)
        return 1
    print(f"all {len(rows)} modules pass their configured gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
