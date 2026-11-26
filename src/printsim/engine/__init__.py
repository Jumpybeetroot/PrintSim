from __future__ import annotations

import importlib
from typing import Any

from .diagnostics import (
    PrinterComparison,
    PrinterDiagnosis,
    apply_dot_overrides,
    compare_printers,
    diagnose_printer,
)
from .evaluator import ScoreBreakdown, SimulationReport, evaluate_printer
from .optimizer import OptimizationCandidate, OptimizationResult, optimize_design
from .pareto import ParetoPoint, ParetoResult, is_pymoo_available, optimize_pareto_front

_LAZY_SWEEP_EXPORTS = frozenset({"SweepCaseResult", "SweepReport", "run_sweep"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_SWEEP_EXPORTS:
        mod = importlib.import_module(".sweep", __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PrinterComparison",
    "PrinterDiagnosis",
    "ScoreBreakdown",
    "SimulationReport",
    "SweepCaseResult",
    "SweepReport",
    "apply_dot_overrides",
    "compare_printers",
    "diagnose_printer",
    "evaluate_printer",
    "OptimizationCandidate",
    "OptimizationResult",
    "optimize_design",
    "is_pymoo_available",
    "ParetoPoint",
    "ParetoResult",
    "optimize_pareto_front",
    "run_sweep",
]
