"""PrintSim local 3D printer engineering touchstone knowledge base (`printsim.kb`)."""

from printsim.kb.calibrate import analyze_coverage_gaps, run_kb_calibration
from printsim.kb.discord_bot import resolve_bot_token, scan_discord_channel
from printsim.kb.extract import (
    extract_structured_records,
    infer_preset_hint,
    parse_klipper_shaper_csv,
    score_engineering_signal,
)
from printsim.kb.harvesters import (
    DEFAULT_ANSWER_OVERFLOW_QUERIES,
    DEFAULT_PUBLIC_DISCOURSE_TARGETS,
    DEFAULT_PUBLIC_GITHUB_REPOS,
    harvest_answer_overflow,
    harvest_discourse,
    harvest_github_issues,
    ingest_local_path,
    seed_from_benchmark_docs,
)
from printsim.kb.store import TouchstoneKB, compute_content_hash, default_kb_path

__all__ = [
    "DEFAULT_ANSWER_OVERFLOW_QUERIES",
    "DEFAULT_PUBLIC_DISCOURSE_TARGETS",
    "DEFAULT_PUBLIC_GITHUB_REPOS",
    "TouchstoneKB",
    "analyze_coverage_gaps",
    "compute_content_hash",
    "default_kb_path",
    "extract_structured_records",
    "harvest_answer_overflow",
    "harvest_discourse",
    "harvest_github_issues",
    "infer_preset_hint",
    "ingest_local_path",
    "parse_klipper_shaper_csv",
    "resolve_bot_token",
    "run_kb_calibration",
    "scan_discord_channel",
    "seed_from_benchmark_docs",
    "score_engineering_signal",
]
