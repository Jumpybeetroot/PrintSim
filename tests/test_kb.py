"""Offline unit & CLI tests for PrintSim's local 3D printer engineering knowledge base (`printsim.kb`)."""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from printsim.cli import app
from printsim.kb import (
    TouchstoneKB,
    analyze_coverage_gaps,
    extract_structured_records,
    harvest_answer_overflow,
    harvest_discourse,
    harvest_github_issues,
    infer_preset_hint,
    ingest_local_path,
    parse_klipper_shaper_csv,
    resolve_bot_token,
    run_kb_calibration,
    scan_discord_channel,
    score_engineering_signal,
    seed_from_benchmark_docs,
)
from printsim.kb.calibrate import _classify_build_bucket


class _FakeHTTPResponse:
    """Minimal mock for ``urllib.request.urlopen`` context manager."""

    def __init__(self, body: bytes | str, headers: dict[str, str] | None = None, status: int = 200) -> None:
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self.status = status
        msg = Message()
        for k, v in (headers or {}).items():
            msg[k] = v
        self.headers = msg

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# 1. SQLite + FTS5 Store, Deduplication & Search
# ---------------------------------------------------------------------------

def test_store_fts5_dedup_and_search(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "kb.db"
    with TouchstoneKB(db_path) as kb:
        assert db_path.exists()

        # Deduplication on identical raw_item
        id1, ins1 = kb.upsert_raw_item(
            source_type="local_export",
            source_name="monolith",
            channel="monolith-builds",
            author="CloakedWayne",
            title="Road to 50k AWD Test",
            raw_text="10mm Aramid Kevlar belt at 266 Hz pluck hit 119.4 Hz ZV on Y axis.",
            signal_score=0.92,
            signal_tags=["shaper_data", "belt_data"],
            review_status="approved",
        )
        id2, ins2 = kb.upsert_raw_item(
            source_type="local_export",
            source_name="monolith",
            channel="monolith-builds",
            author="CloakedWayne",
            title="Road to 50k AWD Test",
            raw_text="10mm Aramid Kevlar belt at 266 Hz pluck hit 119.4 Hz ZV on Y axis.",
            signal_score=0.92,
            signal_tags=["shaper_data", "belt_data"],
            review_status="approved",
        )
        assert ins1 is True
        assert ins2 is False
        assert id1 == id2

        # Add structured rows
        kb.add_build(
            raw_item_id=id1,
            build_name="CloakedWayne 300 AWD",
            architecture="corexy",
            printer_model="Monolith 300",
            frame_profile="2020",
            frame_size_mm=300.0,
            panel_spec="5mm Al 5052-H32",
            drive_mode="awd",
            belt_width_mm=10.0,
            belt_core="aramid",
            belt_pluck_hz=266.0,
            x_shaper_type="mzv",
            x_shaper_hz=175.0,
            y_shaper_type="zv",
            y_shaper_hz=119.4,
            preset_hint="voron_monolith_awd",
        )
        kb.add_component_weight(
            raw_item_id=id1,
            category="toolhead",
            component_name="XOL 2 Carbon",
            mass_g=215.0,
            details="Complete with Chube Conduction",
        )
        kb.add_material_test(
            raw_item_id=id1,
            material_name="PPA-CF Annealed",
            modulus_gpa=12.5,
            hdt_c=190.0,
            creep_limit_n=120.0,
            test_notes="Zero creep under 20 lbf AWD belt load",
        )
        kb.add_troubleshooting_case(
            raw_item_id=id1,
            printer_class="Voron Trident",
            symptom="VFA & ringing",
            measured_signature="48 Hz Y peak",
            root_cause="Foam gasket decoupled 4mm 5052 side panel",
            verified_fix="Bolted 5052-H32 directly to extrusion T-slots",
        )

        # FTS5 search across all tables
        hits = kb.search("Aramid")
        assert len(hits) >= 2
        assert any(h["item_type"] == "build" for h in hits)

        # Table-scoped search + hyphenated query escaping
        mat_hits = kb.search("PPA-CF", table="material_tests")
        assert len(mat_hits) == 1
        assert mat_hits[0]["record"]["modulus_gpa"] == pytest.approx(12.5)

        # Decimal / special-character fallback search
        dec_hits = kb.search("119.4")
        assert len(dec_hits) >= 1

        # Filtered list_builds
        awd_builds = kb.list_builds(architecture="corexy", drive_mode="awd", min_y_hz=100.0)
        assert len(awd_builds) == 1
        assert awd_builds[0]["build_name"] == "CloakedWayne 300 AWD"
        assert kb.list_builds(drive_mode="2wd") == []

        st = kb.stats()
        assert st["raw_items"] == 1
        assert st["builds"] == 1
        assert st["component_weights"] == 1
        assert st["material_tests"] == 1
        assert st["troubleshooting_cases"] == 1
        assert st["fts_entries"] == 5


# ---------------------------------------------------------------------------
# 2. Signal Scoring
# ---------------------------------------------------------------------------

def test_signal_scoring_separates_engineering_from_chat() -> None:
    casual_msgs = [
        "good morning everyone!",
        "lol that blue filament looks awesome on your desk",
        "anyone know when the next livestream starts?",
    ]
    for msg in casual_msgs:
        score, tags = score_engineering_signal(msg)
        assert score == 0.0, f"Expected 0.0 for casual chat: {msg}"
        assert tags == []

    shaper_msg = (
        "Voron Trident 300 AWD 9mm EPDM @ 230 Hz pluck: "
        "X: MZV @ 142.5 Hz (<=38000 mm/s2), Y: ZV @ 96.2 Hz (<=19500 mm/s2). "
        "PPA-CF mounts annealed at 120C show zero creep."
    )
    score, tags = score_engineering_signal(shaper_msg)
    assert score >= 0.85
    assert {"shaper_data", "belt_data", "material_data"}.issubset(set(tags))

    weight_msg = "Just weighed my StealthBurner toolhead: 432 g vs XOL 2: 248 g."
    w_score, w_tags = score_engineering_signal(weight_msg)
    assert w_score >= 0.30
    assert "weight_data" in w_tags

    att_score, att_tags = score_engineering_signal(
        "Here is my latest graph",
        attachments=[{"filename": "shaper_calibrate_y_2026.png", "url": "https://cdn.example/a.png"}],
    )
    assert att_score >= 0.35
    assert "shaper_attachment" in att_tags


# ---------------------------------------------------------------------------
# 3. Deterministic Structured Extraction
# ---------------------------------------------------------------------------

def test_extract_structured_records_klipper_weights_materials_troubleshooting() -> None:
    klipper_post = """
    Voron Trident 300 4040 AWD build with 10mm aramid belts @ 260 Hz pluck and 48V TMC5160.
    Recommended shaper_type_x = mzv, shaper_freq_x = 162.4 Hz
    Recommended shaper_type_y = smooth_zv, shaper_freq_y = 108.5 Hz
    X max_accel <= 42000 mm/s2, Y max_accel <= 28k
    StealthBurner: 430g
    XOL toolhead weighs 245 g
    Y-gantry total: 1.12 kg
    PPA-CF: 12.8 GPa modulus, HDT 195 C, zero creep at 95 N.
    Had severe ringing and VFA at 52 Hz caused by foam gasket under the 4mm Al 5052 panels;
    fixed by bolting panels directly to the 4040 extrusions.
    """
    extracted = extract_structured_records(
        klipper_post,
        title="Trident 4040 AWD Calibration",
        meta={"author": "builder42"},
    )

    assert len(extracted["builds"]) == 1
    b = extracted["builds"][0]
    assert b["architecture"] == "corexy"
    assert b["frame_profile"] == "4040"
    assert b["drive_mode"] == "awd"
    assert b["belt_width_mm"] == pytest.approx(10.0)
    assert b["belt_core"] == "aramid"
    assert b["belt_pluck_hz"] == pytest.approx(260.0)
    assert b["voltage_v"] == pytest.approx(48.0)
    assert b["x_shaper_type"] == "mzv"
    assert b["x_shaper_hz"] == pytest.approx(162.4)
    assert b["x_max_accel"] == pytest.approx(42000.0)
    assert b["y_shaper_type"] == "smooth_zv"
    assert b["y_shaper_hz"] == pytest.approx(108.5)
    assert b["y_max_accel"] == pytest.approx(28000.0)
    assert b["preset_hint"] == "doomcube_4040"

    weights = {w["component_name"].lower(): w for w in extracted["component_weights"]}
    assert any("stealthburner" in k and w["mass_g"] == pytest.approx(430.0) for k, w in weights.items())
    assert any("xol" in k and w["mass_g"] == pytest.approx(245.0) for k, w in weights.items())
    assert any("gantry" in k and w["mass_g"] == pytest.approx(1120.0) for k, w in weights.items())

    assert len(extracted["material_tests"]) >= 1
    ppa = next(m for m in extracted["material_tests"] if "ppa" in m["material_name"].lower())
    assert ppa["modulus_gpa"] == pytest.approx(12.8)
    assert ppa["hdt_c"] == pytest.approx(195.0)

    assert len(extracted["troubleshooting_cases"]) == 1
    tc = extracted["troubleshooting_cases"][0]
    assert "foam" in tc["root_cause"].lower()
    assert "fixed" in tc["verified_fix"].lower()


# ---------------------------------------------------------------------------
# 4. Klipper Shaper CSV Parser
# ---------------------------------------------------------------------------

def test_parse_klipper_shaper_csv(tmp_path: Path) -> None:
    csv_file = tmp_path / "calibration_data_y_20261122.csv"
    csv_file.write_text(
        "\n".join(
            [
                "# Klipper shaper calibration dump",
                "freq,psd_x,psd_y,psd_z,psd_xyz,mzv(64.2),zv(66.8)",
                "20.0,10.0,50.0,5.0,65.0,0.01,0.02",
                "40.0,20.0,800.0,10.0,830.0,0.10,0.12",
                "45.0,15.0,300.0,8.0,323.0,0.05,0.06",
                "60.0,25.0,1500.0,20.0,1545.0,0.40,0.42",
                "64.2,30.0,3400.0,25.0,3455.0,0.95,0.90",
                "70.0,20.0,900.0,10.0,930.0,0.20,0.25",
                "105.0,12.0,1200.0,15.0,1227.0,0.15,0.18",
                "115.0,10.0,200.0,5.0,215.0,0.04,0.05",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_klipper_shaper_csv(csv_file)
    assert parsed["axis"] == "Y"
    assert parsed["peak_freq_hz"] == pytest.approx(64.2)
    assert parsed["peak_psd"] == pytest.approx(3455.0)
    assert parsed["sample_count"] == 8
    assert parsed["fitted_shapers"]["mzv"] == pytest.approx(64.2)
    sec_freqs = [p["freq_hz"] for p in parsed["secondary_peaks"]]
    assert 105.0 in sec_freqs
    assert 40.0 in sec_freqs


# ---------------------------------------------------------------------------
# 5. Seed from docs/monolith_discord_benchmarks.md
# ---------------------------------------------------------------------------

def test_seed_from_monolith_benchmarks_doc(tmp_path: Path) -> None:
    db_path = tmp_path / "seeded.db"
    with TouchstoneKB(db_path) as kb:
        res1 = seed_from_benchmark_docs(kb)
        assert res1["files_processed"] >= 2
        assert res1["builds_added"] >= 18  # 6 monolith + 13 community touchstones
        assert res1["weights_added"] >= 16
        assert res1["material_tests_added"] >= 9
        assert res1["troubleshooting_added"] >= 4

        # Verify all 6 named Monolith builds + key community touchstone builds are present
        builds = kb.list_builds()
        names = [b["build_name"] for b in builds]
        for expected in (
            "Hanarius",
            "onewayne",
            "Stas_Robotmaker",
            "Beattune",
            "CloakedWayne",
            "dderg",
            "Stock Voron Trident 300",
            "Stock Voron 2.4 300",
            "RatRig V-Core 3.1 300",
            "VzBot 330 AWD",
            "Ender 3 V2 Klipper",
            "Prusa MK4S Input Shaper",
            "FLSUN V400 Klipper",
            "Doron Velta",
        ):
            assert any(expected in n for n in names), f"Missing seeded build: {expected}"

        hanarius = next(b for b in builds if "Hanarius" in b["build_name"])
        assert hanarius["x_shaper_hz"] == pytest.approx(86.8)
        assert hanarius["y_shaper_hz"] == pytest.approx(56.8)
        assert hanarius["drive_mode"] == "2wd"
        assert hanarius["belt_width_mm"] == pytest.approx(6.0)

        dderg = next(b for b in builds if "dderg" in b["build_name"])
        assert dderg["x_shaper_hz"] == pytest.approx(186.0)
        assert dderg["y_shaper_hz"] == pytest.approx(122.0)
        assert dderg["drive_mode"] == "awd"
        assert dderg["belt_width_mm"] == pytest.approx(12.0)
        assert dderg["voltage_v"] == pytest.approx(48.0)

        # Verify belt core shootout + structural filament TDS entries extracted
        shootout_hits = kb.search("Aramid", table="material_tests")
        assert len(shootout_hits) == 1
        assert shootout_hits[0]["record"]["shaper_hz"] == pytest.approx(119.4)
        assert shootout_hits[0]["record"]["belt_pluck_hz"] == pytest.approx(266.0)

        for mat_query, expected_gpa in [("PPA-CF", 10.8), ("PPS-CF", 9.2), ("PET-CF15", 6.2)]:
            mhits = kb.search(mat_query, table="material_tests")
            assert len(mhits) >= 1, f"Missing material TDS for {mat_query}"
            assert mhits[0]["record"]["modulus_gpa"] == pytest.approx(expected_gpa)

        # Verify component weights across all 4 categories
        w_hits = kb.search("StealthBurner", table="component_weights")
        assert len(w_hits) == 1
        assert w_hits[0]["record"]["mass_g"] == pytest.approx(415.0)
        assert w_hits[0]["record"]["category"] == "toolhead"

        # Idempotent re-seeding adds zero duplicates
        total_builds_first = len(builds)
        res2 = seed_from_benchmark_docs(kb)
        assert res2["builds_added"] == 0
        assert res2["weights_added"] == 0
        assert res2["material_tests_added"] == 0
        assert len(kb.list_builds()) == total_builds_first


# ---------------------------------------------------------------------------
# 6. Local Discord Export Ingestion + Review Queue Export + Approval Round-Trip
# ---------------------------------------------------------------------------

def test_local_discord_export_and_review_queue_roundtrip(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    discord_export = {
        "guild": {"name": "Monolith 3D"},
        "channel": {"name": "monolith-builds"},
        "messages": [
            {
                "id": "1001",
                "timestamp": "2026-10-01T12:00:00Z",
                "author": {"name": "chatterbox"},
                "content": "hey guys how is everyone doing today?",
                "attachments": [],
            },
            {
                "id": "1002",
                "timestamp": "2026-10-01T12:05:00Z",
                "author": {"name": "speedbuilder"},
                "content": (
                    "Finished my Voron Trident 300 AWD with 9mm EPDM belts @ 225 Hz pluck! "
                    "X: MZV @ 154.0 Hz (<=39k), Y: ZV @ 102.5 Hz (<=21k). "
                    "A4T toolhead: 218 g."
                ),
                "attachments": [{"fileName": "axes_shaper_x.png", "url": "https://cdn.example/x.png"}],
            },
            {
                "id": "1003",
                "timestamp": "2026-10-01T12:10:00Z",
                "author": {"name": "tinker"},
                "content": "My Ender 3 VFA ringing was caused by a loose belt; fixed by tightening belt tension.",
                "attachments": [],
            },
        ],
    }
    (export_dir / "monolith_builds.json").write_text(json.dumps(discord_export), encoding="utf-8")

    # Also drop a Klipper shaper CSV in the same directory
    (export_dir / "resonances_x_test.csv").write_text(
        "freq,psd_x,psd_y,psd_z,psd_xyz\n30.0,100,5,5,110\n92.5,2400,20,10,2430\n120.0,80,5,5,90\n",
        encoding="utf-8",
    )

    db_path = tmp_path / "review_test.db"
    review_dir = tmp_path / "review_out"
    with TouchstoneKB(db_path) as kb:
        ing = ingest_local_path(kb, export_dir, min_score=0.20, auto_approve=False)
        assert ing["files_scanned"] == 2
        assert ing["items_seen"] == 4
        assert ing["items_skipped_low_signal"] == 1  # chatterbox dropped
        assert ing["items_kept"] == 3
        assert kb.stats()["builds"] == 0  # still pending human review

        exp = kb.export_review_queue(review_dir, min_score=0.20)
        assert exp["count"] == 3
        json_path = Path(exp["json_path"])
        md_path = Path(exp["markdown_path"])
        assert json_path.exists()
        assert md_path.exists()
        md_text = md_path.read_text(encoding="utf-8")
        assert "speedbuilder" in md_text
        assert "154.0" in md_text

        # Simulate human editing review_queue.json: approve speedbuilder, reject tinker, leave CSV pending
        qdata = json.loads(json_path.read_text(encoding="utf-8"))
        for it in qdata["items"]:
            if it["author"] == "speedbuilder":
                it["decision"] = "approve"
            elif it["author"] == "tinker":
                it["decision"] = "reject"
            else:
                it["decision"] = "pending"
        json_path.write_text(json.dumps(qdata, indent=2), encoding="utf-8")

        applied = kb.apply_review_decisions(json_path)
        assert applied["approved"] == 1
        assert applied["rejected"] == 1
        assert applied["pending"] == 1
        assert applied["builds_added"] == 1
        assert applied["weights_added"] == 1

        builds = kb.list_builds()
        assert len(builds) == 1
        assert builds[0]["x_shaper_hz"] == pytest.approx(154.0)
        assert builds[0]["y_shaper_hz"] == pytest.approx(102.5)
        assert kb.stats()["raw_by_status"]["approved"] == 1
        assert kb.stats()["raw_by_status"]["rejected"] == 1
        assert kb.stats()["raw_by_status"]["pending"] == 1


# ---------------------------------------------------------------------------
# 7. Mocked Discourse & GitHub Public API Harvesters
# ---------------------------------------------------------------------------

def test_mocked_discourse_and_github_harvesters(tmp_path: Path) -> None:
    db_path = tmp_path / "public.db"

    def fake_urlopen(req: Any, timeout: float = 20.0) -> _FakeHTTPResponse:
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "search.json" in url:
            return _FakeHTTPResponse(json.dumps({"topics": [{"id": 42}]}))
        if "/t/42.json" in url:
            return _FakeHTTPResponse(
                json.dumps(
                    {
                        "id": 42,
                        "slug": "trident-shaper-test",
                        "title": "Trident 300 Input Shaper Results",
                        "category_id": 7,
                        "created_at": "2026-09-10T00:00:00Z",
                        "post_stream": {
                            "posts": [
                                {
                                    "username": "klipperuser",
                                    "created_at": "2026-09-10T00:00:00Z",
                                    "cooked": "<p>Voron Trident 300 2WD 6mm belts: X: MZV @ 74.5 Hz, Y: MZV @ 51.2 Hz.</p>",
                                }
                            ]
                        },
                    }
                )
            )
        if "api.github.com/search/issues" in url:
            return _FakeHTTPResponse(
                json.dumps(
                    {
                        "items": [
                            {
                                "html_url": "https://github.com/Frix-x/klippain-shaketune/issues/99",
                                "title": "Shake&Tune AWD resonance peak at 128 Hz",
                                "body": "9mm AWD corexy X: smooth_mzv @ 128.0 Hz (32k), Y: smooth_zv @ 86.0 Hz (16k).",
                                "user": {"login": "dev99"},
                                "created_at": "2026-09-15T10:00:00Z",
                            }
                        ]
                    }
                )
            )
        raise AssertionError(f"Unexpected URL requested: {url}")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with TouchstoneKB(db_path) as kb:
            d_res = harvest_discourse(
                kb,
                base_url="https://klipper.discourse.group",
                queries=["input shaper"],
                max_topics=5,
                auto_approve=True,
                delay_s=0.0,
            )
            assert d_res["topics_fetched"] == 1
            assert d_res["items_kept"] == 1
            assert d_res["builds_added"] == 1

            g_res = harvest_github_issues(
                kb,
                repo="Frix-x/klippain-shaketune",
                queries=["resonance"],
                max_items=5,
                auto_approve=True,
                delay_s=0.0,
            )
            assert g_res["issues_fetched"] == 1
            assert g_res["items_kept"] == 1
            assert g_res["builds_added"] == 1

            assert len(kb.list_builds()) == 2


# ---------------------------------------------------------------------------
# 8. Mocked Discord Bot API Pagination, Rate-Limit Handling & Token Guardrail
# ---------------------------------------------------------------------------

def test_discord_bot_token_guardrails() -> None:
    with pytest.raises(ValueError, match="Missing Discord Bot token"):
        resolve_bot_token("")
    with pytest.raises(ValueError, match="self-botting"):
        resolve_bot_token("mfa.user_self_bot_token_example")
    with pytest.raises(ValueError, match="self-botting"):
        resolve_bot_token("Bearer oauth_user_token")
    assert resolve_bot_token("Bot MTEx.valid.token") == "MTEx.valid.token"
    assert resolve_bot_token("MTEx.valid.token") == "MTEx.valid.token"


def test_discord_bot_pagination_rate_limits_and_review_export(tmp_path: Path) -> None:
    db_path = tmp_path / "discord_bot.db"
    dl_dir = tmp_path / "attachments"
    review_dir = tmp_path / "discord_review"

    page1 = [
        {
            "id": "9002",
            "timestamp": "2026-11-01T10:00:00Z",
            "author": {"username": "awd_tester"},
            "content": "DoomCube 4040 AWD 10mm belts @ 250 Hz pluck -> X: MZV @ 168 Hz (45k), Y: ZV @ 112 Hz (25k)",
            "attachments": [
                {
                    "filename": "shaper_calibrate_x.png",
                    "url": "https://cdn.discordapp.com/attachments/1/2/shaper_calibrate_x.png",
                    "content_type": "image/png",
                    "size": 128,
                }
            ],
        },
        {
            "id": "9001",
            "timestamp": "2026-11-01T09:50:00Z",
            "author": {"username": "casual"},
            "content": "nice printer photo!",
            "attachments": [],
        },
    ]
    page2 = [
        {
            "id": "8900",
            "timestamp": "2026-10-31T20:00:00Z",
            "author": {"username": "weight_nerd"},
            "content": "Weighed my Archetype CPAP toolhead: 205 g and Chube Conduction hotend: 46 g.",
            "attachments": [],
        }
    ]

    call_state = {"page1_calls": 0, "urls": [], "auth_headers": []}
    slept: list[float] = []

    def fake_urlopen(req: Any, timeout: float = 20.0) -> _FakeHTTPResponse:
        url = req.full_url
        call_state["urls"].append(url)
        if "Authorization" in req.headers:
            call_state["auth_headers"].append(req.headers["Authorization"])

        if "cdn.discordapp.com" in url:
            return _FakeHTTPResponse(b"\x89PNG\r\n\x1a\nFAKEPNG")

        if "before=9001" in url:
            return _FakeHTTPResponse(
                json.dumps(page2),
                headers={"X-RateLimit-Remaining": "4", "X-RateLimit-Reset-After": "1.0"},
            )

        # Simulate HTTP 429 on first attempt for page 1, then succeed with X-RateLimit-Remaining: 0
        call_state["page1_calls"] += 1
        if call_state["page1_calls"] == 1:
            hdrs = Message()
            hdrs["Retry-After"] = "0.5"
            raise urllib.error.HTTPError(
                url=url,
                code=429,
                msg="Too Many Requests",
                hdrs=hdrs,
                fp=io.BytesIO(b'{"retry_after": 0.25}'),
            )
        return _FakeHTTPResponse(
            json.dumps(page1),
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset-After": "0.15"},
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with TouchstoneKB(db_path) as kb:
            res = scan_discord_channel(
                kb,
                channel_id="1234567890",
                bot_token="FAKE_OFFICIAL_BOT_TOKEN",
                limit=3,
                page_size=2,
                min_score=0.25,
                download_attachments_dir=dl_dir,
                review_out_dir=review_dir,
                sleep_fn=slept.append,
            )

    assert all(h == "Bot FAKE_OFFICIAL_BOT_TOKEN" for h in call_state["auth_headers"])
    assert res["messages_scanned"] == 3
    assert res["pages_fetched"] == 2
    assert res["rate_limit_waits"] == 2  # one HTTP 429 + one X-RateLimit-Remaining=0
    assert slept == [pytest.approx(0.25), pytest.approx(0.15)]
    assert res["items_kept"] == 2  # casual skipped
    assert res["attachments_downloaded"] == 1
    assert (dl_dir / "9002_shaper_calibrate_x.png").exists()
    assert Path(res["review_export"]["json_path"]).exists()
    assert Path(res["review_export"]["markdown_path"]).exists()


# ---------------------------------------------------------------------------
# 9. PrintSim Calibration Comparator
# ---------------------------------------------------------------------------

def test_kb_calibration_against_seeded_builds(tmp_path: Path) -> None:
    db_path = tmp_path / "cal.db"
    with TouchstoneKB(db_path) as kb:
        seed_from_benchmark_docs(kb)
        # Also inject an unmapped non-Voron outlier & a bogus Hz record to verify filtering
        kb.add_build(
            build_name="Ender 7 Community Mod",
            architecture="corexy",
            printer_model="Ender 7",
            x_shaper_type="mzv",
            x_shaper_hz=61.0,
            y_shaper_type="mzv",
            y_shaper_hz=44.0,
            preset_hint="corexy_voron_style",
        )
        kb.add_build(
            build_name="Voron Trident Bogus Sample Rate",
            architecture="corexy",
            printer_model="Voron Trident 300",
            x_shaper_hz=5.0,
            y_shaper_hz=3200.0,
            preset_hint="voron_trident_300",
        )
        rep = run_kb_calibration(kb)

    # 5 Monolith builds + 13 community touchstone builds have valid numeric X/Y Hz
    assert rep["builds_evaluated"] == 18
    assert rep["builds_skipped_unmapped"] == 1
    assert rep["builds_skipped_invalid_hz"] == 1
    assert rep["presets_evaluated"] >= 6
    assert rep["x_mape_pct"] is not None and rep["x_mape_pct"] >= 0.0
    assert rep["y_mape_pct"] is not None and rep["y_mape_pct"] >= 0.0
    assert rep["overall_mape_pct"] is not None and rep["overall_mape_pct"] >= 0.0

    for row in rep["results"]:
        assert row["predicted_x_hz"] > 0.0
        assert row["predicted_y_hz"] > 0.0
        assert "Ender 7" not in row["build_name"]
        assert "Bogus Sample Rate" not in row["build_name"]
        if row["measured_x_hz"] is not None:
            expected_err = round(((row["predicted_x_hz"] - row["measured_x_hz"]) / row["measured_x_hz"]) * 100.0, 2)
            assert row["x_error_pct"] == pytest.approx(expected_err, abs=0.02)


def test_harvest_answer_overflow_mocked(tmp_path: Path) -> None:
    db_path = tmp_path / "ao.db"

    search_html = """
    <html><body>
      <a href="/m/111111111111">Voron Trident Shaper</a>
      <a href="https://www.answeroverflow.com/m/222222222222">Toolhead weights</a>
      <a href="/m/111111111111">Duplicate link</a>
      <script>self.__next_f.push([1, "{\\"url\\":\\"\\/m\\/333333333333\\"}"])</script>
    </body></html>
    """

    page_json_ld = """
    <html><head>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "QAPage",
        "mainEntity": {
          "@type": "Question",
          "name": "Voron Trident 300 SHAPER_CALIBRATE results",
          "text": "Stock Voron Trident 300 2020 2WD 6mm belts @ 110 Hz pluck: X: MZV @ 54.6 Hz (<=8600), Y: MZV @ 47.4 Hz (<=6600).",
          "datePublished": "2026-10-12T10:00:00Z",
          "author": {"@type": "Person", "name": "trident_builder"},
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "Good baseline numbers for StealthBurner + CW2 (415 g) on clip-in acrylic panels."
          }
        }
      }
      </script>
    </head><body><main>Fallback</main></body></html>
    """

    page_next_data = """
    <html><head><title>XOL 2 Weight Check - Answer Overflow</title></head><body>
      <script id="__NEXT_DATA__" type="application/json">
      {
        "props": {
          "pageProps": {
            "thread": {"name": "XOL 2 and A4T scale weights"},
            "channel": {"name": "toolheads"},
            "messages": [
              {
                "content": "Weighed my XOL 2 toolhead: 248 g complete with Sherpa Mini and Rapido.",
                "author": {"name": "scale_checker"},
                "timestamp": "2026-10-14T15:30:00Z"
              }
            ]
          }
        }
      }
      </script>
    </body></html>
    """

    page_low_signal = """
    <html><head><title>Off topic chat - Answer Overflow</title></head>
    <body><main><h1>Off topic chat</h1><p>Just saying hello to everyone in the channel today!</p></main></body></html>
    """

    def fake_get_text(url: str, headers: dict[str, str], timeout_s: float = 20.0) -> str:
        if "answeroverflow.com/search" in url:
            return search_html
        if url.endswith("/m/111111111111"):
            return page_json_ld
        if url.endswith("/m/222222222222"):
            return page_next_data
        if url.endswith("/m/333333333333"):
            return page_low_signal
        raise AssertionError(f"Unexpected URL: {url}")

    with patch("printsim.kb.harvesters._http_get_text", side_effect=fake_get_text):
        with TouchstoneKB(db_path) as kb:
            res = harvest_answer_overflow(
                kb,
                queries=["SHAPER_CALIBRATE voron trident"],
                max_pages=10,
                auto_approve=True,
                delay_s=0.0,
            )
            assert res["source"] == "answeroverflow.com"
            assert res["pages_fetched"] == 3
            assert res["items_seen"] == 3
            assert res["items_kept"] == 2
            assert res["items_skipped_low_signal"] == 1
            assert res["builds_added"] == 1
            assert res["weights_added"] >= 1
            assert res["errors"] == []

            builds = kb.list_builds()
            assert len(builds) == 1
            assert builds[0]["x_shaper_hz"] == pytest.approx(54.6)
            assert builds[0]["y_shaper_hz"] == pytest.approx(47.4)
            assert builds[0]["preset_hint"] == "voron_trident_300"

            w_hits = kb.search("XOL", table="component_weights")
            assert len(w_hits) >= 1
            assert w_hits[0]["record"]["mass_g"] == pytest.approx(248.0)


def test_coverage_gaps_preset_hints_and_sanity_filters(tmp_path: Path) -> None:
    # 1. Verify Ender 7 (CoreXY) is NOT misclassified as bedslinger, while Ender 3 IS bedslinger
    assert (
        _classify_build_bucket(
            {
                "build_name": "Creality Ender 7 CoreXY",
                "architecture": "corexy",
                "printer_model": "Ender 7",
                "drive_mode": "2wd",
                "belt_width_mm": 6.0,
            }
        )
        != "bedslinger"
    )
    assert (
        _classify_build_bucket(
            {
                "build_name": "Ender 3 V2 Klipper",
                "architecture": "cartesian_bedslinger",
                "printer_model": "Ender 3 V2",
            }
        )
        == "bedslinger"
    )

    # 2. Verify infer_preset_hint cleanly distinguishes Voron 2.4 vs Trident vs VzBot vs RatRig
    assert infer_preset_hint(printer_model="Voron 2.4 300") == "corexy_voron_style"
    assert infer_preset_hint(printer_model="V2.4 350") == "corexy_voron_style"
    assert infer_preset_hint(printer_model="Voron Trident 300") == "voron_trident_300"
    assert infer_preset_hint(printer_model="VzBot 330 AWD", drive_mode="awd") == "vzbot_330_awd"
    assert infer_preset_hint(printer_model="RatRig V-Core 3.1 300", frame_profile="3030", belt_width_mm=9.0) == "ratrig_vcore31_300"

    # 3. Verify extract_structured_records rejects out-of-range shaper Hz (<15 or >250 Hz)
    bogus_post = (
        "ADXL345 resonance sweep config: X: 5 Hz start step, Y: 3200 Hz sample rate. "
        "No valid shaper frequency reported."
    )
    bogus_extracted = extract_structured_records(bogus_post, title="ADXL config question")
    assert bogus_extracted["builds"] == []

    # Valid shaper after a bogus step frequency in the same post is still captured
    mixed_post = "Sweep X: 5 Hz start, then final result X: MZV @ 61.5 Hz (<=11000), Y: MZV @ 48.0 Hz (<=6700)."
    mixed_extracted = extract_structured_records(mixed_post, title="Voron 2.4 300 calibration")
    assert len(mixed_extracted["builds"]) == 1
    assert mixed_extracted["builds"][0]["x_shaper_hz"] == pytest.approx(61.5)
    assert mixed_extracted["builds"][0]["y_shaper_hz"] == pytest.approx(48.0)

    # 4. Verify analyze_coverage_gaps reports all target buckets covered after seeding both docs
    db_path = tmp_path / "gaps.db"
    with TouchstoneKB(db_path) as kb:
        seed_from_benchmark_docs(kb)
        gaps = analyze_coverage_gaps(kb)

    build_status = {g["bucket"]: g["status"] for g in gaps["build_gaps"]}
    for bkt, status in build_status.items():
        assert status == "covered", f"Expected build bucket '{bkt}' to be covered, got {status}"

    weight_status = {g["category"]: g["status"] for g in gaps["weight_gaps"]}
    for cat, status in weight_status.items():
        assert status == "covered", f"Expected weight category '{cat}' to be covered, got {status}"

    mat_status = {g["topic"]: g["status"] for g in gaps["material_gaps"]}
    for topic, status in mat_status.items():
        assert status == "covered", f"Expected material topic '{topic}' to be covered, got {status}"
    assert gaps["priority_missing"] == []


# ---------------------------------------------------------------------------
# 10. CLI Smoke Tests (`printsim kb ...`)
# ---------------------------------------------------------------------------

def test_cli_kb_end_to_end_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "cli_kb.db"
    review_dir = tmp_path / "cli_review"
    note_file = tmp_path / "build_note.md"
    note_file.write_text(
        "# Test CoreXY Build\n"
        "Voron Trident 300 2WD 6mm belts: X: MZV @ 82.0 Hz (<=20k), Y: MZV @ 54.0 Hz (<=8.5k).\n"
        "StealthBurner toolhead: 415 g.\n",
        encoding="utf-8",
    )

    r_seed = runner.invoke(app, ["kb", "seed", "--db", str(db_path), "--json"])
    assert r_seed.exit_code == 0, r_seed.output
    seed_data = json.loads(r_seed.output)
    assert seed_data["builds_added"] == 19  # 6 monolith + 13 community touchstones
    assert seed_data["weights_added"] == 16

    r_stats = runner.invoke(app, ["kb", "stats", "--db", str(db_path), "--json"])
    assert r_stats.exit_code == 0, r_stats.output
    assert json.loads(r_stats.output)["builds"] == 19

    r_search = runner.invoke(app, ["kb", "search", "Hanarius", "--db", str(db_path), "--json"])
    assert r_search.exit_code == 0, r_search.output
    assert len(json.loads(r_search.output)) >= 1

    r_ingest = runner.invoke(app, ["kb", "ingest", str(note_file), "--db", str(db_path), "--json"])
    assert r_ingest.exit_code == 0, r_ingest.output
    assert json.loads(r_ingest.output)["items_kept"] == 1

    r_rev = runner.invoke(
        app,
        ["kb", "review", "--out", str(review_dir), "--min-score", "0.20", "--db", str(db_path), "--json"],
    )
    assert r_rev.exit_code == 0, r_rev.output
    rev_info = json.loads(r_rev.output)
    assert rev_info["count"] == 1

    # Approve the exported item and verify CLI approve command
    qpath = Path(rev_info["json_path"])
    qjson = json.loads(qpath.read_text(encoding="utf-8"))
    qjson["items"][0]["decision"] = "approve"
    qpath.write_text(json.dumps(qjson, indent=2), encoding="utf-8")

    r_app = runner.invoke(app, ["kb", "approve", str(qpath), "--db", str(db_path), "--json"])
    assert r_app.exit_code == 0, r_app.output
    assert json.loads(r_app.output)["builds_added"] == 1

    r_cal = runner.invoke(app, ["kb", "calibrate", "--db", str(db_path), "--json"])
    assert r_cal.exit_code == 0, r_cal.output
    cal_data = json.loads(r_cal.output)
    assert cal_data["builds_evaluated"] == 19  # 18 seeded with Hz + 1 newly approved

    r_gaps = runner.invoke(app, ["kb", "gaps", "--db", str(db_path), "--json"])
    assert r_gaps.exit_code == 0, r_gaps.output
    gaps_data = json.loads(r_gaps.output)
    assert gaps_data["priority_missing"] == []

    # Verify sync-public CLI wiring for --include-answer-overflow / --no-include-answer-overflow
    with (
        patch("printsim.kb.harvest_discourse", return_value={"source": "klipper.discourse.group", "items_seen": 1, "items_kept": 1, "items_skipped_low_signal": 0, "errors": []}),
        patch("printsim.kb.harvest_github_issues", return_value={"repo": "mock/repo", "items_seen": 1, "items_kept": 1, "items_skipped_low_signal": 0, "errors": []}),
        patch("printsim.kb.harvest_answer_overflow", return_value={"source": "answeroverflow.com", "items_seen": 2, "items_kept": 1, "items_skipped_low_signal": 1, "errors": []}) as mock_ao,
    ):
        r_sync_with = runner.invoke(
            app,
            ["kb", "sync-public", "--source", "all", "--delay", "0", "--db", str(db_path), "--json"],
        )
        assert r_sync_with.exit_code == 0, r_sync_with.output
        assert mock_ao.call_count == 1
        sync_with_data = json.loads(r_sync_with.output)
        assert any(r.get("source") == "answeroverflow.com" for r in sync_with_data["reports"])

        r_sync_without = runner.invoke(
            app,
            ["kb", "sync-public", "--source", "all", "--no-include-answer-overflow", "--delay", "0", "--db", str(db_path), "--json"],
        )
        assert r_sync_without.exit_code == 0, r_sync_without.output
        assert mock_ao.call_count == 1  # Not called when disabled
        sync_without_data = json.loads(r_sync_without.output)
        assert not any(r.get("source") == "answeroverflow.com" for r in sync_without_data["reports"])
