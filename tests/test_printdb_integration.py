"""Tests for PrintDB integration (hardware catalogs, materials, profiles, Z-drives, bed mounts, and physics constants)."""

from __future__ import annotations

import pytest

from printsim.fea.components import _G_ACC, _RAIL_BLOCK_MASS_G, _RAIL_CARRIAGE_CATALOG
from printsim.fea.klipper_shaper import DEFAULT_SHAPER_DAMPING_RATIO, EI_SHAPER_VIB_TOL
from printsim.schema import (
    BED_MOUNT_CATALOG,
    BELT_CATALOG,
    HOTEND_CATALOG,
    MATERIAL_CATALOG,
    PROFILE_CATALOG,
    STEPPER_CATALOG,
    Z_DRIVE_MECHANISM_CATALOG,
    get_printdb,
    get_printdb_status,
    is_printdb_available,
)
from printsim.schema.printdb_bridge import get_record_specs, get_reference_constant
from printsim.thermal.filaments import FILAMENT_CATALOG


def test_printdb_status_shape() -> None:
    status = get_printdb_status()
    assert isinstance(status["available"], bool)
    if status["available"]:
        assert status["root"] is not None
    else:
        assert status["root"] is None


def test_reference_constant_fallback() -> None:
    val = get_reference_constant("references/does-not-exist", "missing_key", 42.5)
    assert val == pytest.approx(42.5)


def test_physics_constants_present() -> None:
    assert _G_ACC == pytest.approx(9.80665)
    assert DEFAULT_SHAPER_DAMPING_RATIO == pytest.approx(0.1)
    assert EI_SHAPER_VIB_TOL == pytest.approx(0.05)


def test_printdb_hydrated_catalogs_when_available() -> None:
    if not is_printdb_available():
        pytest.skip("PrintDB repository not accessible in this sandbox")

    db = get_printdb()
    assert db is not None

    # Every single stepper motor in STEPPER_CATALOG is backed by PrintDB
    for key, entry in STEPPER_CATALOG.items():
        assert "printdb_id" in entry, f"Stepper {key!r} missing printdb_id"
    assert STEPPER_CATALOG["ldo_2504ac"]["printdb_id"] == "components/motors/ldo-42sth48-2504ac"
    assert STEPPER_CATALOG["excit3d_max"]["printdb_id"] == "components/motors/excit3d-max-ol"
    assert STEPPER_CATALOG["isik_ouroboros"]["printdb_id"] == "components/motors/isik-tech-ouroboros-encstp"
    assert STEPPER_CATALOG["ldo_2504ac"]["motor_mass_g"] == pytest.approx(350.0)
    assert STEPPER_CATALOG["ldo_2504ac"]["holding_torque_nm"] == pytest.approx(0.55)

    # Every single belt in BELT_CATALOG is backed by PrintDB
    for key, entry in BELT_CATALOG.items():
        assert "printdb_id" in entry, f"Belt {key!r} missing printdb_id"
    assert BELT_CATALOG["gt2_6mm"]["printdb_id"] == "components/belts/gates-2gt-6mm-ll"
    assert BELT_CATALOG["gt2_12mm"]["printdb_id"] == "components/belts/gates-2gt-12mm-ll"
    assert BELT_CATALOG["at3_gen3_16mm"]["printdb_id"] == "components/belts/contitech-at3-gen3-16mm"

    # Every single hotend in HOTEND_CATALOG is backed by PrintDB
    for key, entry in HOTEND_CATALOG.items():
        assert "printdb_id" in entry, f"Hotend {key!r} missing printdb_id"
    assert HOTEND_CATALOG["e3d_revo_voron"]["printdb_id"] == "components/hotends/e3d-revo-voron"
    assert HOTEND_CATALOG["chube_compact"]["printdb_id"] == "components/hotends/lukes-lab-chube-compact"
    assert HOTEND_CATALOG["chube_compact"]["structural_stiffness_n_per_mm"] == pytest.approx(2975.0)
    assert HOTEND_CATALOG["tricorn_long"]["printdb_id"] == "components/hotends/tricorn-long"

    # Every Z-drive and bed mount in PrintSim is backed by PrintDB
    for key, entry in Z_DRIVE_MECHANISM_CATALOG.items():
        assert "printdb_id" in entry, f"Z-drive {key!r} missing printdb_id"
    for key, entry in BED_MOUNT_CATALOG.items():
        assert "printdb_id" in entry, f"Bed mount {key!r} missing printdb_id"

    # Linear rails hydrated from PrintDB
    assert _RAIL_BLOCK_MASS_G["mgn12_rail"] == (pytest.approx(34.0), pytest.approx(54.0))
    assert _RAIL_CARRIAGE_CATALOG["mgn12_rail"][6] == pytest.approx(3720.0)
    assert _RAIL_BLOCK_MASS_G["mgn9_rail"] == (pytest.approx(16.0), pytest.approx(26.0))
    assert _RAIL_BLOCK_MASS_G["mgn15_rail"] == (pytest.approx(59.0), pytest.approx(92.0))
    assert _RAIL_BLOCK_MASS_G["hgh15_rail"][0] == pytest.approx(180.0)

    # Materials, profiles & filaments hydrated from PrintDB
    al = MATERIAL_CATALOG["aluminum_6061_t6"]
    assert al.youngs_modulus_pa == pytest.approx(68.9e9)
    assert al.yield_strength_pa == pytest.approx(276.0e6)
    assert MATERIAL_CATALOG["ppa_cf_printed"].youngs_modulus_pa == pytest.approx(10.8e9)
    assert PROFILE_CATALOG["cf_tube_20x20_carbonara"].area_m2 == pytest.approx(1.257e-4, rel=1e-2)
    assert FILAMENT_CATALOG["pla"].density_kg_m3 == pytest.approx(1240.0)
    assert FILAMENT_CATALOG["abs"].density_kg_m3 == pytest.approx(1040.0)
    assert FILAMENT_CATALOG["tpu_95a"].density_kg_m3 == pytest.approx(1210.0)

    # Direct SI record helper
    motor_si = get_record_specs("components/motors/ldo-42sth48-2504ac", si=True)
    assert motor_si is not None
    assert motor_si["mass_kg"] == pytest.approx(0.35)
    assert motor_si["phase_inductance_h"] == pytest.approx(0.0015)

    # New physics references & assemblies queryable in PrintDB
    assert db.get("assemblies/gantries/carbonara-300-cf-x-beam-kit").specs["dressed_mass_g"] == pytest.approx(121.0)
    assert db.get("assemblies/toolheads/microwaved-bowden-chube-v1").specs["dressed_mass_g"] == pytest.approx(225.0)
    assert db.get("references/dynamics/empirical-printer-shaper-benchmarks").specs["voron_trident_300_x_mzv_hz"] == pytest.approx(52.5)
