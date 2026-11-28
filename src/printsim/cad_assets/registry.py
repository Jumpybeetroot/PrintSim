"""Runtime hardware superelement registry & cached loader (strict SI internals)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from printsim.schema import get_material

ASSETS_ROOT: Path = Path(__file__).resolve().parent.parent / "assets" / "hardware"

_DEFAULT_CATEGORIES: tuple[str, ...] = (
    "hotends",
    "toolheads",
    "assemblies",
    "gantry_brackets",
    "frame_braces",
    "x_beams",
    "carriages",
)

HARDWARE_CATEGORIES: tuple[str, ...] = _DEFAULT_CATEGORIES


@dataclass(frozen=True)
class SubpartRecoveryData:
    """Compact Craig-Bampton boundary-to-subpart recovery operators (strict SI internals).

    Maps any 12-DOF boundary displacement vector ``q_b = [u0(3), r0(3), u1(3), r1(3)]``
    (from static FEA or a modal eigenvector) directly to per-subpart strain energy,
    kinetic energy, local elastic deflection, and peak von Mises stress in pure NumPy.
    """

    names: tuple[str, ...]
    materials: tuple[str, ...]
    mass_kg: np.ndarray  # (P,)
    com_m: np.ndarray  # (P, 3)
    K_bb: np.ndarray  # (P, 12, 12)
    M_bb: np.ndarray  # (P, 12, 12)
    disp_op: np.ndarray  # (P, 3, 12)
    stress_op: np.ndarray  # (P, 6, 12)
    field_points_m: np.ndarray | None = None  # (K, 3) CAD meters
    field_disp_op: np.ndarray | None = None  # (K, 3, 12)
    field_stress_op: np.ndarray | None = None  # (K, 6, 12)
    field_subpart_idx: np.ndarray | None = None  # (K,)

    def scaled(self, scale_k: float, scale_m: float) -> "SubpartRecoveryData":
        return SubpartRecoveryData(
            names=self.names,
            materials=self.materials,
            mass_kg=self.mass_kg * float(scale_m),
            com_m=self.com_m.copy(),
            K_bb=self.K_bb * float(scale_k),
            M_bb=self.M_bb * float(scale_m),
            disp_op=self.disp_op.copy(),
            stress_op=self.stress_op * float(scale_k),
            field_points_m=self.field_points_m.copy() if self.field_points_m is not None else None,
            field_disp_op=self.field_disp_op.copy() if self.field_disp_op is not None else None,
            field_stress_op=(
                self.field_stress_op * float(scale_k) if self.field_stress_op is not None else None
            ),
            field_subpart_idx=(
                self.field_subpart_idx.copy() if self.field_subpart_idx is not None else None
            ),
        )

    def recover_field(self, q_b: np.ndarray) -> dict[str, Any] | None:
        """Evaluate exact 3D Tet4 internal flex (µm) and von Mises stress (MPa) at sampled points."""
        if (
            self.field_points_m is None
            or self.field_disp_op is None
            or self.field_stress_op is None
            or len(self.field_points_m) == 0
        ):
            return None
        q = np.asarray(q_b, dtype=np.float64).reshape(12)
        du = np.einsum("kij,j->ki", self.field_disp_op.astype(np.float64, copy=False), q)  # (K, 3)
        flex_um = np.linalg.norm(du, axis=1) * 1e6
        sig = np.einsum("kij,j->ki", self.field_stress_op.astype(np.float64, copy=False), q)  # (K, 6)
        sxx, syy, szz = sig[:, 0], sig[:, 1], sig[:, 2]
        txy, tyz, txz = sig[:, 3], sig[:, 4], sig[:, 5]
        vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (
            txy**2 + tyz**2 + txz**2
        )
        vm_mpa = np.sqrt(np.maximum(vm2, 0.0)) * 1e-6
        pts_mm = np.round(self.field_points_m.astype(np.float64) * 1e3, 2)
        max_flex = float(np.max(flex_um)) if len(flex_um) > 0 else 0.0
        max_vm = float(np.max(vm_mpa)) if len(vm_mpa) > 0 else 0.0
        return {
            "source": "cad_3d_tet4",
            "points_mm": pts_mm.tolist(),
            "flex_um": np.round(flex_um, 3).tolist(),
            "stress_mpa": np.round(vm_mpa, 3).tolist(),
            "max_flex_um": round(max_flex, 3),
            "max_stress_mpa": round(max_vm, 3),
            "subpart_idx": (
                self.field_subpart_idx.astype(int).tolist()
                if self.field_subpart_idx is not None
                else []
            ),
        }

    def recover(self, q_b: np.ndarray) -> list[dict[str, Any]]:
        """Evaluate exact per-subpart strain/kinetic energy, local flex, and von Mises stress."""
        q = np.asarray(q_b, dtype=np.float64).reshape(12)
        n_parts = len(self.names)
        if n_parts == 0:
            return []
        u_arr = np.empty(n_parts, dtype=np.float64)
        t_arr = np.empty(n_parts, dtype=np.float64)
        flex_um = np.empty(n_parts, dtype=np.float64)
        vm_mpa = np.empty(n_parts, dtype=np.float64)
        for i in range(n_parts):
            u_arr[i] = max(0.0, 0.5 * float(q @ self.K_bb[i] @ q))
            t_arr[i] = max(0.0, 0.5 * float(q @ self.M_bb[i] @ q))
            du = self.disp_op[i] @ q
            flex_um[i] = float(np.linalg.norm(du)) * 1e6
            sv = self.stress_op[i] @ q
            sxx, syy, szz, txy, tyz, txz = (
                float(sv[0]),
                float(sv[1]),
                float(sv[2]),
                float(sv[3]),
                float(sv[4]),
                float(sv[5]),
            )
            vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (
                txy**2 + tyz**2 + txz**2
            )
            vm_mpa[i] = math.sqrt(max(vm2, 0.0)) * 1e-6
        u_tot = float(np.sum(u_arr))
        t_tot = float(np.sum(t_arr))
        out: list[dict[str, Any]] = []
        for i in range(n_parts):
            out.append(
                {
                    "name": self.names[i],
                    "material_id": self.materials[i],
                    "mass_g": round(float(self.mass_kg[i]) * 1e3, 2),
                    "com_mm": [round(float(v) * 1e3, 2) for v in self.com_m[i]],
                    "strain_energy_j": float(u_arr[i]),
                    "strain_energy_pct": round((u_arr[i] / u_tot * 100.0) if u_tot > 1e-18 else 0.0, 2),
                    "kinetic_energy": float(t_arr[i]),
                    "kinetic_energy_pct": round((t_arr[i] / t_tot * 100.0) if t_tot > 1e-18 else 0.0, 2),
                    "local_flex_um": round(float(flex_um[i]), 3),
                    "stress_mpa": round(float(vm_mpa[i]), 3),
                }
            )
        return out


@dataclass
class LoadedHardwareAsset:
    """In-memory Craig-Bampton superelement bundle for a CAD-backed hardware component."""

    category: str
    asset_id: str
    manifest: dict[str, Any]
    K_cb: np.ndarray
    M_cb: np.ndarray
    K_bb: np.ndarray
    port_coords_m: np.ndarray
    fixed_interface_freqs_hz: list[float]
    com_m: np.ndarray
    inertia_kg_m2: np.ndarray
    visual_glb_path: Path | None
    stiffness_scale_applied: float = 1.0
    mass_scale_applied: float = 1.0
    subpart_recovery: SubpartRecoveryData | None = None


@dataclass(frozen=True)
class _RawCachedAsset:
    """Immutable on-disk bundle cached in memory for <0.05 ms instantiation."""

    manifest: dict[str, Any]
    K_cb: np.ndarray
    M_cb: np.ndarray
    K_bb: np.ndarray
    port_coords_m: np.ndarray
    fixed_interface_freqs_hz: tuple[float, ...]
    com_m: np.ndarray
    inertia_kg_m2: np.ndarray
    visual_glb_path: Path | None
    subpart_recovery: SubpartRecoveryData | None = None


_ASSET_CACHE: dict[tuple[str, str], _RawCachedAsset] = {}
_VISUAL_MANIFEST_CACHE: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}

# PrintDB-calibrated printed-polymer gantry-bracket creep, HDT & moisture knockdown defaults.
# Hydrated lazily from PrintDB records (``materials/generic-abs``, ``materials/bambu-pet-cf``,
# ``materials/generic-pa12-cf15``, ``materials/treed-pc-pbt-gf``, ``materials/bambu-ppa-cf``,
# ``materials/bambu-pps-cf``) with WASM-safe built-in fallbacks.
_POLYMER_PRINTDB_IDS: dict[str, str] = {
    "abs_printed": "materials/generic-abs",
    "pet_cf_printed": "materials/bambu-pet-cf",
    "pa12_cf_printed": "materials/generic-pa12-cf15",
    "pc_pbt_gf_printed": "materials/treed-pc-pbt-gf",
    "ppa_cf_printed": "materials/bambu-ppa-cf",
    "pps_cf_printed": "materials/bambu-pps-cf",
}

_POLYMER_CREEP_FALLBACKS: dict[str, dict[str, float]] = {
    "abs_printed": {
        "hdt_c": 95.0,
        "tg_c": 105.0,
        "creep_limit_n": 45.0,
        "water_absorption_pct": 0.25,
        "infill_shell_factor": 0.78,
    },
    "pet_cf_printed": {
        "hdt_c": 182.0,
        "tg_c": 75.0,
        "creep_limit_n": 85.0,
        "water_absorption_pct": 0.37,
        "infill_shell_factor": 0.78,
    },
    "pa12_cf_printed": {
        "hdt_c": 135.0,
        "tg_c": 56.0,
        "creep_limit_n": 65.0,
        "water_absorption_pct": 1.60,
        "infill_shell_factor": 0.78,
    },
    "pc_pbt_gf_printed": {
        "hdt_c": 120.0,
        "tg_c": 138.0,
        "creep_limit_n": 72.0,
        "water_absorption_pct": 0.30,
        "infill_shell_factor": 0.78,
    },
    "ppa_cf_printed": {
        "hdt_c": 196.0,
        "tg_c": 85.0,
        "creep_limit_n": 120.0,
        "water_absorption_pct": 0.65,  # annealed conditioned effective uptake (1.30% raw TDS)
        "infill_shell_factor": 0.78,
    },
    "pps_cf_printed": {
        "hdt_c": 235.0,
        "tg_c": 100.0,
        "creep_limit_n": 140.0,
        "water_absorption_pct": 0.05,
        "infill_shell_factor": 0.78,
    },
}

_POLYMER_CREEP_CACHE: dict[str, dict[str, float]] = {}


def _get_polymer_creep_params(material_id: str) -> dict[str, float]:
    """Return PrintDB-hydrated polymer HDT/creep/moisture calibration parameters (WASM-safe)."""
    cached = _POLYMER_CREEP_CACHE.get(material_id)
    if cached is not None:
        return cached

    base = dict(
        _POLYMER_CREEP_FALLBACKS.get(
            material_id,
            {
                "hdt_c": 110.0,
                "tg_c": 90.0,
                "creep_limit_n": 60.0,
                "water_absorption_pct": 0.50,
                "infill_shell_factor": 0.78,
            },
        )
    )
    rid = _POLYMER_PRINTDB_IDS.get(material_id)
    if rid is not None:
        try:
            from printsim.schema.printdb_bridge import get_record_specs

            raw = get_record_specs(rid, si=False)
            if raw:
                if raw.get("hd_temp_c") is not None:
                    base["hdt_c"] = float(raw["hd_temp_c"])
                if raw.get("glass_transition_c") is not None:
                    base["tg_c"] = float(raw["glass_transition_c"])
                if raw.get("creep_limit_n") is not None:
                    base["creep_limit_n"] = float(raw["creep_limit_n"])
                if raw.get("water_absorption_pct") is not None:
                    # Annealed PPA-CF halves equilibrium moisture plasticization vs raw unannealed TDS
                    wa = float(raw["water_absorption_pct"])
                    base["water_absorption_pct"] = 0.5 * wa if material_id == "ppa_cf_printed" else wa
        except Exception:
            pass

    _POLYMER_CREEP_CACHE[material_id] = base
    return base


def _orthotropic_effective_ratio(material_id: str) -> float:
    """Mean 3D structural modulus factor from FDM orthotropic ratios (1, e2_ratio, e3_ratio)."""
    mat = get_material(material_id)
    return float((1.0 + 0.65 * float(mat.e2_ratio) + 0.35 * float(mat.e3_ratio)) / 2.0)


def gantry_bracket_polymer_knockdown(
    material_id: str,
    chamber_temp_c: float = 25.0,
    *,
    reference_is_fdm: bool = False,
    reference_material_id: str | None = None,
    belt_tension_n: float | None = None,
) -> dict[str, float]:
    """Compute PrintDB-calibrated FDM anisotropy, moisture & HDT/creep stiffness knockdown for printed brackets.

    Calibrated against PrintDB polymer TDS & community creep records (Bambu PPA-CF/PPS-CF/PET-CF,
    Polymaker PA12-CF15, TreeD PC-PBT-GF, generic ABS). Returns exact factor decomposition and
    temperature-derated creep tension limit (N). For non-printed metals, returns neutral ``1.0`` factors.
    """
    mat = get_material(material_id)
    if not bool(mat.is_printed_polymer):
        return {
            "fdm_anisotropy_infill_factor": 1.0,
            "moisture_conditioning_factor": 1.0,
            "hdt_creep_factor": 1.0,
            "tension_creep_factor": 1.0,
            "total_stiffness_factor": 1.0,
            "hdt_c": float(mat.hdt_c or 300.0),
            "base_creep_limit_n": 250.0,
            "effective_creep_limit_n": 250.0,
        }

    params = _get_polymer_creep_params(material_id)
    aniso_cur = _orthotropic_effective_ratio(material_id)
    if reference_is_fdm and reference_material_id:
        aniso_ref = max(_orthotropic_effective_ratio(reference_material_id), 0.25)
        fdm_factor = float(np.clip(aniso_cur / aniso_ref, 0.55, 1.35))
    else:
        fdm_factor = float(params["infill_shell_factor"]) * aniso_cur

    # Conditioned ambient moisture plasticization (strongest for hygroscopic PA12-CF, ~0 for PPS-CF)
    water_pct = max(0.0, float(params["water_absorption_pct"]))
    moisture_factor = 1.0 / (1.0 + 0.11 * water_pct)

    # Sustained-load thermal creep secant-modulus knockdown vs PrintDB HDT @ 1.8 MPa
    hdt_c = max(float(params["hdt_c"]), 55.0)
    tau = max(0.0, (float(chamber_temp_c) - 25.0) / max(hdt_c - 25.0, 20.0))
    hdt_creep_factor = 1.0 / (1.0 + 0.18 * (tau**2.2))

    # Optional sustained belt-tension creep knockdown above 50% of PrintDB creep tension ceiling
    base_creep_n = max(float(params["creep_limit_n"]), 20.0)
    eff_creep_n = base_creep_n * hdt_creep_factor * moisture_factor
    if belt_tension_n is not None and float(belt_tension_n) > 0.0:
        load_ratio = float(belt_tension_n) / max(eff_creep_n, 1.0)
        tension_creep_factor = 1.0 / (1.0 + 0.22 * (max(load_ratio - 0.5, 0.0) ** 2.0))
    else:
        tension_creep_factor = 1.0

    total = float(np.clip(fdm_factor * moisture_factor * hdt_creep_factor * tension_creep_factor, 0.25, 1.10))
    return {
        "fdm_anisotropy_infill_factor": round(fdm_factor, 4),
        "moisture_conditioning_factor": round(moisture_factor, 4),
        "hdt_creep_factor": round(hdt_creep_factor, 4),
        "tension_creep_factor": round(tension_creep_factor, 4),
        "total_stiffness_factor": round(total, 4),
        "hdt_c": round(hdt_c, 1),
        "base_creep_limit_n": round(base_creep_n, 1),
        "effective_creep_limit_n": round(eff_creep_n, 1),
    }


def clear_asset_cache() -> None:
    """Clear the in-memory hardware superelement & visual manifest caches (useful after re-baking)."""
    _ASSET_CACHE.clear()
    _VISUAL_MANIFEST_CACHE.clear()
    _POLYMER_CREEP_CACHE.clear()


def _asset_dir(category: str, asset_id: str | None) -> Path | None:
    if not category or not asset_id:
        return None
    cat = str(category).strip()
    aid = str(asset_id).strip()
    if not cat or not aid or "/" in cat or "/" in aid or "\\" in cat or "\\" in aid:
        return None
    if cat in (".", "..") or aid in (".", ".."):
        return None
    return ASSETS_ROOT / cat / aid


def _manifest_with_visual_url(
    manifest: dict[str, Any],
    category: str,
    asset_id: str,
    *,
    has_superelement: bool | None = None,
) -> dict[str, Any]:
    out = dict(manifest)
    out["visual_url"] = f"/api/assets/hardware/{category}/{asset_id}/visual.glb"
    if has_superelement is not None:
        out["has_superelement"] = bool(has_superelement)
    return out


def _read_visual_only_manifest(
    category: str, asset_id: str
) -> tuple[dict[str, Any], Path] | None:
    """Read a lightweight visual-only bundle (``manifest.json`` + ``visual.glb`` without ``superelement.npz``)."""
    key = (category, asset_id)
    cached = _VISUAL_MANIFEST_CACHE.get(key)
    if cached is not None:
        return cached

    adir = _asset_dir(category, asset_id)
    if adir is None:
        return None
    manifest_path = adir / "manifest.json"
    if not manifest_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    glb_name = str(manifest.get("visual_glb") or "visual.glb")
    glb_path = adir / glb_name
    if not glb_path.is_file():
        return None

    entry = (manifest, glb_path)
    _VISUAL_MANIFEST_CACHE[key] = entry
    return entry


def _read_raw_bundle(category: str, asset_id: str) -> _RawCachedAsset | None:
    key = (category, asset_id)
    cached = _ASSET_CACHE.get(key)
    if cached is not None:
        return cached

    adir = _asset_dir(category, asset_id)
    if adir is None:
        return None
    manifest_path = adir / "manifest.json"
    npz_path = adir / "superelement.npz"
    if not (manifest_path.is_file() and npz_path.is_file()):
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subpart_recovery: SubpartRecoveryData | None = None
    with np.load(npz_path, allow_pickle=False) as data:
        K_cb = np.array(data["K_cb"], dtype=np.float64)
        M_cb = np.array(data["M_cb"], dtype=np.float64)
        K_bb = np.array(data["K_bb"], dtype=np.float64)
        port_coords_m = np.array(data["port_coords_m"], dtype=np.float64)
        freqs = tuple(float(x) for x in np.asarray(data["fixed_interface_freqs_hz"], dtype=np.float64).ravel())
        com_m = np.array(data["com_m"], dtype=np.float64).reshape(3)
        inertia_kg_m2 = np.array(data["inertia_kg_m2"], dtype=np.float64).reshape(3, 3)
        if "subpart_K_bb" in data.files and "subpart_names" in data.files:
            sp_names = tuple(str(x) for x in np.asarray(data["subpart_names"]).ravel())
            sp_mats = (
                tuple(str(x) for x in np.asarray(data["subpart_materials"]).ravel())
                if "subpart_materials" in data.files
                else tuple(str(manifest.get("reference_material_id") or "") for _ in sp_names)
            )
            sp_mass = np.array(data["subpart_mass_kg"], dtype=np.float64).reshape(-1)
            sp_com = np.array(data["subpart_com_m"], dtype=np.float64).reshape(-1, 3)
            sp_K = np.array(data["subpart_K_bb"], dtype=np.float64).reshape(-1, 12, 12)
            sp_M = np.array(data["subpart_M_bb"], dtype=np.float64).reshape(-1, 12, 12)
            sp_disp = np.array(data["subpart_disp_op"], dtype=np.float64).reshape(-1, 3, 12)
            sp_stress = np.array(data["subpart_stress_op"], dtype=np.float64).reshape(-1, 6, 12)
            for arr in (sp_mass, sp_com, sp_K, sp_M, sp_disp, sp_stress):
                arr.setflags(write=False)
            field_pts: np.ndarray | None = None
            field_disp: np.ndarray | None = None
            field_stress: np.ndarray | None = None
            field_sp_idx: np.ndarray | None = None
            if (
                "field_points_m" in data.files
                and "field_disp_op" in data.files
                and "field_stress_op" in data.files
            ):
                field_pts = np.array(data["field_points_m"], dtype=np.float32).reshape(-1, 3)
                field_disp = np.array(data["field_disp_op"], dtype=np.float32).reshape(-1, 3, 12)
                field_stress = np.array(data["field_stress_op"], dtype=np.float32).reshape(-1, 6, 12)
                field_sp_idx = (
                    np.array(data["field_subpart_idx"], dtype=np.int16).reshape(-1)
                    if "field_subpart_idx" in data.files
                    else None
                )
                for arr in (field_pts, field_disp, field_stress):
                    arr.setflags(write=False)
                if field_sp_idx is not None:
                    field_sp_idx.setflags(write=False)
            if len(sp_names) > 0 and sp_K.shape[0] == len(sp_names):
                subpart_recovery = SubpartRecoveryData(
                    names=sp_names,
                    materials=sp_mats,
                    mass_kg=sp_mass,
                    com_m=sp_com,
                    K_bb=sp_K,
                    M_bb=sp_M,
                    disp_op=sp_disp,
                    stress_op=sp_stress,
                    field_points_m=field_pts,
                    field_disp_op=field_disp,
                    field_stress_op=field_stress,
                    field_subpart_idx=field_sp_idx,
                )

    K_cb.setflags(write=False)
    M_cb.setflags(write=False)
    K_bb.setflags(write=False)
    port_coords_m.setflags(write=False)
    com_m.setflags(write=False)
    inertia_kg_m2.setflags(write=False)

    glb_name = str(manifest.get("visual_glb") or "visual.glb")
    glb_path = adir / glb_name
    visual_path = glb_path if glb_path.is_file() else None

    raw = _RawCachedAsset(
        manifest=manifest,
        K_cb=K_cb,
        M_cb=M_cb,
        K_bb=K_bb,
        port_coords_m=port_coords_m,
        fixed_interface_freqs_hz=freqs,
        com_m=com_m,
        inertia_kg_m2=inertia_kg_m2,
        visual_glb_path=visual_path,
        subpart_recovery=subpart_recovery,
    )
    _ASSET_CACHE[key] = raw
    return raw


def has_hardware_asset(category: str, asset_id: str | None) -> bool:
    """Return True if a baked superelement bundle exists for ``(category, asset_id)``."""
    adir = _asset_dir(category, asset_id)
    if adir is None:
        return False
    return (adir / "manifest.json").is_file() and (adir / "superelement.npz").is_file()


def has_visual_asset(category: str, asset_id: str | None) -> bool:
    """Return True if a baked ``visual.glb`` exists for ``(category, asset_id)``."""
    adir = _asset_dir(category, asset_id)
    if adir is None:
        return False
    if (adir / "visual.glb").is_file():
        return True
    return get_visual_glb_path(category, asset_id) is not None


def get_hardware_asset_manifest(category: str, asset_id: str | None) -> dict[str, Any] | None:
    """Return a copy of the baked ``manifest.json`` (with ``visual_url`` and ``has_superelement``) or ``None``."""
    if not asset_id:
        return None
    aid = str(asset_id)
    raw = _read_raw_bundle(category, aid)
    if raw is not None:
        return _manifest_with_visual_url(raw.manifest, category, aid, has_superelement=True)
    vis = _read_visual_only_manifest(category, aid)
    if vis is not None:
        return _manifest_with_visual_url(vis[0], category, aid, has_superelement=False)
    return None


def get_visual_glb_path(category: str, asset_id: str | None) -> Path | None:
    """Return the path to ``visual.glb`` for ``(category, asset_id)`` if present (supports visual-only bundles)."""
    if not asset_id:
        return None
    aid = str(asset_id)
    adir = _asset_dir(category, aid)
    if adir is None:
        return None
    cached_raw = _ASSET_CACHE.get((category, aid))
    if cached_raw is not None:
        return cached_raw.visual_glb_path
    vis = _read_visual_only_manifest(category, aid)
    if vis is not None:
        return vis[1]
    default_glb = adir / "visual.glb"
    return default_glb if default_glb.is_file() else None


def list_hardware_assets() -> dict[str, dict[str, dict[str, Any]]]:
    """Discover all baked CAD hardware assets (superelements and visual-only bundles) grouped by category."""
    catalog: dict[str, dict[str, dict[str, Any]]] = {cat: {} for cat in _DEFAULT_CATEGORIES}
    if not ASSETS_ROOT.is_dir():
        return catalog

    for cat_dir in sorted(p for p in ASSETS_ROOT.iterdir() if p.is_dir()):
        cat_name = cat_dir.name
        bucket = catalog.setdefault(cat_name, {})
        for asset_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            aid = asset_dir.name
            raw = _read_raw_bundle(cat_name, aid)
            if raw is not None:
                bucket[aid] = _manifest_with_visual_url(
                    raw.manifest, cat_name, aid, has_superelement=True
                )
            else:
                vis = _read_visual_only_manifest(cat_name, aid)
                if vis is not None:
                    bucket[aid] = _manifest_with_visual_url(
                        vis[0], cat_name, aid, has_superelement=False
                    )
    return catalog


def load_superelement(
    category: str,
    asset_id: str,
    material_id: str | None = None,
    chamber_temp_c: float = 25.0,
    stiffness_mult: float = 1.0,
) -> LoadedHardwareAsset | None:
    """Load a baked CAD superelement with optional material/temperature/stiffness scaling.

    For single-material assets (``not manifest["is_multi_material"]``), providing
    ``material_id`` scales ``K_cb`` and ``K_bb`` by ``E_eff(material_id, chamber_temp_c) / E_ref``,
    scales ``M_cb`` and ``inertia_kg_m2`` by ``rho / rho_ref``, and updates
    ``fixed_interface_freqs_hz`` by ``sqrt(scale_k / scale_m)``.
    For multi-material assets, only ``stiffness_mult`` is applied to ``K_cb`` and ``K_bb``.
    """
    raw = _read_raw_bundle(category, asset_id)
    if raw is None:
        return None

    manifest = _manifest_with_visual_url(raw.manifest, category, asset_id, has_superelement=True)
    scale_k = float(stiffness_mult)
    scale_m = 1.0

    is_multi = bool(manifest.get("is_multi_material", False))
    if material_id is not None and not is_multi:
        mat = get_material(material_id)
        e_ref = float(manifest["reference_youngs_modulus_pa"])
        rho_ref = float(manifest["reference_density_kg_m3"])
        e_eff = float(mat.effective_modulus_pa(float(chamber_temp_c)))
        scale_k *= e_eff / max(e_ref, 1e-12)
        scale_m *= float(mat.density_kg_m3) / max(rho_ref, 1e-12)
        if category == "gantry_brackets" and bool(mat.is_printed_polymer):
            kd = gantry_bracket_polymer_knockdown(
                material_id,
                chamber_temp_c=float(chamber_temp_c),
                reference_is_fdm=bool(manifest.get("is_fdm_anisotropic", False)),
                reference_material_id=str(manifest.get("reference_material_id") or "") or None,
            )
            scale_k *= float(kd["total_stiffness_factor"])
            manifest["polymer_knockdown_applied"] = kd

    K_cb = raw.K_cb * scale_k
    K_bb = raw.K_bb * scale_k
    M_cb = raw.M_cb * scale_m
    inertia = raw.inertia_kg_m2 * scale_m
    freq_scale = math.sqrt(max(scale_k, 0.0) / max(scale_m, 1e-18))
    freqs = [f * freq_scale for f in raw.fixed_interface_freqs_hz]

    subpart_rec = (
        raw.subpart_recovery.scaled(scale_k, scale_m)
        if raw.subpart_recovery is not None
        else None
    )

    return LoadedHardwareAsset(
        category=category,
        asset_id=asset_id,
        manifest=manifest,
        K_cb=K_cb,
        M_cb=M_cb,
        K_bb=K_bb,
        port_coords_m=raw.port_coords_m.copy(),
        fixed_interface_freqs_hz=freqs,
        com_m=raw.com_m.copy(),
        inertia_kg_m2=inertia,
        visual_glb_path=raw.visual_glb_path,
        stiffness_scale_applied=scale_k,
        mass_scale_applied=scale_m,
        subpart_recovery=subpart_rec,
    )


load_hardware_asset = load_superelement
