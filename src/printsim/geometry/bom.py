"""Manufacturable Cut-List, 1D Stock Nesting, and Bill of Materials (BOM) Generator."""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from printsim.fea.components import _RAIL_BLOCK_MASS_G
from printsim.geometry.builder import (
    AssemblyModel,
    build_assembly,
    front_brace_dimensions_mm,
    panel_mechanical_properties,
    top_brace_dimensions_mm,
)
from printsim.schema import (
    BED_MOUNT_CATALOG,
    EXTRUSION_FILLINGS,
    FRONT_BRACE_CATALOG,
    GANTRY_BRACKET_CATALOG,
    TOP_BRACE_CATALOG,
    Z_DRIVE_MECHANISM_CATALOG,
    PrinterSpec,
    get_material,
    get_profile,
)


@dataclass
class CutListItem:
    """A single manufacturable structural cut-list line item."""

    part_code: str
    name: str
    profile_id: str
    material_id: str
    role: str
    length_mm: float
    quantity: int
    total_length_mm: float
    unit_mass_g: float
    total_mass_g: float
    unit_cost_usd: float
    total_cost_usd: float
    notes: str = ""
    net_length_mm: float = 0.0


@dataclass
class BOMLineItem:
    """A single line item in the complete printer hardware Bill of Materials."""

    sku: str
    category: str
    name: str
    specification: str
    quantity: int | float
    unit: str
    unit_cost_usd: float
    total_cost_usd: float
    total_mass_g: float


@dataclass
class StockNestingSummary:
    """1D First-Fit Decreasing stock bar nesting summary for one profile+material."""

    profile_id: str
    material_id: str
    stock_length_mm: float
    bars_required: int
    used_length_mm: float
    waste_length_mm: float
    waste_pct: float
    bar_cuts_mm: list[list[float]]


@dataclass
class PrinterBOM:
    """Complete manufacturable cut-list, stock nesting plan, and hardware BOM."""

    printer_name: str
    architecture: str
    cut_list: list[CutListItem]
    stock_nesting: list[StockNestingSummary]
    items: list[BOMLineItem]
    category_costs_usd: dict[str, float]
    total_cost_usd: float
    total_mass_kg: float
    total_fasteners: int
    total_belt_length_m: float
    total_extrusion_length_m: float

    def to_dict(self) -> dict:
        """Return a cleanly rounded, JSON-serializable dictionary representation."""
        return {
            "printer_name": self.printer_name,
            "architecture": self.architecture,
            "summary": {
                "total_cost_usd": round(float(self.total_cost_usd), 2),
                "total_mass_kg": round(float(self.total_mass_kg), 3),
                "total_fasteners": int(self.total_fasteners),
                "total_belt_length_m": round(float(self.total_belt_length_m), 2),
                "total_extrusion_length_m": round(float(self.total_extrusion_length_m), 3),
            },
            "category_costs_usd": {
                k: round(float(v), 2) for k, v in self.category_costs_usd.items()
            },
            "total_cost_usd": round(float(self.total_cost_usd), 2),
            "total_mass_kg": round(float(self.total_mass_kg), 3),
            "total_fasteners": int(self.total_fasteners),
            "total_belt_length_m": round(float(self.total_belt_length_m), 2),
            "total_extrusion_length_m": round(float(self.total_extrusion_length_m), 3),
            "cut_list": [
                {
                    "part_code": c.part_code,
                    "name": c.name,
                    "profile_id": c.profile_id,
                    "material_id": c.material_id,
                    "role": c.role,
                    "length_mm": round(float(c.length_mm), 1),
                    "net_length_mm": round(float(c.net_length_mm or c.length_mm), 1),
                    "quantity": int(c.quantity),
                    "total_length_mm": round(float(c.total_length_mm), 1),
                    "unit_mass_g": round(float(c.unit_mass_g), 1),
                    "total_mass_g": round(float(c.total_mass_g), 1),
                    "unit_cost_usd": round(float(c.unit_cost_usd), 2),
                    "total_cost_usd": round(float(c.total_cost_usd), 2),
                    "notes": c.notes,
                }
                for c in self.cut_list
            ],
            "stock_nesting": [
                {
                    "profile_id": s.profile_id,
                    "material_id": s.material_id,
                    "stock_length_mm": round(float(s.stock_length_mm), 1),
                    "bars_required": int(s.bars_required),
                    "used_length_mm": round(float(s.used_length_mm), 1),
                    "waste_length_mm": round(float(s.waste_length_mm), 1),
                    "waste_pct": round(float(s.waste_pct), 2),
                    "bar_cuts_mm": [
                        [round(float(cut), 1) for cut in bar] for bar in s.bar_cuts_mm
                    ],
                }
                for s in self.stock_nesting
            ],
            "items": [
                {
                    "sku": item.sku,
                    "category": item.category,
                    "name": item.name,
                    "specification": item.specification,
                    "quantity": (
                        int(item.quantity)
                        if isinstance(item.quantity, int)
                        else round(float(item.quantity), 2)
                    ),
                    "unit": item.unit,
                    "unit_cost_usd": round(float(item.unit_cost_usd), 2),
                    "total_cost_usd": round(float(item.total_cost_usd), 2),
                    "total_mass_g": round(float(item.total_mass_g), 1),
                }
                for item in self.items
            ],
        }

    def to_csv(self) -> str:
        """Export Cut List and Hardware BOM as valid RFC-4180 CSV."""
        buf = io.StringIO(newline="")
        writer = csv.writer(buf)
        writer.writerow(
            [
                "section",
                "code",
                "category",
                "name",
                "specification",
                "length_mm",
                "quantity",
                "unit",
                "unit_cost_usd",
                "total_cost_usd",
                "total_mass_g",
            ]
        )
        for c in self.cut_list:
            writer.writerow(
                [
                    "CUT_LIST",
                    c.part_code,
                    c.role,
                    c.name,
                    f"{c.profile_id} / {c.material_id}",
                    f"{c.length_mm:.1f}",
                    str(c.quantity),
                    "pcs",
                    f"{c.unit_cost_usd:.2f}",
                    f"{c.total_cost_usd:.2f}",
                    f"{c.total_mass_g:.1f}",
                ]
            )
        for item in self.items:
            qty_str = (
                str(item.quantity)
                if isinstance(item.quantity, int)
                else f"{item.quantity:.2f}"
            )
            writer.writerow(
                [
                    "BOM",
                    item.sku,
                    item.category,
                    item.name,
                    item.specification,
                    "",
                    qty_str,
                    item.unit,
                    f"{item.unit_cost_usd:.2f}",
                    f"{item.total_cost_usd:.2f}",
                    f"{item.total_mass_g:.1f}",
                ]
            )
        return buf.getvalue()

    def to_markdown(self) -> str:
        """Render a GitHub-flavored Markdown Bill of Materials and Cut List report."""
        lines: list[str] = [
            f"# Bill of Materials & Cut List — {self.printer_name}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "| :--- | ---: |",
            f"| Architecture | `{self.architecture}` |",
            f"| Estimated Total Cost | ${self.total_cost_usd:,.2f} |",
            f"| Estimated Total Mass | {self.total_mass_kg:.3f} kg |",
            f"| Total Cut-List Stock Length | {self.total_extrusion_length_m:.3f} m |",
            f"| Total Drive Belt Length | {self.total_belt_length_m:.2f} m |",
            f"| Total Fasteners | {self.total_fasteners} pcs |",
            "",
            "## Cost Breakdown by Category",
            "",
            "| Category | Cost (USD) | Share |",
            "| :--- | ---: | ---: |",
        ]
        for cat, cost in self.category_costs_usd.items():
            share = (cost / self.total_cost_usd * 100.0) if self.total_cost_usd > 0 else 0.0
            lines.append(f"| `{cat}` | ${cost:,.2f} | {share:.1f}% |")

        lines.extend(
            [
                "",
                "## Structural Cut List",
                "",
                "| Code | Part | Role | Profile | Material | Length (mm) | Qty | Total (mm) | Mass (g) | Cost (USD) |",
                "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for c in self.cut_list:
            lines.append(
                f"| `{c.part_code}` | {c.name} | `{c.role}` | `{c.profile_id}` | `{c.material_id}` "
                f"| {c.length_mm:.1f} | {c.quantity} | {c.total_length_mm:.1f} "
                f"| {c.total_mass_g:.1f} | ${c.total_cost_usd:.2f} |"
            )

        lines.extend(
            [
                "",
                "## 1D Stock Bar Nesting",
                "",
                "| Profile | Material | Stock Bar (mm) | Bars | Used (mm) | Waste (mm) | Waste (%) | Bar Cut Plan (mm) |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
            ]
        )
        for s in self.stock_nesting:
            plan = "; ".join(
                "[" + ", ".join(f"{cut:.1f}" for cut in bar) + "]" for bar in s.bar_cuts_mm
            )
            lines.append(
                f"| `{s.profile_id}` | `{s.material_id}` | {s.stock_length_mm:.0f} "
                f"| {s.bars_required} | {s.used_length_mm:.1f} | {s.waste_length_mm:.1f} "
                f"| {s.waste_pct:.1f}% | {plan} |"
            )

        lines.extend(
            [
                "",
                "## Complete Hardware BOM",
                "",
                "| SKU | Category | Item | Specification | Qty | Unit | Unit ($) | Total ($) | Mass (g) |",
                "| :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |",
            ]
        )
        for item in self.items:
            qty_str = (
                str(item.quantity)
                if isinstance(item.quantity, int)
                else f"{item.quantity:.2f}"
            )
            lines.append(
                f"| `{item.sku}` | `{item.category}` | {item.name} | {item.specification} "
                f"| {qty_str} | {item.unit} | ${item.unit_cost_usd:.2f} "
                f"| ${item.total_cost_usd:.2f} | {item.total_mass_g:.1f} |"
            )

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engineering Cost & Hardware Reference Tables
# ---------------------------------------------------------------------------

_PROFILE_MFG_MULTIPLIER: dict[str, float] = {
    "extrusion": 2.2,
    "tube": 1.85,
    "rod": 1.6,
    "solid_bar": 1.8,
    "rail": 2.6,
}

_CUT_FEE_USD = 0.75

_RAIL_HARDWARE: dict[str, tuple[str, float, float, float, float]] = {
    # profile_id -> (series_label, rail_usd_per_m, carriage_usd, mass_c_g, mass_h_g)
    "mgn9_rail": ("MGN9", 32.0, 12.0, *_RAIL_BLOCK_MASS_G["mgn9_rail"]),
    "mgn12_rail": ("MGN12", 42.0, 16.0, *_RAIL_BLOCK_MASS_G["mgn12_rail"]),
    "mgn15_rail": ("MGN15", 58.0, 22.0, *_RAIL_BLOCK_MASS_G["mgn15_rail"]),
    "hgh15_rail": ("HGH15CA", 72.0, 32.0, *_RAIL_BLOCK_MASS_G["hgh15_rail"]),
    "steel_rod_8mm": ("LM8UU", 14.0, 4.5, *_RAIL_BLOCK_MASS_G["steel_rod_8mm"]),
    "steel_rod_10mm": ("LM10UU", 18.0, 6.0, *_RAIL_BLOCK_MASS_G["steel_rod_10mm"]),
    "steel_rod_12mm": ("LM12UU", 24.0, 8.0, *_RAIL_BLOCK_MASS_G["steel_rod_12mm"]),
}

_BELT_COST_USD_PER_M: dict[tuple[str, str], float] = {
    ("gt2_6mm", "fiberglass"): 4.50,
    ("gt2_6mm", "aramid"): 6.50,
    ("gt2_6mm", "steel"): 6.00,
    ("gt2_9mm", "fiberglass"): 6.50,
    ("gt2_9mm", "aramid"): 9.00,
    ("gt2_9mm", "steel"): 8.50,
    ("gt2_12mm", "fiberglass"): 8.50,
    ("gt2_12mm", "aramid"): 11.50,
    ("gt2_12mm", "steel"): 11.00,
}

_BELT_MASS_G_PER_M: dict[str, float] = {
    "gt2_6mm": 12.0,
    "gt2_9mm": 18.0,
    "gt2_12mm": 24.0,
}


@dataclass
class _MergedSegment:
    """Internal representation of a physical bar merged from collinear FEA elements."""

    start_id: str
    end_id: str
    profile_id: str
    material_id: str
    role_bucket: str
    roles: list[str]
    member_ids: list[str]
    length_m: float

    @property
    def canonical_role(self) -> str:
        unique = set(self.roles)
        if len(unique) == 1:
            return self.roles[0]
        return self.role_bucket


def _role_bucket(role: str) -> str:
    if role in ("frame", "tower", "rail"):
        return "frame"
    if role == "gantry":
        return "gantry"
    if role == "brace":
        return "brace"
    return role


def _unit_direction(p0: np.ndarray, p1: np.ndarray) -> np.ndarray | None:
    vec = p1 - p0
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return None
    return vec / norm


def _merge_collinear_members(assembly: AssemblyModel) -> list[_MergedSegment]:
    """Merge collinear adjacent structural members sharing (profile, material, role_bucket)."""
    nmap = assembly.node_map
    segments: list[_MergedSegment] = []

    for m in assembly.members:
        # Skip virtual enclosure shear diaphragm elements, auxiliary FEA rail duplicates, and toolhead/bed sub-beams
        if (
            m.role in ("toolhead", "bed")
            or m.id.startswith("panel_shear_")
            or m.id.startswith("x_rail_aux_")
            or m.id.startswith(("toolhead_", "nozzle_"))
            or m.id.startswith(("brace_top_open_", "brace_front_open_"))
        ):
            continue
        length_m = m.length_m(nmap)
        if length_m < 1e-6:
            continue
        segments.append(
            _MergedSegment(
                start_id=m.start_id,
                end_id=m.end_id,
                profile_id=m.profile.id,
                material_id=m.material.id,
                role_bucket=_role_bucket(m.role),
                roles=[m.role],
                member_ids=[m.id],
                length_m=length_m,
            )
        )

    # Iteratively merge any two segments meeting at a shared node with collinear direction
    collinear_cos_tol = 0.9999
    merged = True
    while merged:
        merged = False
        # Group indices by (profile_id, material_id, role_bucket)
        by_key: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for idx, seg in enumerate(segments):
            by_key[(seg.profile_id, seg.material_id, seg.role_bucket)].append(idx)

        for indices in by_key.values():
            if len(indices) < 2:
                continue
            # Build node -> incident segment indices map for this key
            node_to_segs: dict[str, list[int]] = defaultdict(list)
            for idx in indices:
                seg = segments[idx]
                node_to_segs[seg.start_id].append(idx)
                node_to_segs[seg.end_id].append(idx)

            pair_to_merge: tuple[int, int, str, str] | None = None
            for shared_node, inc in node_to_segs.items():
                if len(inc) < 2:
                    continue
                p_shared = nmap[shared_node].position_m
                for a_pos in range(len(inc)):
                    i = inc[a_pos]
                    seg_i = segments[i]
                    other_i = seg_i.end_id if seg_i.start_id == shared_node else seg_i.start_id
                    d_into = _unit_direction(nmap[other_i].position_m, p_shared)
                    if d_into is None:
                        continue
                    for b_pos in range(a_pos + 1, len(inc)):
                        j = inc[b_pos]
                        seg_j = segments[j]
                        other_j = (
                            seg_j.end_id if seg_j.start_id == shared_node else seg_j.start_id
                        )
                        if other_i == other_j:
                            continue
                        d_out = _unit_direction(p_shared, nmap[other_j].position_m)
                        if d_out is None:
                            continue
                        if float(np.dot(d_into, d_out)) >= collinear_cos_tol:
                            pair_to_merge = (i, j, other_i, other_j)
                            break
                    if pair_to_merge is not None:
                        break
                if pair_to_merge is not None:
                    break

            if pair_to_merge is not None:
                i, j, new_start, new_end = pair_to_merge
                seg_i = segments[i]
                seg_j = segments[j]
                combined = _MergedSegment(
                    start_id=new_start,
                    end_id=new_end,
                    profile_id=seg_i.profile_id,
                    material_id=seg_i.material_id,
                    role_bucket=seg_i.role_bucket,
                    roles=seg_i.roles + seg_j.roles,
                    member_ids=seg_i.member_ids + seg_j.member_ids,
                    length_m=seg_i.length_m + seg_j.length_m,
                )
                # Remove higher index first, then replace lower index
                hi, lo = max(i, j), min(i, j)
                segments.pop(hi)
                segments[lo] = combined
                merged = True
                break

    return segments


def _describe_cut_group(role: str, profile_id: str, member_ids: list[str]) -> str:
    """Create a descriptive engineering name for a grouped cut-list entry."""
    profile_name = get_profile(profile_id).name
    ids_lower = [mid.lower() for mid in member_ids]
    if all(mid.startswith("col_") for mid in ids_lower):
        return f"Vertical Frame Column ({profile_name})"
    if all(mid.startswith("tower_") for mid in ids_lower):
        return f"Delta Vertical Tower ({profile_name})"
    if all(mid.startswith("z_col_") for mid in ids_lower):
        return f"Z Portal Upright ({profile_name})"
    if all(mid.startswith("x_gantry") for mid in ids_lower):
        return f"X Gantry Beam ({profile_name})"
    if all(mid.startswith("y_rail") for mid in ids_lower):
        return f"Y Rail Support Extrusion ({profile_name})"
    if all(mid.startswith("arm_") for mid in ids_lower):
        return f"Delta Effector Truss Rod ({profile_name})"
    if role == "brace":
        return f"Diagonal Structural Brace ({profile_name})"
    if role == "gantry":
        return f"Gantry Beam ({profile_name})"
    if role == "rail":
        return f"Rail Support Beam ({profile_name})"
    if role == "tower":
        return f"Vertical Tower ({profile_name})"
    return f"Frame Member ({profile_name})"


def _compute_net_cut_length_mm(
    assembly: AssemblyModel,
    seg: _MergedSegment,
    centerline_mm: float,
) -> float:
    """Compute non-overlapping buildable butt-joint cut length (mm) for a merged member."""
    spec = assembly.spec
    role = seg.canonical_role
    ids_lower = [mid.lower() for mid in seg.member_ids]
    # Vertical columns/towers, diagonal braces, and delta arms keep full length
    if (
        role in ("brace", "tower")
        or all(mid.startswith(("col_", "tower_", "z_col_", "arm_")) for mid in ids_lower)
    ):
        return round(centerline_mm, 1)

    upright_prof = get_profile(spec.frame.effective_upright_profile)
    upright_w_mm = upright_prof.width_m * 1000.0
    if spec.architecture == "corexy" and role in ("frame", "gantry", "rail"):
        return round(max(centerline_mm - 2.0 * upright_w_mm, 10.0), 1)
    if spec.architecture == "cartesian_bedslinger" and role in ("gantry",):
        return round(max(centerline_mm - 2.0 * upright_w_mm, 10.0), 1)
    return round(centerline_mm, 1)


def _build_cut_list(
    assembly: AssemblyModel,
    clip_corner_overlaps: bool = False,
) -> list[CutListItem]:
    """Generate sorted, deduplicated manufacturable CutListItem list."""
    merged_segments = _merge_collinear_members(assembly)
    grouped: dict[tuple[str, str, float, float, str], list[_MergedSegment]] = defaultdict(list)

    for seg in merged_segments:
        centerline_mm = round(seg.length_m * 1000.0, 1)
        net_mm = _compute_net_cut_length_mm(assembly, seg, centerline_mm)
        length_mm = net_mm if clip_corner_overlaps else centerline_mm
        key = (seg.profile_id, seg.material_id, length_mm, net_mm, seg.canonical_role)
        grouped[key].append(seg)

    # Sort by role (asc), profile_id (asc), length_mm (desc), material_id (asc)
    sorted_keys = sorted(
        grouped.keys(),
        key=lambda k: (k[4], k[0], -k[2], k[1]),
    )

    cut_list: list[CutListItem] = []
    for idx, key in enumerate(sorted_keys, start=1):
        profile_id, material_id, length_mm, net_mm, role = key
        segs = grouped[key]
        qty = len(segs)
        profile = get_profile(profile_id)
        material = get_material(material_id)

        length_m = length_mm * 1e-3
        unit_mass_kg = profile.area_m2 * length_m * material.density_kg_m3
        unit_mass_g = unit_mass_kg * 1000.0
        total_mass_g = unit_mass_g * qty

        mfg_mult = _PROFILE_MFG_MULTIPLIER.get(profile.shape, 1.8)
        unit_cost_usd = unit_mass_kg * material.cost_per_kg_usd * mfg_mult + _CUT_FEE_USD
        total_cost_usd = unit_cost_usd * qty

        all_member_ids = [mid for s in segs for mid in s.member_ids]
        name = _describe_cut_group(role, profile_id, all_member_ids)
        if not clip_corner_overlaps and net_mm < length_mm - 0.1:
            notes = (
                f"Saw cut ±0.5 mm (net butt-joint {net_mm:.1f} mm); "
                f"from {len(all_member_ids)} FEA segment(s)"
            )
        else:
            notes = f"Saw cut ±0.5 mm; from {len(all_member_ids)} FEA segment(s)"

        cut_list.append(
            CutListItem(
                part_code=f"CUT-{idx:02d}",
                name=name,
                profile_id=profile_id,
                material_id=material_id,
                role=role,
                length_mm=round(length_mm, 1),
                quantity=qty,
                total_length_mm=round(length_mm * qty, 1),
                unit_mass_g=round(unit_mass_g, 1),
                total_mass_g=round(total_mass_g, 1),
                unit_cost_usd=round(unit_cost_usd, 2),
                total_cost_usd=round(total_cost_usd, 2),
                notes=notes,
                net_length_mm=round(net_mm, 1),
            )
        )

    return cut_list


def _nest_stock_bars(
    cut_list: list[CutListItem],
    stock_length_mm: float = 1000.0,
    kerf_mm: float = 3.0,
) -> list[StockNestingSummary]:
    """Perform 1D First-Fit Decreasing (FFD) bin packing per (profile_id, material_id)."""
    cuts_by_stock: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in cut_list:
        cuts_by_stock[(item.profile_id, item.material_id)].extend(
            [item.length_mm] * item.quantity
        )

    summaries: list[StockNestingSummary] = []
    for (profile_id, material_id) in sorted(cuts_by_stock.keys()):
        cuts = sorted(cuts_by_stock[(profile_id, material_id)], reverse=True)
        if not cuts:
            continue

        max_cut = cuts[0]
        eff_stock = float(stock_length_mm)
        if max_cut > eff_stock:
            if max_cut <= 1000.0:
                eff_stock = 1000.0
            elif max_cut <= 2000.0:
                eff_stock = 2000.0
            else:
                eff_stock = math.ceil(max_cut / 500.0) * 500.0

        # FFD packing: each bin tracks cuts and consumed length (cuts + inter-cut kerfs)
        bins: list[list[float]] = []
        bin_used: list[float] = []

        for cut in cuts:
            placed = False
            for b_idx, used in enumerate(bin_used):
                needed = cut + kerf_mm
                if used + needed <= eff_stock + 1e-6:
                    bins[b_idx].append(round(cut, 1))
                    bin_used[b_idx] = used + needed
                    placed = True
                    break
            if not placed:
                bins.append([round(cut, 1)])
                bin_used.append(cut)

        bars_required = len(bins)
        used_length_mm = sum(cuts)
        total_stock_mm = bars_required * eff_stock
        waste_length_mm = max(0.0, total_stock_mm - used_length_mm)
        waste_pct = (waste_length_mm / total_stock_mm * 100.0) if total_stock_mm > 0 else 0.0

        summaries.append(
            StockNestingSummary(
                profile_id=profile_id,
                material_id=material_id,
                stock_length_mm=round(eff_stock, 1),
                bars_required=bars_required,
                used_length_mm=round(used_length_mm, 1),
                waste_length_mm=round(waste_length_mm, 1),
                waste_pct=round(waste_pct, 2),
                bar_cuts_mm=bins,
            )
        )

    return summaries


def _normalize_z_drive(z_drive: str) -> str:
    """Normalize schema Z-drive identifiers and common aliases."""
    mapping = {
        "belted_quad_z": "belted_quad_z",
        "quad_belted": "belted_quad_z",
        "belted_z": "belted_z",
        "leadscrew_triple": "leadscrew_triple",
        "triple_leadscrew": "leadscrew_triple",
        "leadscrew_dual": "leadscrew_dual",
        "dual_leadscrew": "leadscrew_dual",
        "leadscrew_dual_belt_sync": "leadscrew_dual_belt_sync",
        "dual_leadscrew_belt_sync": "leadscrew_dual_belt_sync",
        "leadscrew_single": "leadscrew_single",
        "single_leadscrew": "leadscrew_single",
    }
    return mapping.get(z_drive, z_drive)


def _z_motor_count(architecture: str, z_drive: str) -> int:
    if architecture == "delta":
        return 0
    norm = _normalize_z_drive(z_drive)
    return {
        "belted_quad_z": 4,
        "belted_z": 2,
        "leadscrew_triple": 3,
        "leadscrew_dual": 2,
        "leadscrew_dual_belt_sync": 1,
        "leadscrew_single": 1,
    }.get(norm, 2)


def _rail_hw_info(profile_id: str, block_type: str = "H") -> tuple[str, float, float, float]:
    bt = "C" if str(block_type).upper().strip() == "C" else "H"
    if profile_id in _RAIL_HARDWARE:
        series, rail_usd_m, car_usd, mass_c_g, mass_h_g = _RAIL_HARDWARE[profile_id]
        car_mass_g = float(mass_h_g if bt == "H" else mass_c_g)
        if profile_id.startswith("steel_rod_"):
            car_name = f"{series} Linear Ball Bearing"
        elif profile_id == "hgh15_rail":
            car_name = "HGH15CA Heavy Linear Carriage"
        else:
            car_name = f"{series}{bt} Linear Rail Carriage"
        return (car_name, rail_usd_m, car_usd, car_mass_g)
    prof = get_profile(profile_id)
    return (f"{prof.name} Carriage", 35.0, 15.0, 60.0)


def _add_bom_item(
    items: list[BOMLineItem],
    sku: str,
    category: str,
    name: str,
    specification: str,
    quantity: int | float,
    unit: str,
    unit_cost_usd: float,
    unit_mass_g: float,
) -> None:
    total_cost = round(float(quantity) * float(unit_cost_usd), 2)
    total_mass = round(float(quantity) * float(unit_mass_g), 1)
    items.append(
        BOMLineItem(
            sku=sku,
            category=category,
            name=name,
            specification=specification,
            quantity=quantity,
            unit=unit,
            unit_cost_usd=round(float(unit_cost_usd), 2),
            total_cost_usd=total_cost,
            total_mass_g=total_mass,
        )
    )


def generate_bom(
    spec_or_assembly: PrinterSpec | AssemblyModel,
    stock_length_mm: float = 1000.0,
    kerf_mm: float = 3.0,
    clip_corner_overlaps: bool = False,
) -> PrinterBOM:
    """Generate a manufacturable cut-list, 1D stock nesting plan, and complete hardware BOM.

    Parameters
    ----------
    spec_or_assembly:
        A declarative ``PrinterSpec`` or an already-built ``AssemblyModel``.
    stock_length_mm:
        Standard raw bar length in mm for 1D stock nesting (default 1000.0 mm).
    kerf_mm:
        Saw blade kerf allowance in mm between cuts (default 3.0 mm).
    clip_corner_overlaps:
        When True, shorten horizontal extrusions to net non-overlapping butt-joint cut
        lengths; when False (default), preserve FEA centerline spans in ``length_mm``
        while still reporting exact ``net_length_mm`` on every ``CutListItem``.
    """
    if stock_length_mm <= 0:
        raise ValueError("stock_length_mm must be positive")
    if kerf_mm < 0:
        raise ValueError("kerf_mm must be non-negative")

    if isinstance(spec_or_assembly, AssemblyModel):
        assembly = spec_or_assembly
        spec = assembly.spec
    else:
        spec = spec_or_assembly
        assembly = build_assembly(spec)

    cut_list = _build_cut_list(assembly, clip_corner_overlaps=clip_corner_overlaps)
    stock_nesting = _nest_stock_bars(
        cut_list, stock_length_mm=stock_length_mm, kerf_mm=kerf_mm
    )

    items: list[BOMLineItem] = []

    w = spec.frame.width_mm
    d = spec.frame.depth_mm
    h = spec.frame.height_mm
    bx, by, bz = spec.kinematics.build_volume_mm
    arch = spec.architecture

    # ------------------------------------------------------------------
    # 1. FRAME (structural extrusions, tubes, braces, & cavity filling)
    # ------------------------------------------------------------------
    for c in cut_list:
        prof = get_profile(c.profile_id)
        mat = get_material(c.material_id)
        items.append(
            BOMLineItem(
                sku=f"FRM-{c.part_code}",
                category="frame",
                name=c.name,
                specification=f"{prof.name}, {mat.name}, L={c.length_mm:.1f} mm",
                quantity=c.quantity,
                unit="pcs",
                unit_cost_usd=c.unit_cost_usd,
                total_cost_usd=c.total_cost_usd,
                total_mass_g=c.total_mass_g,
            )
        )

    fill_id = spec.frame.extrusion_filling
    if fill_id != "none" and assembly.extrusion_fill_mass_kg > 0.0:
        fill_spec = EXTRUSION_FILLINGS.get(fill_id, EXTRUSION_FILLINGS["none"])
        fill_sku_id = fill_id.upper().replace("_", "-")
        fill_qty_kg = round(float(assembly.extrusion_fill_mass_kg), 4)
        fill_cost_per_kg = float(fill_spec["cost_per_kg_usd"])
        fill_mode = "Bonded composite core" if fill_spec["bonded"] else "Unbonded granular/foam damper"
        _add_bom_item(
            items,
            sku=f"EXT-FILL-{fill_sku_id}",
            category="frame",
            name=f"Frame Extrusion Cavity Filling — {fill_spec['name']}",
            specification=(
                f"{fill_mode}, rho={fill_spec['density_kg_m3']:.0f} kg/m^3, "
                f"zeta={assembly.frame_damping_ratio:.4f}"
            ),
            quantity=fill_qty_kg,
            unit="kg",
            unit_cost_usd=fill_cost_per_kg,
            unit_mass_g=1000.0,
        )

    if arch == "corexy" and spec.frame.top_brace != "none" and assembly.top_brace_active:
        tb_id = spec.frame.top_brace
        tb_info = TOP_BRACE_CATALOG.get(tb_id, TOP_BRACE_CATALOG["none"])
        tb_dims = top_brace_dimensions_mm(spec)
        tb_mat = (
            "5.0 mm Al 5052-H32 CNC plate"
            if tb_info.get("kind") == "cnc_plate"
            else "25x5 mm Al 6061-T6 flat bar kit"
        )
        _add_bom_item(
            items,
            sku=f"FRM-TOP-{tb_id.upper().replace('_', '-')}",
            category="frame",
            name=f"Top Frame Brace — {tb_info['label']}",
            specification=(
                f"{tb_mat}, {tb_dims['cutout_w_mm']:.0f}x{tb_dims['cutout_d_mm']:.0f} mm open center"
            ),
            quantity=1,
            unit="pcs" if tb_info.get("kind") == "cnc_plate" else "kit",
            unit_cost_usd=float(assembly.top_brace_cost_usd),
            unit_mass_g=float(assembly.top_brace_mass_kg) * 1000.0,
        )

    if arch == "corexy" and spec.frame.front_brace != "none" and assembly.front_brace_active:
        fb_id = spec.frame.front_brace
        fb_info = FRONT_BRACE_CATALOG.get(fb_id, FRONT_BRACE_CATALOG["none"])
        fb_dims = front_brace_dimensions_mm(spec)
        fb_mat = (
            "5.0 mm Al 5052-H32 CNC plate"
            if fb_info.get("kind") == "cnc_plate"
            else "25x5 mm Al 6061-T6 flat bar kit"
        )
        _add_bom_item(
            items,
            sku=f"FRM-FRONT-{fb_id.upper().replace('_', '-')}",
            category="frame",
            name=f"Front Frame Brace — {fb_info['label']}",
            specification=(
                f"{fb_mat}, {fb_dims['cutout_w_mm']:.0f}x{fb_dims['cutout_h_mm']:.0f} mm open door"
            ),
            quantity=1,
            unit="pcs" if fb_id == "cnc_portal_frame" else "kit",
            unit_cost_usd=float(assembly.front_brace_cost_usd),
            unit_mass_g=float(assembly.front_brace_mass_kg) * 1000.0,
        )

    bkt_id = spec.gantry.bracket_material
    bkt_info = GANTRY_BRACKET_CATALOG.get(bkt_id, GANTRY_BRACKET_CATALOG["abs_printed"])
    bkt_sku_id = bkt_id.upper().replace("_", "-")
    bkt_kit_mass_g = round(160.0 + 3.0 * float(bkt_info["carriage_mass_delta_g"]), 1)
    _add_bom_item(
        items,
        sku=f"GNT-BKT-{bkt_sku_id}",
        category="gantry",
        name=f"XY Gantry Bracket & Motor Mount Kit ({bkt_info['short_name']})",
        specification=(
            f"{bkt_info['name']}, E={bkt_info['youngs_modulus_gpa']:.1f} GPa, "
            f"HDT {bkt_info['hdt_c']:.0f} C, creep limit {bkt_info['creep_tension_limit_n']:.0f} N"
        ),
        quantity=1,
        unit="kit",
        unit_cost_usd=float(bkt_info["kit_cost_usd"]),
        unit_mass_g=bkt_kit_mass_g,
    )
    if spec.gantry.effective_cf_rail_mounting == "two_wall_clamped_inserts":
        insert_mass_g = round(
            assembly.x_beam_insert_mass_kg * 1000.0
            if assembly.x_beam_insert_mass_kg > 0.0
            else (4.0 + 18.0 * (w * 1e-3)),
            1,
        )
        _add_bom_item(
            items,
            sku="GNT-CF-INSERTS",
            category="gantry",
            name="Carbonara 2-Wall CF X-Beam Internal Insert & Ti Bolt Kit",
            specification=(
                f"100%-infill annealed CF/GF internal rail inserts + XY-joint end inserts, "
                f"M5 hex standoffs & Grade 5 Ti M3x28 clamped hardware (L={w:.0f} mm)"
            ),
            quantity=1,
            unit="kit",
            unit_cost_usd=18.00,
            unit_mass_g=insert_mass_g,
        )

    # ------------------------------------------------------------------
    # 2. MOTION (linear rails, carriages, belts, pulleys, idlers, Z-drive)
    # ------------------------------------------------------------------
    rail_mat = get_material(spec.gantry.rail_material)
    x_rail_prof = get_profile(spec.gantry.x_rail_profile)
    y_rail_prof = get_profile(spec.gantry.y_rail_profile)
    z_rail_prof_id = (spec.bed.z_rail_profile or "").strip() or y_rail_prof.id

    # Track total linear rail length (mm) and carriage count for fasteners
    total_bolted_rail_mm = 0.0
    total_carriages = 0

    def _add_rail_and_carriage(
        sku_prefix: str,
        axis_label: str,
        rail_prof_id: str,
        rail_qty: int,
        rail_len_mm: float,
        carriage_qty: int,
        mount_note: str = "",
        block_type: str = "H",
    ) -> None:
        nonlocal total_bolted_rail_mm, total_carriages
        prof = get_profile(rail_prof_id)
        car_name, rail_usd_m, car_usd, car_mass_g = _rail_hw_info(
            rail_prof_id, block_type=block_type
        )
        rail_len_m = rail_len_mm * 1e-3
        rail_unit_mass_g = prof.area_m2 * rail_len_m * rail_mat.density_kg_m3 * 1000.0
        is_rod = rail_prof_id.startswith("steel_rod_") or prof.shape == "rod"
        rail_unit_cost = rail_len_m * rail_usd_m + (3.0 if is_rod else 4.0)

        spec_str = f"{rail_mat.name}, L={rail_len_mm:.0f} mm"
        if mount_note:
            spec_str += f", {mount_note}"

        if is_rod:
            rod_diam_mm = round(prof.width_m * 1e3)
            rod_spec_str = (
                f"{rail_mat.name}, Ø{rod_diam_mm:.0f}mm h6 chrome-plated shaft, L={rail_len_mm:.0f} mm"
            )
            if mount_note:
                rod_spec_str += f", {mount_note}"
            _add_bom_item(
                items,
                sku=f"MOT-{sku_prefix}-ROD",
                category="motion",
                name=f"{axis_label} Hardened Chrome Steel Smooth Rod ({prof.name})",
                specification=rod_spec_str,
                quantity=rail_qty,
                unit="pcs",
                unit_cost_usd=rail_unit_cost,
                unit_mass_g=rail_unit_mass_g,
            )
            _add_bom_item(
                items,
                sku=f"MOT-{sku_prefix}-BRG",
                category="motion",
                name=f"{axis_label} {car_name}",
                specification=f"Linear ball bearing for Ø{rod_diam_mm:.0f}mm smooth rod ({prof.id})",
                quantity=carriage_qty,
                unit="pcs",
                unit_cost_usd=car_usd,
                unit_mass_g=car_mass_g,
            )
        else:
            _add_bom_item(
                items,
                sku=f"MOT-{sku_prefix}-RAIL",
                category="motion",
                name=f"{axis_label} Linear Rail ({prof.name})",
                specification=spec_str,
                quantity=rail_qty,
                unit="pcs",
                unit_cost_usd=rail_unit_cost,
                unit_mass_g=rail_unit_mass_g,
            )
            _add_bom_item(
                items,
                sku=f"MOT-{sku_prefix}-CAR",
                category="motion",
                name=f"{axis_label} {car_name}",
                specification=f"Preloaded block for {prof.id}",
                quantity=carriage_qty,
                unit="pcs",
                unit_cost_usd=car_usd,
                unit_mass_g=car_mass_g,
            )
            if prof.shape == "rail":
                total_bolted_rail_mm += rail_len_mm * rail_qty
        total_carriages += carriage_qty

    z_norm = _normalize_z_drive(spec.bed.z_drive)
    x_mount_id = spec.gantry.effective_x_rail_mount
    x_mount_note = {
        "front": "front-face mount (Voron standard)",
        "top": "top-face mount (RatRig/Annex)",
        "bottom": "bottom under-slung mount",
        "dual_top_bottom": "dual top+bottom I-beam mount",
    }.get(x_mount_id, f"{x_mount_id} mount")
    y_mount_note = {
        "top": "top-face mount (Voron standard)",
        "inner": "inner-side mount (Monolith/VzBot)",
        "bottom": "bottom-face mount (inverted)",
    }.get(spec.gantry.y_rail_mount, f"{spec.gantry.y_rail_mount} mount")

    if arch == "corexy":
        x_rail_len = round(min(w - 40.0, max(bx + 60.0, w - 60.0)), 0)
        y_rail_len = round(min(d - 40.0, max(by + 60.0, d - 60.0)), 0)
        z_rail_len = round(min(h - 60.0, max(bz + 80.0, h - 100.0)), 0)
        x_rail_qty = 2 if spec.gantry.effective_dual_x_rails else 1
        z_rail_qty = {
            "belted_quad_z": 4,
            "leadscrew_triple": 3,
            "leadscrew_dual": 2,
            "leadscrew_single": 2,
            "belted_z": 2,
        }.get(z_norm, 4)

        _add_rail_and_carriage(
            "X",
            "X-Axis",
            x_rail_prof.id,
            x_rail_qty,
            x_rail_len,
            x_rail_qty,
            mount_note=x_mount_note,
            block_type=spec.gantry.rail_block_type,
        )
        _add_rail_and_carriage(
            "Y",
            "Y-Axis",
            y_rail_prof.id,
            2,
            y_rail_len,
            2,
            mount_note=y_mount_note,
            block_type=spec.gantry.rail_block_type,
        )
        if z_norm == "leadscrew_dual_belt_sync":
            z_rod_prof_id = (
                spec.bed.z_rail_profile
                if spec.bed.z_rail_profile.startswith("steel_rod")
                else "steel_rod_10mm"
            )
            z_rod_prof = get_profile(z_rod_prof_id)
            rod_diam_mm = round(z_rod_prof.width_m * 1e3)
            z_rod_len_m = z_rail_len * 1e-3
            rod_unit_mass_g = z_rod_prof.area_m2 * z_rod_len_m * rail_mat.density_kg_m3 * 1000.0
            _, rod_usd_m, _, _ = _rail_hw_info(z_rod_prof_id)
            lmk_info = {
                8: ("LMK8LUU", 6.50, 44.0),
                10: ("LMK10LUU", 8.50, 68.0),
                12: ("LMK12LUU", 11.00, 96.0),
            }.get(int(rod_diam_mm), ("LMK10LUU", 8.50, 68.0))
            _add_bom_item(
                items,
                sku="MOT-Z-ROD",
                category="motion",
                name=f"Z-Axis Hardened Chrome Steel Smooth Rod ({z_rod_prof.name})",
                specification=f"{rail_mat.name}, Ø{rod_diam_mm:.0f}mm h6 chrome-plated shaft, L={z_rail_len:.0f} mm",
                quantity=4,
                unit="pcs",
                unit_cost_usd=z_rod_len_m * rod_usd_m + 3.0,
                unit_mass_g=rod_unit_mass_g,
            )
            _add_bom_item(
                items,
                sku="MOT-Z-BRG",
                category="motion",
                name=f"Z-Axis {lmk_info[0]} Flanged Linear Bearing",
                specification=f"Square-flanged double-length linear ball bushing for Ø{rod_diam_mm:.0f}mm smooth rod",
                quantity=4,
                unit="pcs",
                unit_cost_usd=lmk_info[1],
                unit_mass_g=lmk_info[2],
            )
            total_carriages += 4
        elif z_rail_prof_id != "none":
            _add_rail_and_carriage("Z", "Z-Axis", z_rail_prof_id, z_rail_qty, z_rail_len, z_rail_qty)

    elif arch == "cartesian_bedslinger":
        x_rail_len = round(min(w - 30.0, max(bx + 60.0, w - 50.0)), 0)
        y_rail_len = round(min(d - 30.0, max(by + 80.0, d - 50.0)), 0)
        z_rail_len = round(min(h - 50.0, max(bz + 80.0, h - 80.0)), 0)
        x_rail_qty = 2 if spec.gantry.effective_dual_x_rails else 1
        y_car_qty = 4 if y_rail_prof.shape == "rod" else 2

        _add_rail_and_carriage(
            "X",
            "X-Axis",
            x_rail_prof.id,
            x_rail_qty,
            x_rail_len,
            x_rail_qty,
            mount_note=x_mount_note,
            block_type=spec.gantry.rail_block_type,
        )
        _add_rail_and_carriage(
            "Y",
            "Y-Bed",
            y_rail_prof.id,
            2,
            y_rail_len,
            y_car_qty,
            block_type=spec.gantry.rail_block_type,
        )
        if z_rail_prof_id != "none":
            _add_rail_and_carriage("Z", "Z-Axis", z_rail_prof_id, 2, z_rail_len, 2)

    elif arch == "delta":
        tower_rail_len = round(min(h - 60.0, max(bz + 180.0, h - 160.0)), 0)
        if z_rail_prof_id != "none":
            _add_rail_and_carriage("TWR", "Delta Tower", z_rail_prof_id, 3, tower_rail_len, 3)

    # Drive belts, pulleys, idlers (architecture-specific exact formulas)
    belt = spec.gantry.belt
    is_awd = arch == "corexy" and spec.gantry.drive_mode == "awd"
    belt_cost_m = _BELT_COST_USD_PER_M.get(
        (belt.type, belt.core),
        float(belt.catalog_entry.get("cost_usd_per_m", 6.0)),
    )
    belt_mass_g_m = _BELT_MASS_G_PER_M.get(
        belt.type, round(belt.linear_mass_kg_m * 1000.0, 2)
    )
    belt_width_hw_scale = max(1.0, round(0.7 + 0.05 * belt.hardware_width_mm, 2))

    if arch == "corexy":
        if is_awd:
            # AWD CoreXY uses dual shorter loops routing around 4 corner motors & 12 idlers
            drive_belt_m = round(2.0 * (2.0 * (w + d) + 500.0) / 1000.0, 2)
            pulley_qty = 4
            idler_qty = 12
        else:
            drive_belt_m = round(2.0 * (2.0 * (w + d) + 400.0) / 1000.0, 2)
            pulley_qty = 2
            idler_qty = 8
    elif arch == "cartesian_bedslinger":
        drive_belt_m = round((2.0 * w + 150.0 + 2.0 * d + 150.0) / 1000.0, 2)
        pulley_qty = 2
        idler_qty = 2
    elif arch == "delta":
        drive_belt_m = round(3.0 * (2.0 * h + 120.0) / 1000.0, 2)
        pulley_qty = 3
        idler_qty = 3
    else:
        drive_belt_m = round(2.0 * (w + d) / 1000.0, 2)
        pulley_qty = 2
        idler_qty = 4

    belt_layout_prefix = "AWD dual-loop, " if is_awd else ""
    _add_bom_item(
        items,
        sku="MOT-BELT-XY",
        category="motion",
        name=f"Drive Belt ({belt.type.upper()}, {belt.core} core)",
        specification=f"{belt_layout_prefix}Pitch {belt.pitch_mm:.1f}mm, pretension {belt.pretension_n:.0f}N",
        quantity=drive_belt_m,
        unit="m",
        unit_cost_usd=belt_cost_m,
        unit_mass_g=belt_mass_g_m,
    )
    _add_bom_item(
        items,
        sku="MOT-PULLEY-DRV",
        category="motion",
        name=f"GT2 {belt.pulley_teeth}T Drive Pulley",
        specification=f"5mm bore, aluminum, matches {belt.type}",
        quantity=pulley_qty,
        unit="pcs",
        unit_cost_usd=4.50 * belt_width_hw_scale,
        unit_mass_g=12.0 * belt_width_hw_scale,
    )
    _add_bom_item(
        items,
        sku="MOT-IDLER",
        category="motion",
        name=f"GT2 {belt.pulley_teeth}T Flange Idler Pulley",
        specification=f"Dual-bearing idler for {belt.type}",
        quantity=idler_qty,
        unit="pcs",
        unit_cost_usd=3.50 * belt_width_hw_scale,
        unit_mass_g=10.0 * belt_width_hw_scale,
    )

    ds_qty = 0
    if spec.gantry.double_shear_pulleys:
        ds_qty = 4 if is_awd else (3 if arch == "delta" else 2)
        _add_bom_item(
            items,
            sku="MOT-DBL-SHEAR",
            category="motion",
            name="Double-Shear Bearing & Shaft Support Kit",
            specification=(
                f"F695-2RS/625 flanged bearings, extended 5mm shaft support & "
                f"CNC/rigid double-shear motor mount ({'AWD 4-corner' if is_awd else f'{ds_qty}-corner'})"
            ),
            quantity=ds_qty,
            unit="sets",
            unit_cost_usd=14.00,
            unit_mass_g=35.0,
        )

    if arch == "delta":
        arm_len_mm = spec.kinematics.delta_arm_length_mm
        arm_mass_g = (arm_len_mm * 1e-3) * 28.0 + 10.0  # CF tube + rod ends
        _add_bom_item(
            items,
            sku="MOT-DELTA-RODS",
            category="motion",
            name="Carbon Fiber Delta Parallel Rod Assembly",
            specification=f"6mm OD carbon tube w/ ball joints, L={arm_len_mm:.1f} mm",
            quantity=6,
            unit="pcs",
            unit_cost_usd=11.50,
            unit_mass_g=arm_mass_g,
        )

    # Z-drive hardware (CoreXY & Cartesian)
    if arch != "delta":
        z_mech_id = spec.bed.effective_z_leadscrew_type
        z_mech = Z_DRIVE_MECHANISM_CATALOG.get(z_mech_id, Z_DRIVE_MECHANISM_CATALOG["tr8x4_pom"])
        z_kind = str(z_mech["kind"])

        if z_norm in ("belted_quad_z", "belted_z") or z_kind == "belt":
            z_axes = (
                4
                if z_norm == "belted_quad_z"
                else {
                    "belted_z": 2,
                    "leadscrew_triple": 3,
                    "leadscrew_dual": 2,
                    "leadscrew_dual_belt_sync": 2,
                    "leadscrew_single": 1,
                }.get(z_norm, 2)
            )
            belt_w_mm = int(round(float(z_mech.get("diameter_mm", 6.0)))) if z_kind == "belt" else 6
            belt_mass_g = _BELT_MASS_G_PER_M.get(
                f"gt2_{belt_w_mm}mm", round(float(belt_w_mm) * 2.0, 1)
            )
            belt_cost_usd = 4.50 if belt_w_mm <= 6 else 6.80
            z_belt_m = round(z_axes * (h + 120.0) / 1000.0, 2)
            z_belt_label = (
                str(z_mech["short_name"])
                if z_kind == "belt"
                else f"GT2 {belt_w_mm}mm Z-Belt"
            )
            _add_bom_item(
                items,
                sku="MOT-Z-BELT",
                category="motion",
                name=f"Z-Drive GT2 {belt_w_mm}mm Belt",
                specification=f"{z_axes}x vertical Z drive belts (fiberglass core, {z_belt_label})",
                quantity=z_belt_m,
                unit="m",
                unit_cost_usd=belt_cost_usd,
                unit_mass_g=belt_mass_g,
            )
            if z_mech_id == "gt2_9mm_planetary_z":
                gear_name = "5:1 Planetary Z-Drive Gearbox & 9mm Pulley Kit"
                gear_spec = "5:1 planetary reduction gearbox, 20T 9mm GT2 drive pulley, idlers & shaft kit"
                gear_cost = 38.00
                gear_mass = 190.0
            elif z_mech_id == "gt2_9mm_worm_z":
                gear_name = "20:1 Self-Locking Worm Z-Drive Gearbox & 9mm Pulley Kit"
                gear_spec = "Self-locking 20:1 worm reduction gearbox (anti-drop), 20T 9mm GT2 pulley & idlers"
                gear_cost = 48.00
                gear_mass = 240.0
            elif z_mech_id == "kevinakasam_belted_z":
                gear_name = "KevinAkaSam 80T/16T (5:1) Belted-Z Pulley & Shaft Kit"
                gear_spec = "80T + 16T GT2 6mm pulleys, 188mm closed-loop belt, 5mm shaft & bearings"
                gear_cost = 16.50
                gear_mass = 85.0
            else:
                gear_name = (
                    "Z-Drive 80T/16T Reduction Pulley & Shaft Kit"
                    if belt_w_mm <= 6
                    else f"Z-Drive 80T/16T ({belt_w_mm}mm) Reduction Pulley & Shaft Kit"
                )
                gear_spec = f"80T + 16T GT2 ({belt_w_mm}mm) pulleys, 188mm closed loop belt, 5mm shaft & bearings"
                gear_cost = 16.50 if belt_w_mm <= 6 else 22.00
                gear_mass = 85.0 if belt_w_mm <= 6 else 105.0
            _add_bom_item(
                items,
                sku="MOT-Z-GEAR",
                category="motion",
                name=gear_name,
                specification=gear_spec,
                quantity=z_axes,
                unit="sets",
                unit_cost_usd=gear_cost,
                unit_mass_g=gear_mass,
            )
            if z_mech_id == "kevinakasam_belted_z":
                keybak_qty = 1 if z_axes <= 2 else 2
                _add_bom_item(
                    items,
                    sku="MOT-Z-KEYBAK",
                    category="motion",
                    name="Keybak Retractable Z-Axis Counterbalance Reel",
                    specification="Constant-force retractable Z tether preventing unpowered bed/gantry drop",
                    quantity=keybak_qty,
                    unit="pcs",
                    unit_cost_usd=14.00,
                    unit_mass_g=85.0,
                )
            if z_norm == "leadscrew_dual_belt_sync":
                z_sync_belt_mm = round(2.0 * w + d + 200.0, 0)
                _add_bom_item(
                    items,
                    sku="MOT-Z-SYNC-BELT",
                    category="motion",
                    name="Z-Axis Closed-Loop GT2 Z-Sync Belt",
                    specification=f"Closed-loop 2GT {belt_w_mm}mm fiberglass belt, L={z_sync_belt_mm:.0f} mm",
                    quantity=1,
                    unit="pcs",
                    unit_cost_usd=7.50,
                    unit_mass_g=round((z_sync_belt_mm * 1e-3) * belt_mass_g, 1),
                )
                _add_bom_item(
                    items,
                    sku="MOT-Z-SYNC-PULLEY",
                    category="motion",
                    name="Z-Sync GT2 20T Pulley",
                    specification="1x 5mm bore (Z stepper) + 2x 5mm/8mm bore (belted Z shafts), aluminum",
                    quantity=3,
                    unit="pcs",
                    unit_cost_usd=4.20,
                    unit_mass_g=12.0,
                )
        else:
            ls_qty = {
                "leadscrew_triple": 3,
                "leadscrew_dual": 2,
                "leadscrew_dual_belt_sync": 2,
                "leadscrew_single": 1,
            }.get(z_norm, 2)
            ls_len_mm = round(min(h - 40.0, bz + 100.0), 0)
            if z_kind == "ballscrew":
                diam_mm = int(round(float(z_mech.get("diameter_mm", 12.0))))
                lead_mm = float(z_mech.get("lead_mm", 4.0))
                is_1605 = z_mech_id == "sfu1605_fk12" or diam_mm >= 16
                bs_code = "SFU1605" if is_1605 else "SFU1204"
                brg_code = "FK12/BK12" if is_1605 else "BK10"
                ls_label = bs_code
                ls_mass_g = round((ls_len_mm * 1e-3) * float(z_mech.get("mass_kg_per_m", 0.68)) * 1000.0, 1)
                _add_bom_item(
                    items,
                    sku="MOT-Z-LS",
                    category="motion",
                    name=f"{bs_code} Rolled Ballscrew & Ball Nut",
                    specification=f"{diam_mm}mm dia, {lead_mm:.0f}mm lead C7 ballscrew w/ recirculating ball nut, L={ls_len_mm:.0f} mm",
                    quantity=ls_qty,
                    unit="pcs",
                    unit_cost_usd=(44.00 if is_1605 else 32.00) + (ls_len_mm * 1e-3) * (32.0 if is_1605 else 24.0),
                    unit_mass_g=ls_mass_g,
                )
                _add_bom_item(
                    items,
                    sku="MOT-Z-BK",
                    category="motion",
                    name=f"{brg_code} Angular-Contact Thrust Block & Coupler",
                    specification=f"Preloaded {brg_code} angular-contact thrust bearing block + diaphragm coupler",
                    quantity=ls_qty,
                    unit="sets",
                    unit_cost_usd=26.00 if is_1605 else 18.00,
                    unit_mass_g=240.0 if is_1605 else 160.0,
                )
                if bool(z_mech.get("wobble_decoupled", False)):
                    _add_bom_item(
                        items,
                        sku="MOT-Z-WOBBLEX",
                        category="motion",
                        name="WobbleX WS12 Oldham Z-Wobble Decoupler Kit",
                        specification=f"Self-aligning hardened steel pin-and-ball Z-wobble decoupler for {bs_code} nut",
                        quantity=ls_qty,
                        unit="sets",
                        unit_cost_usd=16.00,
                        unit_mass_g=48.0,
                    )
            else:
                is_tr8x4 = abs(float(z_mech.get("lead_mm", 8.0)) - 4.0) < 0.5
                ls_label = "TR8x4" if is_tr8x4 else "TR8x8"
                ls_starts = "2-start" if is_tr8x4 else "4-start"
                ls_mass_g = (ls_len_mm * 1e-3) * 320.0  # TR8 stainless ~320 g/m
                is_brass = "brass" in z_mech_id
                nut_mat = "Machined C360 Brass" if is_brass else "POM Anti-Backlash"
                _add_bom_item(
                    items,
                    sku="MOT-Z-LS",
                    category="motion",
                    name=f"{ls_label} Stainless Leadscrew",
                    specification=f"8mm dia, 2mm pitch, {ls_starts}, L={ls_len_mm:.0f} mm",
                    quantity=ls_qty,
                    unit="pcs",
                    unit_cost_usd=11.00 + (ls_len_mm * 1e-3) * 14.0,
                    unit_mass_g=ls_mass_g,
                )
                _add_bom_item(
                    items,
                    sku="MOT-Z-NUT",
                    category="motion",
                    name=(
                        f"{ls_label} {nut_mat} Leadscrew Nut & Pillow-Block Bearing"
                        if z_norm == "leadscrew_dual_belt_sync"
                        else f"{ls_label} {nut_mat} Leadscrew Nut & 5x8mm Coupler"
                    ),
                    specification=(
                        f"{nut_mat} nut + KFL08/608 pillow-block thrust bearings"
                        if z_norm == "leadscrew_dual_belt_sync"
                        else f"{nut_mat} nut + flexible aluminum shaft coupler"
                    ),
                    quantity=ls_qty,
                    unit="sets",
                    unit_cost_usd=11.00 if is_brass else 8.50,
                    unit_mass_g=45.0 if is_brass else 35.0,
                )
                if bool(z_mech.get("wobble_decoupled", False)):
                    _add_bom_item(
                        items,
                        sku="MOT-Z-WOBBLEX",
                        category="motion",
                        name="WobbleX WS8 Oldham Z-Wobble Decoupler Kit",
                        specification=f"Self-aligning hardened steel pin-and-ball Z-wobble decoupler for {ls_label} nut",
                        quantity=ls_qty,
                        unit="sets",
                        unit_cost_usd=14.50,
                        unit_mass_g=42.0,
                    )
            if z_norm == "leadscrew_dual_belt_sync":
                z_sync_belt_mm = round(2.0 * w + d + 200.0, 0)
                _add_bom_item(
                    items,
                    sku="MOT-Z-SYNC-BELT",
                    category="motion",
                    name="Z-Axis Closed-Loop GT2 Z-Sync Belt",
                    specification=(
                        f"Closed-loop 2GT 6mm fiberglass belt, L={z_sync_belt_mm:.0f} mm "
                        f"(syncs dual {ls_label} screws to single Z stepper)"
                    ),
                    quantity=1,
                    unit="pcs",
                    unit_cost_usd=7.50,
                    unit_mass_g=round((z_sync_belt_mm * 1e-3) * _BELT_MASS_G_PER_M["gt2_6mm"], 1),
                )
                _add_bom_item(
                    items,
                    sku="MOT-Z-SYNC-PULLEY",
                    category="motion",
                    name="Z-Sync GT2 20T Pulley",
                    specification=f"1x 5mm bore (Z stepper) + 2x {'10/12mm' if z_kind == 'ballscrew' else '8mm'} bore ({ls_label}), aluminum",
                    quantity=3,
                    unit="pcs",
                    unit_cost_usd=4.20,
                    unit_mass_g=12.0,
                )

    # ------------------------------------------------------------------
    # 3. MOTORS (XY/Tower steppers + Z steppers + 1x Extruder stepper)
    # ------------------------------------------------------------------
    xy_stepper = spec.kinematics.xy_stepper
    z_stepper = spec.kinematics.z_stepper
    xy_motor_qty = 3 if arch == "delta" else (4 if is_awd else 2)
    z_motor_qty = _z_motor_count(arch, spec.bed.z_drive)

    xy_unit_cost = 12.0 + xy_stepper.holding_torque_nm * 18.0
    xy_motor_label = "Tower" if arch == "delta" else ("XY AWD" if is_awd else "XY")
    _add_bom_item(
        items,
        sku="MTR-XY",
        category="motors",
        name=f"{xy_motor_label} Stepper Motor — {xy_stepper.name}",
        specification=(
            f"{xy_stepper.holding_torque_nm:.2f} N·m, {xy_stepper.rated_current_a:.1f}A, "
            f"{xy_stepper.inductance_mh:.1f}mH, {xy_stepper.supply_voltage_v:.0f}V"
        ),
        quantity=xy_motor_qty,
        unit="pcs",
        unit_cost_usd=xy_unit_cost,
        unit_mass_g=xy_stepper.motor_mass_g,
    )

    if z_motor_qty > 0:
        z_unit_cost = 11.0 + z_stepper.holding_torque_nm * 16.0
        _add_bom_item(
            items,
            sku="MTR-Z",
            category="motors",
            name=f"Z-Axis Stepper Motor — {z_stepper.name}",
            specification=(
                f"{z_stepper.holding_torque_nm:.2f} N·m, {z_stepper.rated_current_a:.1f}A, "
                f"{z_stepper.supply_voltage_v:.0f}V"
            ),
            quantity=z_motor_qty,
            unit="pcs",
            unit_cost_usd=z_unit_cost,
            unit_mass_g=z_stepper.motor_mass_g,
        )

    is_direct = spec.toolhead.extruder.type == "direct_drive"
    _add_bom_item(
        items,
        sku="MTR-EXT",
        category="motors",
        name=(
            "NEMA14 Pancake Extruder Stepper (LDO 36STH20)"
            if is_direct
            else "NEMA17 Extruder Stepper (40mm)"
        ),
        specification=(
            "36mm round pancake, 1.0A, 10T gear"
            if is_direct
            else "42x40mm, 0.45 N·m, 1.5A"
        ),
        quantity=1,
        unit="pcs",
        unit_cost_usd=16.50 if is_direct else 14.00,
        unit_mass_g=105.0 if is_direct else 280.0,
    )

    total_motors = xy_motor_qty + z_motor_qty + 1

    # ------------------------------------------------------------------
    # 4. HOTEND & EXTRUDER & COOLING
    # ------------------------------------------------------------------
    hotend = spec.toolhead.hotend
    extruder = spec.toolhead.extruder
    block_mat = get_material(hotend.block_material)
    nozzle_mat = get_material(hotend.nozzle_material)
    hb_mat = get_material(hotend.heatbreak_material)

    hotend_cost = 38.0
    if hotend.heater_power_w >= 65.0:
        hotend_cost += 16.0
    elif hotend.heater_power_w >= 50.0:
        hotend_cost += 8.0
    if hotend.block_material == "copper_c110":
        hotend_cost += 14.0
    if hotend.heatbreak_material == "titanium_grade5":
        hotend_cost += 16.0
    if hotend.nozzle_material == "hardened_steel":
        hotend_cost += 12.0
    if hotend.melt_zone_length_mm >= 18.0:
        hotend_cost += 10.0

    _add_bom_item(
        items,
        sku="HOT-ASM",
        category="hotend",
        name=f"Hotend Assembly — {hotend.name}",
        specification=(
            f"{hotend.heater_power_w:.0f}W heater, {hotend.nozzle_diameter_mm:.2f}mm "
            f"{nozzle_mat.id} nozzle, {block_mat.id} block, {hb_mat.id} heatbreak, "
            f"{hotend.melt_zone_length_mm:.0f}mm melt zone"
        ),
        quantity=1,
        unit="pcs",
        unit_cost_usd=hotend_cost,
        unit_mass_g=float(hotend.mass_g),
    )

    _add_bom_item(
        items,
        sku="HOT-EXT",
        category="hotend",
        name=(
            "Direct-Drive Dual-Gear Extruder Assembly"
            if is_direct
            else "Bowden Dual-Gear Extruder & PTFE Tube Assembly"
        ),
        specification=(
            f"{extruder.type}, {extruder.gear_ratio:.1f}:1 ratio, "
            f"{extruder.filament_diameter_mm:.2f}mm filament, "
            f"{extruder.max_push_force_n:.0f}N max push"
        ),
        quantity=1,
        unit="pcs",
        unit_cost_usd=44.00 if is_direct else 32.00,
        unit_mass_g=130.0 if is_direct else 110.0,
    )

    part_fan_qty = 2 if hotend.part_cooling_cfm >= 10.0 else (1 if hotend.part_cooling_cfm > 0 else 0)
    if part_fan_qty > 0:
        _add_bom_item(
            items,
            sku="HOT-FAN-PART",
            category="hotend",
            name="5015 Dual-Ball-Bearing Part Cooling Blower",
            specification=f"24V radial blower, {hotend.part_cooling_cfm:.1f} CFM total system flow",
            quantity=part_fan_qty,
            unit="pcs",
            unit_cost_usd=8.50,
            unit_mass_g=28.0,
        )
    _add_bom_item(
        items,
        sku="HOT-FAN-HS",
        category="hotend",
        name="4010 Hotend Heatsink Axial Cooling Fan",
        specification=f"24V dual-ball bearing, R_th={hotend.heatsink_thermal_resistance_k_w:.1f} K/W",
        quantity=1,
        unit="pcs",
        unit_cost_usd=6.50,
        unit_mass_g=18.0,
    )

    # ------------------------------------------------------------------
    # 5. BED (build plate + heater + PEI spring steel sheet)
    # ------------------------------------------------------------------
    bed = spec.bed
    bed_x, bed_y = bed.size_mm
    bed_area_m2 = (bed_x * 1e-3) * (bed_y * 1e-3)
    plate_mat = get_material(bed.plate_material)
    plate_vol_m3 = bed_area_m2 * (bed.plate_thickness_mm * 1e-3)
    plate_mass_kg = plate_vol_m3 * plate_mat.density_kg_m3
    plate_cost = max(22.0, plate_mass_kg * plate_mat.cost_per_kg_usd * 2.2)

    _add_bom_item(
        items,
        sku="BED-PLATE",
        category="bed",
        name=f"Precision Build Plate ({plate_mat.name})",
        specification=f"{bed_x:.0f}x{bed_y:.0f}x{bed.plate_thickness_mm:.1f} mm, stress-relieved flat",
        quantity=1,
        unit="pcs",
        unit_cost_usd=plate_cost,
        unit_mass_g=plate_mass_kg * 1000.0,
    )

    heater_kind = "Silicone AC Mains" if bed.heater_power_w >= 400.0 else "PCB/Silicone 24V DC"
    heater_cost = 18.0 + bed.heater_power_w * 0.045
    _add_bom_item(
        items,
        sku="BED-HEATER",
        category="bed",
        name=f"{heater_kind} Bed Heater Pad",
        specification=f"{bed.heater_power_w:.0f}W, {bed_x - 20:.0f}x{bed_y - 20:.0f} mm w/ thermistor & thermal fuse",
        quantity=1,
        unit="pcs",
        unit_cost_usd=heater_cost,
        unit_mass_g=bed_area_m2 * 2200.0,
    )

    pei_cost = 14.0 + bed_area_m2 * 180.0
    _add_bom_item(
        items,
        sku="BED-PEI",
        category="bed",
        name="PEI Spring Steel Build Sheet & Magnetic Base",
        specification=f"{bed_x:.0f}x{bed_y:.0f} mm double-sided textured/smooth PEI + high-temp magnet",
        quantity=1,
        unit="sets",
        unit_cost_usd=pei_cost,
        unit_mass_g=bed_area_m2 * 3500.0,
    )

    bed_mount_id = bed.mount_type
    bed_mount_cfg = BED_MOUNT_CATALOG.get(bed_mount_id, BED_MOUNT_CATALOG["kinematic_3pt"])
    mount_pts = int(bed_mount_cfg.get("mount_points", 3))
    mount_cost = float(bed_mount_cfg.get("cost_usd", 25.0))
    radial_mode = str(bed_mount_cfg.get("radial_dof_mode", "maxwell_vgroove"))
    tap_label = "Tap-rated" if bool(bed_mount_cfg.get("tap_rated", False)) else "non-Tap"
    standoff_h = float(bed_mount_cfg.get("standoff_height_mm", 15.0))
    mount_mass_map = {
        "rigid_4pt": 48.0,
        "floating_4pt": 65.0,
        "silicone_4pt": 28.0,
        "springs_4pt": 36.0,
        "trident_3pt_standoffs": 42.0,
        "kinematic_3pt": 135.0,
        "whopping_pochard_kinematic": 165.0,
        "cnc_kelvin_magnetic": 195.0,
        "captive_belleville_kinematic": 225.0,
        "dual_leaf_ti_kinematic": 180.0,
        "spherical_3pt_ratrig": 145.0,
    }
    mount_mass_g = mount_mass_map.get(bed_mount_id, 95.0 if mount_pts == 3 else 50.0)
    _add_bom_item(
        items,
        sku="BED-MOUNT",
        category="bed",
        name=f"Bed Mounting Hardware ({bed_mount_cfg['name']})",
        specification=(
            f"{mount_pts}-pt ({radial_mode}), h={standoff_h:.0f}mm, {tap_label}, "
            f"k_z={float(bed_mount_cfg['k_axial_n_per_mm']):.0f} N/mm/pt"
        ),
        quantity=1,
        unit="sets",
        unit_cost_usd=mount_cost,
        unit_mass_g=mount_mass_g,
    )

    # ------------------------------------------------------------------
    # 6. ENCLOSURE / STRUCTURAL PANELS (material, thickness & mounting)
    # ------------------------------------------------------------------
    has_panels = spec.frame.enclosed_panels and spec.frame.panel_mounting != "none"
    if has_panels:
        pt = spec.frame.panel_thickness_mm
        pmat = get_material(spec.frame.panel_material)
        is_structural_panels = spec.frame.panel_mounting == "bolted_structural"
        mount_label = "Bolted Structural" if is_structural_panels else "Clip-In Enclosure"
        wall_suffix = "Structural Panel" if is_structural_panels else "Enclosure Panel"

        # Opaque structural walls pair with a clear polycarbonate hinged front viewing door
        is_transparent_panel = pmat.id in ("acrylic_pmma", "polycarbonate")
        door_mat = pmat if is_transparent_panel else get_material("polycarbonate")

        if arch == "delta":
            side_w = round(max(w, d) * 0.5 * math.sqrt(3.0), 0)
            panel_defs = [
                ("ENC-LEFT", f"Left Side {wall_suffix}", side_w, h, 1.0),
                ("ENC-RIGHT", f"Right Side {wall_suffix}", side_w, h, 1.0),
                ("ENC-REAR", f"Rear {wall_suffix}", side_w, h, 1.0),
                ("ENC-TOP", f"Top {wall_suffix}", side_w, round(side_w * 0.866, 0), 0.5),
                ("ENC-DOOR", "Front Door Enclosure Panel (w/ Hinges & Latch)", side_w, h, 1.0),
            ]
        else:
            panel_defs = [
                ("ENC-LEFT", f"Left Side {wall_suffix}", d, h, 1.0),
                ("ENC-RIGHT", f"Right Side {wall_suffix}", d, h, 1.0),
                ("ENC-REAR", f"Rear {wall_suffix}", w, h, 1.0),
                ("ENC-TOP", f"Top {wall_suffix}", w, d, 1.0),
                ("ENC-DOOR", "Front Door Enclosure Panel (w/ Hinges & Latch)", w, h, 1.0),
            ]

        for sku, pname, dim_a, dim_b, area_factor in panel_defs:
            is_door = sku == "ENC-DOOR"
            active_mat = door_mat if is_door else pmat
            area_m2 = (dim_a * 1e-3) * (dim_b * 1e-3) * area_factor
            panel_props = panel_mechanical_properties(
                active_mat,
                pt * 1e-3,
                dim_a * 1e-3,
                dim_b * 1e-3,
                spec.frame.panel_mounting,
            )
            mass_kg = area_m2 * panel_props["areal_mass_kg_m2"]
            mass_g = mass_kg * 1000.0
            extra_hw = 8.0 if is_door else 0.0
            # Sheet fabrication factor 1.45x (CNC/laser cutting + sheet waste) + per-panel setup floor
            cost = max(8.0, mass_kg * active_mat.cost_per_kg_usd * 1.45 + 3.0) + extra_hw
            if is_door:
                door_note = (
                    f"Hinged Viewing Door ({mount_label})"
                    if not is_transparent_panel
                    else f"{mount_label} / Hinged Door"
                )
                panel_spec = (
                    f"{active_mat.name}, {pt:.1f} mm, {door_note}, "
                    f"{dim_a:.0f}x{dim_b:.0f} mm"
                )
            else:
                panel_spec = (
                    f"{active_mat.name}, {pt:.1f} mm, {mount_label}, "
                    f"{dim_a:.0f}x{dim_b:.0f} mm"
                )
            _add_bom_item(
                items,
                sku=sku,
                category="enclosure",
                name=pname,
                specification=panel_spec,
                quantity=1,
                unit="pcs",
                unit_cost_usd=cost,
                unit_mass_g=mass_g,
            )

    # ------------------------------------------------------------------
    # 7. FASTENERS (corner joints, M5 frame bolts/T-nuts, M3 rail/motor HW)
    # ------------------------------------------------------------------
    if arch == "corexy":
        corner_joints = 16  # 8 outer cube corners + 4 gantry junctions + 4 reinforced corner brackets
    elif arch == "cartesian_bedslinger":
        corner_joints = 12
    elif arch == "delta":
        corner_joints = 12
    else:
        corner_joints = max(8, len(assembly.nodes) * 2)

    brace_cuts = sum(c.quantity for c in cut_list if c.role == "brace")
    m5_bolt_qty = corner_joints * 2 + brace_cuts * 2
    m5_tnut_qty = m5_bolt_qty

    # M3 linear rail mounting screws spaced every 25 mm along total bolted rail length
    m3_rail_screw_qty = int(math.ceil(total_bolted_rail_mm / 25.0)) if total_bolted_rail_mm > 0 else 0
    m3_motor_toolhead_qty = (
        total_motors * 4
        + total_carriages * 4
        + ds_qty * 4  # double-shear motor mount bearing cap fasteners
        + 16  # toolhead, extruder, bed mounts
    )

    frame_prof_label = (
        f"{spec.frame.effective_upright_profile}+{spec.frame.profile}"
        if spec.frame.upright_profile and spec.frame.upright_profile != spec.frame.profile
        else spec.frame.profile
    )

    _add_bom_item(
        items,
        sku="FST-CORNER",
        category="fasteners",
        name="Cast Aluminum Corner Bracket / Blind Joint",
        specification=f"Matches {frame_prof_label} T-slot frame junctions",
        quantity=corner_joints,
        unit="pcs",
        unit_cost_usd=1.35,
        unit_mass_g=16.0,
    )
    _add_bom_item(
        items,
        sku="FST-M5-BOLT",
        category="fasteners",
        name="M5x10mm Stainless SHCS Frame Bolts",
        specification="ISO 4762 A2-70 stainless steel for frame joints & braces",
        quantity=m5_bolt_qty,
        unit="pcs",
        unit_cost_usd=0.14,
        unit_mass_g=3.2,
    )
    _add_bom_item(
        items,
        sku="FST-M5-TNUT",
        category="fasteners",
        name="M5 Spring-Loaded Roll-In T-Nuts",
        specification=f"Nickel-plated steel T-nuts for {frame_prof_label}",
        quantity=m5_tnut_qty,
        unit="pcs",
        unit_cost_usd=0.18,
        unit_mass_g=4.0,
    )
    if m3_rail_screw_qty > 0:
        _add_bom_item(
            items,
            sku="FST-M3-RAIL",
            category="fasteners",
            name="M3x8mm Linear Rail Mounting Screws & M3 T-Nuts",
            specification="Spaced every 25 mm along total linear rail length",
            quantity=m3_rail_screw_qty,
            unit="pcs",
            unit_cost_usd=0.16,
            unit_mass_g=2.4,
        )
    _add_bom_item(
        items,
        sku="FST-M3-HW",
        category="fasteners",
        name="M3 Stepper, Carriage & Toolhead Hardware Pack",
        specification="Assorted M3x6–M3x30 SHCS, washers & heat-set inserts",
        quantity=m3_motor_toolhead_qty,
        unit="pcs",
        unit_cost_usd=0.12,
        unit_mass_g=1.8,
    )
    if has_panels:
        if spec.frame.panel_mounting == "bolted_structural":
            _add_bom_item(
                items,
                sku="FST-PANEL-STRUCT",
                category="fasteners",
                name="Structural Panel Perimeter Bolts & Roll-In T-Nuts (M5/M3)",
                specification=(
                    f"12x perimeter BHCS bolts + roll-in T-nuts per structural wall (4 walls) "
                    f"for {spec.frame.profile} shear diaphragm"
                ),
                quantity=48,
                unit="sets",
                unit_cost_usd=0.35,
                unit_mass_g=6.5,
            )
            if spec.panels_bolted_to_gantry_effective:
                _add_bom_item(
                    items,
                    sku="FST-PANEL-GANTRY",
                    category="fasteners",
                    name="Gantry Beam Structural Panel Bolts & T-Nuts",
                    specification=(
                        "M3/M5 bolts + roll-in T-nuts tying left/right/rear panels to fixed gantry extrusions"
                    ),
                    quantity=24,
                    unit="sets",
                    unit_cost_usd=0.22,
                    unit_mass_g=4.5,
                )
        else:
            _add_bom_item(
                items,
                sku="FST-PANEL-CLIP",
                category="fasteners",
                name="Clip-In Enclosure Panel Latches & Gasket Clips",
                specification="4x quick-release perimeter panel clips per wall (4 walls) w/ foam seal",
                quantity=16,
                unit="pcs",
                unit_cost_usd=0.65,
                unit_mass_g=4.5,
            )

    # ------------------------------------------------------------------
    # Totals & Summaries
    # ------------------------------------------------------------------
    category_totals_raw: dict[str, float] = defaultdict(float)
    for item in items:
        category_totals_raw[item.category] += item.total_cost_usd

    category_order = ["frame", "gantry", "motion", "motors", "hotend", "bed", "enclosure", "fasteners"]
    category_costs_usd: dict[str, float] = {}
    for cat in category_order:
        if cat in category_totals_raw:
            category_costs_usd[cat] = round(category_totals_raw[cat], 2)
    for cat, val in category_totals_raw.items():
        if cat not in category_costs_usd:
            category_costs_usd[cat] = round(val, 2)

    total_cost_usd = round(sum(category_costs_usd.values()), 2)
    total_mass_kg = round(sum(item.total_mass_g for item in items) / 1000.0, 3)
    total_fasteners = int(
        sum(item.quantity for item in items if item.category == "fasteners")
    )
    total_belt_length_m = round(
        sum(
            float(item.quantity)
            for item in items
            if item.category == "motion" and item.unit == "m"
        ),
        2,
    )
    total_extrusion_length_m = round(
        sum(c.total_length_mm for c in cut_list) / 1000.0, 3
    )

    return PrinterBOM(
        printer_name=spec.name,
        architecture=spec.architecture,
        cut_list=cut_list,
        stock_nesting=stock_nesting,
        items=items,
        category_costs_usd=category_costs_usd,
        total_cost_usd=total_cost_usd,
        total_mass_kg=total_mass_kg,
        total_fasteners=total_fasteners,
        total_belt_length_m=total_belt_length_m,
        total_extrusion_length_m=total_extrusion_length_m,
    )
