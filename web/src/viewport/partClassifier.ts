import { AdjointLens, SubsystemKey } from "../types";

export interface PartGroupClassification {
  id: string;
  label: string;
  subsystem: SubsystemKey;
}

/**
 * Mirror backend `classify_part_group` + `frame_feet` so 3D viewport raycasting
 * maps member/bay IDs to the exact same physical group IDs used by adjoint sensitivity.
 */
export function classifyPartGroup(memberId: string, role = ""): PartGroupClassification {
  const mid = (memberId || "").trim();
  if (mid.startsWith("feet_") || mid.startsWith("foot_") || mid.startsWith("ground_") || role === "foot") {
    return { id: "frame_feet", label: "Machine Feet & Benchtop Pads", subsystem: "frame" };
  }
  if (mid.startsWith("col_fl") || mid.startsWith("col_fr")) {
    return { id: "front_frame_columns", label: "Front Frame Columns", subsystem: "frame" };
  }
  if (mid.startsWith("col_rl") || mid.startsWith("col_rr")) {
    return { id: "rear_frame_columns", label: "Rear Frame Columns", subsystem: "frame" };
  }
  if (mid.startsWith("col_")) {
    return { id: "frame_columns", label: "Frame Vertical Columns", subsystem: "frame" };
  }
  if (mid.startsWith("z_col_")) {
    return { id: "z_gantry_uprights", label: "Z-Axis Gantry Uprights", subsystem: "frame" };
  }
  if (mid.startsWith("tower_") || role === "tower") {
    return { id: "delta_towers", label: "Delta Vertical Towers", subsystem: "frame" };
  }
  if (mid.startsWith("arm_")) {
    return { id: "delta_arms", label: "Delta Diagonal Arms", subsystem: "gantry" };
  }
  if (mid.startsWith("xy_joint")) {
    return { id: "xy_joints", label: "XY-Joint Brackets (CAD)", subsystem: "gantry" };
  }
  if (mid.startsWith("corner_mount_")) {
    return { id: "gantry_brackets", label: "Gantry Motor & Idler Mounts (CAD)", subsystem: "gantry" };
  }
  if (mid.startsWith("y_car_bearing_")) {
    return { id: "y_rail_carriages", label: "Y-Rail Carriage Bearings (Hertzian)", subsystem: "gantry" };
  }
  if (mid.startsWith("cad_z_joint_") || mid.startsWith("z_joint_")) {
    return { id: "z_gantry_joints", label: "Flying-Gantry Z-Joints", subsystem: "gantry" };
  }
  if (mid.startsWith("x_gantry") || mid.startsWith("x_beam")) {
    return { id: "x_gantry_beam", label: "Moving X-Gantry Beam", subsystem: "gantry" };
  }
  if (mid.startsWith("y_rail") || mid.startsWith("gantry_y")) {
    return { id: "y_axis_rail_beams", label: "Y-Axis Rail Beams", subsystem: "gantry" };
  }
  if (mid.startsWith("toolhead_mount") || mid.startsWith("nozzle_stem") || mid === "toolhead_cog_spring") {
    return { id: "toolhead_mount", label: "Toolhead & Nozzle Mount", subsystem: "toolhead" };
  }
  if (mid === "toolhead_cog" || mid === "toolhead_assembly") {
    return { id: "toolhead_assembly", label: "Toolhead Moving Mass (CoG)", subsystem: "toolhead" };
  }
  if (mid === "bed_plate") {
    return { id: "bed_plate", label: "Heated Bed Plate", subsystem: "bed" };
  }
  if (mid.startsWith("bed_") || mid.startsWith("z_bed") || mid.startsWith("trident_bed_arm_")) {
    return { id: "bed_support", label: "Bed Support & Z Arms", subsystem: "bed" };
  }
  if (mid.includes("backbone") || mid === "bay_backbone" || mid === "trident_backbone") {
    return {
      id: "trident_backbone",
      label: "Trident Backbone (Clee 3mm Steel)",
      subsystem: "frame",
    };
  }
  if (
    mid === "top_brace" ||
    mid === "bay_top_brace" ||
    mid === "top_halo_ring" ||
    mid.startsWith("brace_top_open_")
  ) {
    return {
      id: "top_brace",
      label: mid.startsWith("brace_top_open_")
        ? "Top Open-Center Brace Bars (25×5mm Al)"
        : "Top Open-Center Brace",
      subsystem: "frame",
    };
  }
  if (
    mid === "front_brace" ||
    mid === "bay_front_brace" ||
    mid === "front_portal_frame" ||
    mid === "front_header_skirt" ||
    mid.startsWith("brace_front_open_")
  ) {
    return {
      id: "front_brace",
      label: mid.startsWith("brace_front_open_")
        ? "Front Open-Door Brace Bars (25×5mm Al)"
        : "Front Open-Door Brace",
      subsystem: "frame",
    };
  }
  if (mid.startsWith("panel_shear_") || mid.startsWith("bay_") || role === "panel") {
    if (mid.includes("left") || mid.includes("right")) {
      return { id: "side_panels", label: "Left & Right Enclosure Panels", subsystem: "frame" };
    }
    if (mid.includes("rear")) {
      return { id: "rear_panel", label: "Rear Enclosure Panel", subsystem: "frame" };
    }
    if (mid.includes("top")) {
      return { id: "top_panel", label: "Top Enclosure Panel", subsystem: "frame" };
    }
    return { id: "enclosure_panels", label: "Enclosure Panels", subsystem: "frame" };
  }
  if (mid.startsWith("brace_") || mid.startsWith("diag_") || role === "brace") {
    return { id: "frame_bracing", label: "Frame Diagonal Bracing", subsystem: "frame" };
  }
  if (
    mid.startsWith("top_") ||
    mid.startsWith("bot_") ||
    mid.startsWith("base_") ||
    mid.startsWith("g_front") ||
    mid.startsWith("g_rear") ||
    mid.startsWith("g_left") ||
    mid.startsWith("g_right") ||
    mid.startsWith("gantry_rear_bar") ||
    mid.startsWith("portal_top")
  ) {
    return { id: "frame_perimeter_beams", label: "Frame Perimeter Beams", subsystem: "frame" };
  }

  if (role === "rail") {
    return { id: "y_axis_rail_beams", label: "Y-Axis Rail Beams", subsystem: "gantry" };
  }
  if (role === "gantry") {
    return { id: "x_gantry_beam", label: "Moving X-Gantry Beam", subsystem: "gantry" };
  }
  if (role === "bed") {
    return { id: "bed_support", label: "Bed Support & Z Arms", subsystem: "bed" };
  }
  if (role === "toolhead") {
    return { id: "toolhead_mount", label: "Toolhead & Nozzle Mount", subsystem: "toolhead" };
  }
  if (role === "frame") {
    return { id: "frame_perimeter_beams", label: "Frame Perimeter Beams", subsystem: "frame" };
  }
  return {
    id: mid || "frame",
    label: mid ? mid.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) : "Structural Member",
    subsystem: "frame",
  };
}

export function formatBackendLabel(backend: string): string {
  const low = backend.toLowerCase();
  if (low.includes("gmsh") && (low.includes("skfem") || low.includes("scikit"))) {
    return "Gmsh + scikit-fem";
  }
  return backend;
}

export function formatElementLabel(elementType: string): string {
  const low = elementType.toLowerCase();
  if (low === "tet10" || low === "p2" || low === "tet10_p2") return "Tet10 (P2)";
  if (low === "tet4" || low === "p1" || low === "tet4_p1") return "Tet4 (P1)";
  return elementType;
}

export function modeSeverityMeta(severity?: "critical" | "moderate" | "stiff"): {
  label: string;
  cls: string;
} | null {
  if (severity === "critical") return { label: "Limiting", cls: "bad" };
  if (severity === "moderate") return { label: "Moderate", cls: "warn" };
  if (severity === "stiff") return { label: "Stiff", cls: "ok" };
  return null;
}

export function formatHzGain(hz: number): string {
  return hz >= 0.1 ? hz.toFixed(1) : hz.toFixed(2);
}

export function getLensScore(
  entry:
    | {
        heat_score?: number;
        heat_score_compliance?: number;
        heat_score_modal?: number;
        heat_score_mass?: number;
        hz_per_100g?: number;
      }
    | undefined,
  lens: AdjointLens
): number {
  if (!entry) return 0;
  if (lens === "compliance") {
    return entry.heat_score_compliance ?? Math.max(0, entry.heat_score ?? 0);
  }
  if (lens === "modal") {
    return entry.heat_score_modal ?? entry.heat_score ?? 0;
  }
  if (lens === "mass") {
    if (entry.heat_score_mass != null) return entry.heat_score_mass;
    if (entry.heat_score != null && entry.heat_score < 0) return entry.heat_score;
    if (entry.hz_per_100g != null && entry.hz_per_100g < 0) {
      return Math.max(-1, entry.hz_per_100g / 6);
    }
    return 0;
  }
  if (entry.heat_score != null) return entry.heat_score;
  if (entry.hz_per_100g != null && entry.hz_per_100g < 0) {
    return Math.max(-1, entry.hz_per_100g / 6);
  }
  return 0;
}

export function impactLevelMeta(level?: "critical" | "high" | "moderate" | "low"): {
  label: string;
  cls: string;
  dotColor: string;
} {
  if (level === "critical") return { label: "Critical", cls: "bad", dotColor: "#ef4444" };
  if (level === "high") return { label: "High", cls: "warn", dotColor: "#f59e0b" };
  if (level === "moderate") return { label: "Moderate", cls: "warn", dotColor: "#eab308" };
  return { label: "Low", cls: "ok", dotColor: "#64748b" };
}
