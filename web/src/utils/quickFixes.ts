import { ModeShapeData, PrinterSpecData, SubsystemKey } from "../types";

export interface QuickFixAction {
  id: string;
  buttonLabel: string;
  summary: string;
  apply: (spec: PrinterSpecData) => PrinterSpecData;
}

function cloneSpec(spec: PrinterSpecData): PrinterSpecData {
  if (typeof structuredClone === "function") {
    return structuredClone(spec);
  }
  return JSON.parse(JSON.stringify(spec)) as PrinterSpecData;
}

function round1(v: number): number {
  return Math.round(v * 10) / 10;
}

function scaleTuple3(
  tuple: [number, number, number] | undefined,
  fallback: [number, number, number],
  factor: number
): [number, number, number] {
  const src = tuple ?? fallback;
  return [round1(src[0] * factor), round1(src[1] * factor), round1(src[2] * factor)];
}

export interface PartGroupClassification {
  id: string;
  label: string;
  subsystem: SubsystemKey;
}

/**
 * Mirror backend `classify_part_group` + `frame_feet` so 3D viewport raycasting
 * maps member/bay IDs to the exact same physical group IDs used by adjoint sensitivity
 * and deterministic quick-fix actions.
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
  if (mid.startsWith("bed_") || mid.startsWith("z_bed")) {
    return { id: "bed_support", label: "Bed Support & Z Arms", subsystem: "bed" };
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

function makeBoltAlPanelsFix(idSuffix: string): QuickFixAction {
  return {
    id: `bolt_al_panels_${idSuffix}`,
    buttonLabel: "Bolt 4mm Al 5052 Panels",
    summary: "Quick Fix: Bolted 4mm Aluminum 5052 structural enclosure panels",
    apply: (spec) => {
      const next = cloneSpec(spec);
      next.frame.enclosed_panels = true;
      next.frame.panel_mounting = "bolted_structural";
      next.frame.panel_material = "aluminum_5052_sheet";
      next.frame.panel_thickness_mm = Math.max(4, next.frame.panel_thickness_mm || 0);
      return next;
    },
  };
}

function hasBoltedAlPanels(spec: PrinterSpecData): boolean {
  return (
    Boolean(spec.frame.enclosed_panels) &&
    spec.frame.panel_mounting === "bolted_structural" &&
    spec.frame.panel_material === "aluminum_5052_sheet" &&
    (spec.frame.panel_thickness_mm ?? 0) >= 4
  );
}

/**
 * Return a deterministic, physically grounded one-click design upgrade for a given
 * sensitivity target ID, part group ID, or subsystem key.
 */
export function getQuickFixForTarget(
  targetId: string,
  spec: PrinterSpecData
): QuickFixAction | null {
  const tid = (targetId || "").trim().toLowerCase();
  if (!tid) return null;

  // 1. Toolhead Mount & Nozzle Overhang
  if (
    tid === "toolhead_mount" ||
    tid === "toolhead" ||
    tid === "nozzle_stem" ||
    tid === "toolhead_cog_spring"
  ) {
    if (
      spec.toolhead.z_probe_type === "printed_tap" ||
      spec.toolhead.z_probe_type === "cnc_tap"
    ) {
      return {
        id: "switch_tap_to_beacon",
        buttonLabel: "Switch Tap → Static Beacon",
        summary:
          "Quick Fix: Switched Voron Tap to Static Eddy (Beacon/Cartographer) to eliminate sliding-rail play & probe mass",
        apply: (s) => {
          const next = cloneSpec(s);
          next.toolhead.z_probe_type = "static_eddy";
          return next;
        },
      };
    }
    const mountMat = spec.toolhead.mount_material ?? "abs_printed";
    const currStiff = spec.toolhead.internal_stiffness_n_per_mm ?? 450;
    if (mountMat === "abs_printed" || mountMat === "pet_cf_printed") {
      return {
        id: "upgrade_mount_ppa_cf",
        buttonLabel: "Upgrade Mount → PPA-CF",
        summary: "Quick Fix: Upgraded toolhead mount to PPA-CF (≥1150 N/mm)",
        apply: (s) => {
          const next = cloneSpec(s);
          next.toolhead.mount_material = "ppa_cf_printed";
          next.toolhead.internal_stiffness_n_per_mm = Math.max(
            next.toolhead.internal_stiffness_n_per_mm ?? 450,
            1150
          );
          return next;
        },
      };
    }
    if (mountMat !== "aluminum_6061_t6" && mountMat !== "aluminum_7075_t6") {
      return {
        id: "upgrade_mount_cnc_al",
        buttonLabel: "Upgrade Mount → CNC Al",
        summary: "Quick Fix: Upgraded toolhead mount to CNC Aluminum 6061-T6 (≥1850 N/mm)",
        apply: (s) => {
          const next = cloneSpec(s);
          next.toolhead.mount_material = "aluminum_6061_t6";
          next.toolhead.internal_stiffness_n_per_mm = Math.max(currStiff, 1850);
          return next;
        },
      };
    }
    const nozZ = Math.abs(spec.toolhead.nozzle_offset_mm?.[2] ?? 62);
    const cogY = Math.abs(spec.toolhead.cog_offset_mm?.[1] ?? 18);
    if (nozZ > 45 || cogY > 14) {
      return {
        id: "tighten_cog_overhang",
        buttonLabel: "Tighten CoG Overhang",
        summary: "Quick Fix: Tightened toolhead CoG & nozzle lever arm by 25%",
        apply: (s) => {
          const next = cloneSpec(s);
          next.toolhead.cog_offset_mm = scaleTuple3(next.toolhead.cog_offset_mm, [0, -18, -25], 0.75);
          next.toolhead.nozzle_offset_mm = scaleTuple3(
            next.toolhead.nozzle_offset_mm,
            [0, -22, -62],
            0.75
          );
          return next;
        },
      };
    }
    return null;
  }

  // 2. Toolhead Lightening (Moving Mass)
  if (tid === "toolhead_assembly" || tid === "toolhead_cog" || tid === "toolhead_lighten") {
    if (
      spec.toolhead.z_probe_type === "printed_tap" ||
      spec.toolhead.z_probe_type === "cnc_tap"
    ) {
      return {
        id: "switch_tap_to_beacon",
        buttonLabel: "Switch Tap → Static Beacon",
        summary:
          "Quick Fix: Switched Voron Tap to Static Eddy (Beacon/Cartographer) to eliminate sliding-rail play & probe mass",
        apply: (s) => {
          const next = cloneSpec(s);
          next.toolhead.z_probe_type = "static_eddy";
          return next;
        },
      };
    }
    if (spec.toolhead.total_mass_g > 235) {
      const nextMass = Math.max(220, Math.round(spec.toolhead.total_mass_g * 0.75));
      return {
        id: "lighten_toolhead_25",
        buttonLabel: "Lighten Toolhead (-25%)",
        summary: `Quick Fix: Lightened toolhead ${Math.round(spec.toolhead.total_mass_g)}g → ${nextMass}g & tightened CoG (-15%)`,
        apply: (s) => {
          const next = cloneSpec(s);
          next.toolhead.total_mass_g = Math.max(220, Math.round(next.toolhead.total_mass_g * 0.75));
          next.toolhead.cog_offset_mm = scaleTuple3(next.toolhead.cog_offset_mm, [0, -18, -25], 0.85);
          return next;
        },
      };
    }
    return null;
  }

  // 3. X-Gantry Beam
  if (tid === "x_gantry_beam" || tid === "gantry") {
    if (!spec.gantry.x_beam_material.includes("carbon")) {
      return {
        id: "x_beam_carbon",
        buttonLabel: "Switch X-Beam → Carbon Tube",
        summary: "Quick Fix: Switched X-gantry beam to 20×20×2mm Carbon Fiber Tube",
        apply: (s) => {
          const next = cloneSpec(s);
          next.gantry.x_beam_profile = "cf_tube_20x20x2";
          next.gantry.x_beam_material = "carbon_fiber_tube";
          return next;
        },
      };
    }
    const currXMount =
      spec.gantry.x_rail_mount ?? (spec.gantry.dual_x_rails ? "dual_top_bottom" : "front");
    if (currXMount !== "dual_top_bottom") {
      return {
        id: "x_beam_dual_rails",
        buttonLabel: "Upgrade → Dual X-Rails",
        summary: "Quick Fix: Upgraded X-gantry to dual top+bottom linear rails",
        apply: (s) => {
          const next = cloneSpec(s);
          next.gantry.x_rail_mount = "dual_top_bottom";
          next.gantry.dual_x_rails = true;
          return next;
        },
      };
    }
    return null;
  }

  // 4. Y-Axis Rail Beams
  if (tid === "y_axis_rail_beams") {
    const currYMount = spec.gantry.y_rail_mount ?? "top";
    if (currYMount !== "inner") {
      return {
        id: "y_rails_inner",
        buttonLabel: "Mount Y-Rails Inner Side",
        summary: "Quick Fix: Mounted Y-axis linear rails on inner side face",
        apply: (s) => {
          const next = cloneSpec(s);
          next.gantry.y_rail_mount = "inner";
          return next;
        },
      };
    }
    const currBacker = spec.gantry.rail_backer_material ?? "none";
    if (currBacker === "none") {
      return {
        id: "y_rails_steel_backers",
        buttonLabel: "Add Steel Rail Backers",
        summary: "Quick Fix: Added 4140 steel rail backers to stiffen & counter bimetallic bow",
        apply: (s) => {
          const next = cloneSpec(s);
          next.gantry.rail_backer_material = "steel_4140";
          if (!next.gantry.rail_backer_thickness_mm || next.gantry.rail_backer_thickness_mm < 2) {
            next.gantry.rail_backer_thickness_mm = 3;
          }
          return next;
        },
      };
    }
    return null;
  }

  // 5. Enclosure Panels
  if (
    tid === "side_panels" ||
    tid === "rear_panel" ||
    tid === "top_panel" ||
    tid === "enclosure_panels"
  ) {
    if (
      !spec.frame.enclosed_panels ||
      spec.frame.panel_mounting !== "bolted_structural" ||
      spec.frame.panel_material !== "aluminum_5052_sheet"
    ) {
      return makeBoltAlPanelsFix("panels");
    }
    if ((spec.frame.panel_thickness_mm ?? 0) < 5) {
      return {
        id: "panels_5mm_al",
        buttonLabel: "Step Up to 5mm Al Panels",
        summary: "Quick Fix: Stepped up bolted Aluminum 5052 structural panels to 5mm",
        apply: (s) => {
          const next = cloneSpec(s);
          next.frame.panel_thickness_mm = 5;
          return next;
        },
      };
    }
    return null;
  }

  // 6. Outer Frame Columns & Perimeter Beams
  if (
    tid === "front_frame_columns" ||
    tid === "rear_frame_columns" ||
    tid === "frame_columns" ||
    tid === "frame_perimeter_beams" ||
    tid === "z_gantry_uprights" ||
    tid === "frame"
  ) {
    if (spec.architecture === "corexy" && !hasBoltedAlPanels(spec)) {
      return makeBoltAlPanelsFix("frame");
    }
    if (spec.frame.profile === "extrusion_2020") {
      return {
        id: "upgrade_frame_3030",
        buttonLabel: "Upgrade Frame → 3030",
        summary: "Quick Fix: Upgraded frame extrusions from 2020 → 3030",
        apply: (s) => {
          const next = cloneSpec(s);
          next.frame.profile = "extrusion_3030";
          return next;
        },
      };
    }
    if ((spec.frame.extrusion_filling ?? "none") === "none") {
      return {
        id: "fill_frame_epoxy_granite",
        buttonLabel: "Fill Frame → Epoxy Granite",
        summary: "Quick Fix: Filled stationary frame extrusions with Epoxy Granite",
        apply: (s) => {
          const next = cloneSpec(s);
          next.frame.extrusion_filling = "epoxy_granite";
          return next;
        },
      };
    }
    return null;
  }

  // 7. Frame Bracing
  if (tid === "frame_bracing") {
    if (spec.frame.bracing === "none" || spec.frame.bracing === "corner_gussets") {
      return {
        id: "bracing_rear_cross",
        buttonLabel: "Add Rear X-Bracing",
        summary: "Quick Fix: Added Rear Cross + Top diagonal frame bracing",
        apply: (s) => {
          const next = cloneSpec(s);
          next.frame.bracing = "rear_cross";
          return next;
        },
      };
    }
    if (spec.frame.bracing === "rear_cross") {
      return {
        id: "bracing_full_truss",
        buttonLabel: "Upgrade → Full Truss",
        summary: "Quick Fix: Upgraded frame diagonal bracing to Full 3D Truss",
        apply: (s) => {
          const next = cloneSpec(s);
          next.frame.bracing = "full_truss";
          return next;
        },
      };
    }
    return null;
  }

  // 8. Bed Plate & Bed Support
  if (tid === "bed_support" || tid === "bed_plate" || tid === "bed") {
    const mountType = spec.bed.mount_type ?? "kinematic_3pt";
    const bedTemp = spec.bed.target_temp_c ?? 60;
    if (
      mountType === "springs_4pt" ||
      mountType === "silicone_4pt" ||
      (mountType === "rigid_4pt" && bedTemp >= 80)
    ) {
      return {
        id: "bed_kinematic_3pt",
        buttonLabel: "Switch Bed → Kinematic 3-Pt",
        summary: "Quick Fix: Switched heated bed mount to Kinematic 3-Pt (eliminates thermal taco & soft bounce)",
        apply: (s) => {
          const next = cloneSpec(s);
          next.bed.mount_type = "kinematic_3pt";
          return next;
        },
      };
    }
    if (
      spec.architecture === "corexy" &&
      (spec.bed.z_drive === "leadscrew_single" || spec.bed.z_drive === "leadscrew_dual")
    ) {
      return {
        id: "bed_triple_z",
        buttonLabel: "Upgrade Z → Triple-Z Trident",
        summary: "Quick Fix: Upgraded Z-drive to Triple-Z Trident (3-point constrained bed support)",
        apply: (s) => {
          const next = cloneSpec(s);
          next.bed.z_drive = "leadscrew_triple";
          return next;
        },
      };
    }
    if (spec.bed.plate_thickness_mm < 8) {
      return {
        id: "bed_plate_8mm",
        buttonLabel: "Thicken Bed Plate → 8mm",
        summary: "Quick Fix: Thickened cast bed plate to 8mm to curb sag & thermal bow",
        apply: (s) => {
          const next = cloneSpec(s);
          next.bed.plate_thickness_mm = 8;
          return next;
        },
      };
    }
    return null;
  }

  // 9. Frame Feet
  if (tid === "frame_feet") {
    if ((spec.frame.feet_type ?? "rigid_bolted") !== "rigid_bolted") {
      return {
        id: "anchor_feet_rigid",
        buttonLabel: "Anchor Feet → Rigid Bolted",
        summary: "Quick Fix: Anchored frame feet to Rigid Bolted (eliminates benchtop rocking)",
        apply: (s) => {
          const next = cloneSpec(s);
          next.frame.feet_type = "rigid_bolted";
          return next;
        },
      };
    }
    return null;
  }

  return null;
}

/**
 * Pick the highest-impact deterministic quick fix for a specific 3D vibration mode.
 */
export function getQuickFixForMode(
  mode: ModeShapeData,
  spec: PrinterSpecData
): QuickFixAction | null {
  const combinedText = [mode.mode_label, mode.what_is_shaking, mode.how_to_fix]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  // Check explicit physical causes first
  if (
    (combinedText.includes("feet") ||
      combinedText.includes("foot") ||
      combinedText.includes("isolat") ||
      combinedText.includes("benchtop") ||
      combinedText.includes("rocking")) &&
    (spec.frame.feet_type ?? "rigid_bolted") !== "rigid_bolted"
  ) {
    const feetFix = getQuickFixForTarget("frame_feet", spec);
    if (feetFix) return feetFix;
  }

  if (combinedText.includes("panel") && combinedText.includes("drum")) {
    const panelFix = getQuickFixForTarget("side_panels", spec);
    if (panelFix) return panelFix;
  }

  // Build ordered candidate target list from dominant_problem_area + subsystem energy shares
  const candidates: string[] = [];
  const pushTarget = (t: string) => {
    if (!candidates.includes(t)) candidates.push(t);
  };

  const mapSubsystem = (sub: SubsystemKey): string[] => {
    if (sub === "toolhead") return ["toolhead_mount", "toolhead_assembly"];
    if (sub === "gantry") return ["x_gantry_beam", "y_axis_rail_beams"];
    if (sub === "bed") return ["bed_support"];
    return ["frame", "frame_bracing"];
  };

  if (mode.dominant_problem_area) {
    for (const t of mapSubsystem(mode.dominant_problem_area)) {
      pushTarget(t);
    }
  }

  if (mode.subsystem_share_pct) {
    const ranked = (["toolhead", "gantry", "frame", "bed"] as SubsystemKey[])
      .map((k) => ({ k, pct: mode.subsystem_share_pct?.[k] ?? 0 }))
      .filter((x) => x.pct >= 12)
      .sort((a, b) => b.pct - a.pct);
    for (const { k } of ranked) {
      for (const t of mapSubsystem(k)) {
        pushTarget(t);
      }
    }
  }

  // Fallback text heuristics if backend didn't supply dominant_problem_area / subsystem_share_pct
  if (candidates.length === 0) {
    if (combinedText.includes("toolhead") || combinedText.includes("nozzle") || combinedText.includes("whip")) {
      pushTarget("toolhead_mount");
      pushTarget("toolhead_assembly");
    } else if (combinedText.includes("gantry") || combinedText.includes("x-beam") || combinedText.includes("x beam")) {
      pushTarget("x_gantry_beam");
    } else if (combinedText.includes("bed") || combinedText.includes("z-drive")) {
      pushTarget("bed_support");
    } else {
      pushTarget("frame");
      pushTarget("frame_bracing");
    }
  }

  for (const tid of candidates) {
    const fix = getQuickFixForTarget(tid, spec);
    if (fix) return fix;
  }

  return null;
}
