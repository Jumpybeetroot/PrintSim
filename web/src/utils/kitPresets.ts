import {
  BeltRoutingType,
  BracketMaterialType,
  GantryHardwareKitItem,
  GantryHardwareKitSettings,
  GantryHardwareKitType,
  PrinterSpecData,
  XRailMountType,
  YRailMountType,
} from "../types";
import { PresetAffectsItem, clearModelOverrideGroup } from "../components/ModelOverrideControl";

/**
 * Fallback kit_settings definitions mirroring backend GANTRY_HARDWARE_KIT_CATALOG
 * so gantry kit upgrades work seamlessly even if catalog.gantry_hardware_kits is minimal.
 */
export const FALLBACK_GANTRY_KIT_SETTINGS: Partial<
  Record<GantryHardwareKitType, GantryHardwareKitSettings>
> = {
  voron_stock_printed: {
    drive_mode: "2wd",
    belt_routing: "stock_corexy",
    bracket_material: "abs_printed",
    double_shear_pulleys: false,
    x_rail_profile: "mgn12_rail",
    y_rail_profile: "mgn9_rail",
    x_rail_mount: "front",
    y_rail_mount: "bottom",
    rail_block_type: "H",
    belt_type: "gt2_6mm",
    belt_pretension_n: 9.0,
    carriage_mass_g: 196,
    idler_axial_preload: "none",
    z_joint_material_flying: "printed",
  },
  ldo_voron_kit: {
    drive_mode: "2wd",
    belt_routing: "stock_corexy",
    bracket_material: "abs_printed",
    double_shear_pulleys: false,
    x_rail_profile: "mgn12_rail",
    y_rail_profile: "mgn9_rail",
    x_rail_mount: "front",
    y_rail_mount: "bottom",
    rail_block_type: "H",
    belt_type: "gt2_6mm",
    belt_pretension_n: 9.0,
    carriage_mass_g: 180,
    idler_axial_preload: "none",
    z_joint_material_flying: "printed",
  },
  siboor_cnc_awd: {
    drive_mode: "awd",
    belt_routing: "stock_corexy",
    bracket_material: "aluminum_6061_t6",
    double_shear_pulleys: true,
    x_rail_profile: "mgn12_rail",
    y_rail_profile: "mgn9_rail",
    x_rail_mount: "front",
    y_rail_mount: "bottom",
    rail_block_type: "H",
    belt_type: "gt2_9mm",
    belt_pretension_n: 55.0,
    carriage_mass_g: 210,
    idler_axial_preload: "none",
    z_joint_material_flying: "printed",
    sibling_2wd_kit: "siboor_cnc_2wd",
  },
  siboor_cnc_2wd: {
    drive_mode: "2wd",
    belt_routing: "stock_corexy",
    bracket_material: "aluminum_6061_t6",
    double_shear_pulleys: true,
    x_rail_profile: "mgn12_rail",
    y_rail_profile: "mgn9_rail",
    x_rail_mount: "front",
    y_rail_mount: "bottom",
    rail_block_type: "H",
    belt_type: "gt2_6mm",
    belt_pretension_n: 32.0,
    carriage_mass_g: 210,
    idler_axial_preload: "none",
    z_joint_material_flying: "printed",
    sibling_awd_kit: "siboor_cnc_awd",
  },
  monolith_ldo_cnc_awd: {
    drive_mode: "awd",
    belt_routing: "monolith_short",
    bracket_material: "aluminum_6061_t6",
    double_shear_pulleys: true,
    x_rail_profile: "mgn12_rail",
    y_rail_profile: "mgn12_rail",
    x_rail_mount: "front",
    y_rail_mount: "inner",
    rail_block_type: "H",
    belt_type: "gt2_9mm",
    belt_pretension_n: 65.0,
    carriage_mass_g: 214,
    idler_axial_preload: "disc_spring_m5",
    z_joint_material_flying: "milled",
    sibling_2wd_kit: "monolith_ldo_cnc_2wd",
  },
  monolith_ldo_cnc_2wd: {
    drive_mode: "2wd",
    belt_routing: "monolith_short",
    bracket_material: "aluminum_6061_t6",
    double_shear_pulleys: true,
    x_rail_profile: "mgn12_rail",
    y_rail_profile: "mgn12_rail",
    x_rail_mount: "front",
    y_rail_mount: "inner",
    rail_block_type: "H",
    belt_type: "gt2_9mm",
    belt_pretension_n: 48.0,
    carriage_mass_g: 214,
    idler_axial_preload: "disc_spring_m5",
    z_joint_material_flying: "milled",
    sibling_awd_kit: "monolith_ldo_cnc_awd",
  },
  vzbot_330_awd: {
    drive_mode: "awd",
    belt_routing: "stock_corexy",
    bracket_material: "aluminum_6061_t6",
    double_shear_pulleys: true,
    x_rail_profile: "mgn9_rail",
    y_rail_profile: "mgn12_rail",
    x_rail_mount: "top",
    y_rail_mount: "inner",
    rail_block_type: "H",
    belt_type: "gt2_6mm",
    belt_pretension_n: 22.0,
    carriage_mass_g: 194,
    idler_axial_preload: "none",
    z_joint_material_flying: "printed",
    sibling_2wd_kit: "vzbot_330_2wd",
  },
  vzbot_330_2wd: {
    drive_mode: "2wd",
    belt_routing: "stock_corexy",
    bracket_material: "aluminum_6061_t6",
    double_shear_pulleys: true,
    x_rail_profile: "mgn9_rail",
    y_rail_profile: "mgn12_rail",
    x_rail_mount: "top",
    y_rail_mount: "inner",
    rail_block_type: "H",
    belt_type: "gt2_6mm",
    belt_pretension_n: 22.0,
    carriage_mass_g: 194,
    idler_axial_preload: "none",
    z_joint_material_flying: "printed",
    sibling_awd_kit: "vzbot_330_awd",
  },
};

const BRACKET_SHORT_LABELS: Partial<Record<BracketMaterialType, string>> = {
  abs_printed: "Printed ABS",
  pet_cf_printed: "Printed PET-CF",
  pa12_cf_printed: "Printed PA12-CF",
  ppa_cf_printed: "Printed PPA-CF",
  pps_cf_printed: "Printed PPS-CF",
  slm_aluminum: "SLM Al",
  aluminum_6061_t6: "CNC Al 6061",
  aluminum_7075_t6: "CNC Al 7075",
};

const RAIL_MOUNT_SHORT: Record<XRailMountType | YRailMountType, string> = {
  front: "Front",
  top: "Top",
  bottom: "Bottom",
  inner: "Inner",
  dual_top_bottom: "Dual",
};

function shortRailName(profileId: string): string {
  return profileId.replace(/_hollow_rail$/i, " Hollow").replace(/_rail$/i, "").toUpperCase();
}

function shortBeltLabel(beltType: string): string {
  return beltType.replace(/^gt2_/i, "GT2 ").replace(/_/g, " ");
}

export function resolveKitSettings(
  kitId: GantryHardwareKitType,
  catalogKits?: GantryHardwareKitItem[]
): GantryHardwareKitSettings | null {
  const fromCatalog = catalogKits?.find((k) => k.id === kitId)?.kit_settings;
  if (fromCatalog) return fromCatalog;
  return FALLBACK_GANTRY_KIT_SETTINGS[kitId] ?? null;
}

/**
 * Switch `draft.gantry.hardware_kit` to `kitId` and apply all coupled physical settings
 * that ship with that gantry kit (drive mode, belt routing, bracket material, double-shear,
 * X/Y rail type & orientation, belt width & tension, carriage mass, axial bearing preload,
 * and V2.4 Z-joint hardware).
 */
export function applyGantryHardwareKit(
  draft: PrinterSpecData,
  kitId: GantryHardwareKitType,
  catalogKits?: GantryHardwareKitItem[]
): string[] {
  draft.gantry.hardware_kit = kitId;
  const settings = resolveKitSettings(kitId, catalogKits);
  if (!settings) return [];

  const changes: string[] = [];
  const g = draft.gantry;
  const isFlying = draft.architecture === "corexy" && draft.bed.z_drive === "belted_quad_z";

  if (settings.drive_mode && g.drive_mode !== settings.drive_mode) {
    g.drive_mode = settings.drive_mode;
    changes.push(g.drive_mode === "awd" ? "AWD (4-Motor)" : "2WD (2-Motor)");
  }

  if (settings.belt_routing && (g.belt_routing ?? "stock_corexy") !== settings.belt_routing) {
    g.belt_routing = settings.belt_routing as BeltRoutingType;
    changes.push(
      settings.belt_routing === "monolith_short" ? "Monolith Short Path" : "Stock CoreXY Path"
    );
    clearModelOverrideGroup(draft, "belt_routing");
  }

  if (
    settings.bracket_material &&
    (g.bracket_material ?? "abs_printed") !== settings.bracket_material
  ) {
    g.bracket_material = settings.bracket_material;
    changes.push(BRACKET_SHORT_LABELS[settings.bracket_material] ?? settings.bracket_material);
    clearModelOverrideGroup(draft, "gantry_bracket");
  }

  if (
    typeof settings.double_shear_pulleys === "boolean" &&
    Boolean(g.double_shear_pulleys) !== settings.double_shear_pulleys
  ) {
    g.double_shear_pulleys = settings.double_shear_pulleys;
    changes.push(settings.double_shear_pulleys ? "Double-Shear" : "Single-Shear");
  }

  let railsChanged = false;
  if (settings.x_rail_profile && g.x_rail_profile !== settings.x_rail_profile) {
    g.x_rail_profile = settings.x_rail_profile;
    railsChanged = true;
  }
  if (settings.y_rail_profile && g.y_rail_profile !== settings.y_rail_profile) {
    g.y_rail_profile = settings.y_rail_profile;
    railsChanged = true;
  }
  if (settings.x_rail_mount && (g.x_rail_mount ?? "front") !== settings.x_rail_mount) {
    g.x_rail_mount = settings.x_rail_mount;
    if (settings.x_rail_mount !== "dual_top_bottom") {
      g.dual_x_rails = false;
    }
    railsChanged = true;
  }
  if (settings.y_rail_mount && (g.y_rail_mount ?? "top") !== settings.y_rail_mount) {
    g.y_rail_mount = settings.y_rail_mount;
    railsChanged = true;
  }
  if (settings.rail_block_type && (g.rail_block_type ?? "H") !== settings.rail_block_type) {
    g.rail_block_type = settings.rail_block_type;
    railsChanged = true;
  }
  if (railsChanged) {
    const yMountLbl = RAIL_MOUNT_SHORT[g.y_rail_mount ?? "bottom"];
    changes.push(`${yMountLbl} ${shortRailName(g.y_rail_profile)}${g.rail_block_type ?? "H"} Y-Rails`);
  }

  let beltChanged = false;
  if (settings.belt_type && g.belt.type !== settings.belt_type) {
    g.belt.type = settings.belt_type;
    beltChanged = true;
  }
  if (
    typeof settings.belt_pretension_n === "number" &&
    Math.abs((g.belt.pretension_n ?? 0) - settings.belt_pretension_n) > 0.1
  ) {
    g.belt.pretension_n = settings.belt_pretension_n;
    beltChanged = true;
  }
  if (beltChanged) {
    changes.push(`${shortBeltLabel(g.belt.type)} @ ${Math.round(g.belt.pretension_n)} N`);
  }

  if (
    typeof settings.carriage_mass_g === "number" &&
    Math.abs(g.carriage_mass_g - settings.carriage_mass_g) > 0.1
  ) {
    g.carriage_mass_g = settings.carriage_mass_g;
  }

  if (
    settings.idler_axial_preload &&
    (g.idler_axial_preload ?? "auto") !== settings.idler_axial_preload
  ) {
    g.idler_axial_preload = settings.idler_axial_preload;
    if (settings.idler_axial_preload === "disc_spring_m5") {
      changes.push("M5 Disc Springs");
    }
  }

  if (
    isFlying &&
    settings.z_joint_material_flying &&
    (g.z_joint_material ?? "printed") !== settings.z_joint_material_flying
  ) {
    g.z_joint_material = settings.z_joint_material_flying;
    changes.push(
      settings.z_joint_material_flying === "milled" ? "Milled V2 Z-Joints" : "Printed V2 Z-Joints"
    );
  }

  return changes;
}

/**
 * Toggle CoreXY 2WD <-> AWD and keep paired CAD gantry kits (e.g. Siboor CNC 2WD <-> AWD,
 * Monolith CNC 2WD <-> AWD) and their belt width/tension in sync.
 */
export function syncDriveModeWithHardwareKit(
  draft: PrinterSpecData,
  nextDriveMode: "2wd" | "awd",
  catalogKits?: GantryHardwareKitItem[]
): { switchedKitId: GantryHardwareKitType | null; changes: string[] } {
  const curKit = (draft.gantry.hardware_kit ?? "auto") as GantryHardwareKitType;
  const curSettings = resolveKitSettings(curKit, catalogKits);
  const siblingKit =
    nextDriveMode === "awd" ? curSettings?.sibling_awd_kit : curSettings?.sibling_2wd_kit;

  if (siblingKit) {
    const changes = applyGantryHardwareKit(draft, siblingKit, catalogKits);
    return { switchedKitId: siblingKit, changes };
  }

  draft.gantry.drive_mode = nextDriveMode;
  return { switchedKitId: null, changes: [] };
}

/**
 * Sync Z-coupled defaults when switching CoreXY Gantry & Z Architecture
 * (e.g. Fixed Gantry Triple-Z Trident <-> Flying Gantry Quad-Belted Z Voron 2.4).
 */
export function syncZArchitectureDefaults(
  draft: PrinterSpecData,
  nextZDrive: PrinterSpecData["bed"]["z_drive"]
): string[] {
  const prevIsFlying = draft.bed.z_drive === "belted_quad_z";
  const nextIsFlying = nextZDrive === "belted_quad_z";
  const changes: string[] = [];

  draft.bed.z_drive = nextZDrive;
  if (!draft.bed.z_leadscrew_type || draft.bed.z_leadscrew_type === "auto") {
    clearModelOverrideGroup(draft, "z_drive_mechanism");
  }

  if (draft.architecture !== "corexy" || prevIsFlying === nextIsFlying) {
    return changes;
  }

  const curClear = Math.round(draft.frame.gantry_top_clearance_mm ?? 55);
  const kitId = draft.gantry.hardware_kit ?? "auto";
  const isSiboor = kitId === "siboor_cnc_awd" || kitId === "siboor_cnc_2wd";
  const isMonolith = kitId === "monolith_ldo_cnc_awd" || kitId === "monolith_ldo_cnc_2wd";

  if (nextIsFlying) {
    // Fixed Trident -> Flying Voron 2.4
    if (curClear === 110) {
      draft.frame.gantry_top_clearance_mm = 55;
      changes.push("Top Clearance 55 mm");
    }
    const curLead = draft.bed.z_leadscrew_type ?? "auto";
    if (!curLead.includes("z_belt")) {
      const targetBelt = isSiboor ? "gt2_9mm_z_belt" : "gt2_6mm_z_belt";
      draft.bed.z_leadscrew_type = targetBelt;
      clearModelOverrideGroup(draft, "z_drive_mechanism");
      changes.push(isSiboor ? "9 mm Z Belts" : "6 mm Z Belts");
    }
    if (isMonolith && (draft.gantry.z_joint_material ?? "printed") !== "milled") {
      draft.gantry.z_joint_material = "milled";
      changes.push("Milled V2 Z-Joints");
    }
    if (
      Math.round(draft.frame.height_mm) === 500 &&
      Math.round(draft.kinematics.build_volume_mm[2]) === 250 &&
      Math.round(draft.bed.size_mm[0]) === 300
    ) {
      draft.frame.height_mm = 480;
      draft.kinematics.build_volume_mm = [
        draft.kinematics.build_volume_mm[0],
        draft.kinematics.build_volume_mm[1],
        300,
      ];
      changes.push("Frame 480 mm · 300 Z");
    }
  } else {
    // Flying Voron 2.4 -> Fixed Trident
    if (curClear === 55) {
      draft.frame.gantry_top_clearance_mm = 110;
      changes.push("110 mm Top Space");
    }
    const curLead = draft.bed.z_leadscrew_type ?? "auto";
    if (curLead.includes("z_belt") || curLead === "auto") {
      draft.bed.z_leadscrew_type = nextZDrive === "leadscrew_triple" ? "tr8x4_pom" : "tr8x8_pom";
      clearModelOverrideGroup(draft, "z_drive_mechanism");
      changes.push(nextZDrive === "leadscrew_triple" ? "TR8×4 Leadscrews" : "TR8×8 Leadscrews");
    }
    if (nextZDrive === "leadscrew_triple" && draft.bed.mount_type !== "trident_3pt_standoffs") {
      draft.bed.mount_type = "trident_3pt_standoffs";
      clearModelOverrideGroup(draft, "bed_mount");
      changes.push("Trident 3-Pt GE5C");
    }
    if (
      Math.round(draft.frame.height_mm) === 480 &&
      Math.round(draft.kinematics.build_volume_mm[2]) === 300 &&
      Math.round(draft.bed.size_mm[0]) === 300
    ) {
      draft.frame.height_mm = 500;
      draft.kinematics.build_volume_mm = [
        draft.kinematics.build_volume_mm[0],
        draft.kinematics.build_volume_mm[1],
        250,
      ];
      changes.push("Frame 500 mm · 250 Z");
    }
  }

  return changes;
}

/**
 * Build concise `PresetAffectsStrip` items summarizing what a gantry hardware kit configures.
 */
export function getGantryKitAffectsItems(
  kitId: GantryHardwareKitType,
  catalogKits?: GantryHardwareKitItem[]
): PresetAffectsItem[] {
  const s = resolveKitSettings(kitId, catalogKits);
  if (!s) return [];
  const items: PresetAffectsItem[] = [];
  if (s.drive_mode) {
    items.push({ label: "Drive", value: s.drive_mode === "awd" ? "AWD (4M)" : "2WD" });
  }
  if (s.belt_routing) {
    items.push({
      label: "Routing",
      value: s.belt_routing === "monolith_short" ? "Monolith Short" : "Stock CoreXY",
    });
  }
  if (s.bracket_material) {
    items.push({
      label: "Brackets",
      value: BRACKET_SHORT_LABELS[s.bracket_material] ?? s.bracket_material,
    });
  }
  if (s.y_rail_profile && s.y_rail_mount) {
    items.push({
      label: "Y Rails",
      value: `${RAIL_MOUNT_SHORT[s.y_rail_mount]} ${shortRailName(s.y_rail_profile)}${s.rail_block_type ?? "H"}`,
    });
  }
  if (s.belt_type && typeof s.belt_pretension_n === "number") {
    items.push({
      label: "Belt",
      value: `${shortBeltLabel(s.belt_type)} @ ${Math.round(s.belt_pretension_n)} N`,
    });
  }
  return items;
}
