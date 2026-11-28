import * as THREE from "three";
import type {
  CadIntraPartFieldData,
  ToolheadPartModalData,
  ToolheadPartStaticData,
} from "../types";
import { stressColor, stressColorInto } from "./colormaps";
import type { InspectedPartInfo } from "./geometryHelpers";

export type ToolheadPartStage =
  | "rail_carriage"
  | "backplate"
  | "chassis"
  | "chassis_carrier"
  | "chassis_clamshell"
  | "chassis_extruder"
  | "extruder_upper"
  | "hotend_clamp"
  | "hotend_brace"
  | "hotend_neck"
  | "heater_nozzle"
  | "cooling_shroud";

export interface ToolheadSubmeshDescriptor {
  familyId: string;
  label: string;
  stage: ToolheadPartStage;
  isStructural: boolean;
  /** Relative heat/flex weight within its stage (e.g. heatbreak throat = 1.0, heatsink fins = 0.35) */
  heatWeight: number;
  /** Role note shown in Part Inspector */
  roleNote: string;
  /** Local CAD explode translation in SI meters (CAD +X right, -Y front, +Z up) */
  explodeCadM: [number, number, number];
  /** If true, push outward along mesh's own radial vector from toolhead center */
  explodeRadialM?: number;
}

/**
 * Strip trailing instance numbers (`_1`, `_12`, `__1_`) to get a clean part family ID.
 */
function normalizeFamilyId(meshName: string): string {
  return (meshName || "")
    .toLowerCase()
    .trim()
    .replace(/(_+\d+)+_*$/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function humanizeMeshName(familyId: string): string {
  const acronyms: Record<string, string> = {
    cw2: "CW2",
    pcb: "PCB",
    mgn12: "MGN12",
    mgn12h: "MGN12H",
    mgn9: "MGN9",
    mgn9h: "MGN9H",
    ti: "Ti",
    slm: "SLM",
    cnc: "CNC",
    cpap: "CPAP",
    led: "LED",
    ptfe: "PTFE",
    ecas: "ECAS",
    e3d: "E3D",
    v6: "V6",
    ss304: "SS304",
    cucrzr: "CuCrZr",
    dav: "Dragon Ace Volcano",
    mbt: "Microwaved",
    sb: "StealthBurner",
    db: "Dragon Burner",
    xol: "XOL",
    a4t: "A4T",
    arch: "Archetype",
    eva3: "EVA 3",
    hef: "Hotend Fan",
    hw: "Hardware",
    m3: "M3",
    m25: "M2.5",
  };
  return familyId
    .split("_")
    .filter(Boolean)
    .map((tok) => acronyms[tok] ?? tok.charAt(0).toUpperCase() + tok.slice(1))
    .join(" ");
}

/**
 * Map any CAD GLB sub-mesh name (across all 13 toolhead GLBs & 10 hotend GLBs) to its
 * exact human-readable part identity, structural stage, heat weight, and CAD explode offset.
 */
export function describeToolheadSubmesh(
  meshName: string,
  defaultStage: "chassis" | "hotend_neck",
  centerCadMm?: [number, number, number]
): ToolheadSubmeshDescriptor {
  const fam = normalizeFamilyId(meshName);
  const nm = fam || (meshName || "").toLowerCase();
  const [cx, cy, cz] = centerCadMm ?? [0, 0, 0];

  // --- Fasteners / screws / heatsets / standoffs ---
  if (
    nm.includes("fastener") ||
    nm.includes("torx_screw") ||
    nm.includes("setscrew") ||
    nm.includes("m25_screw") ||
    nm.includes("ti_screw") ||
    nm === "screw" ||
    nm.includes("sb_hw") ||
    nm.includes("rapido2_hardware") ||
    nm.includes("din_912") ||
    nm.includes("heatset") ||
    nm.includes("heat_insert") ||
    nm.includes("m3_") ||
    nm.includes("m2_5_") ||
    nm.includes("spacers")
  ) {
    const isHeatset = nm.includes("heatset") || nm.includes("heat_insert");
    return {
      familyId: fam || "fasteners",
      label: isHeatset ? "M3 Heatset Inserts" : "Fasteners & Hardware",
      stage: "cooling_shroud",
      isStructural: false,
      heatWeight: 0.12,
      roleNote: "Clamping hardware & threaded inserts (transfers bolt preload between mating faces).",
      explodeCadM: [0, -0.015, 0],
      explodeRadialM: 0.026,
    };
  }

  // --- Hotend Bottom Brace ---
  if (nm.includes("hotend_brace") || nm.includes("db_brace") || nm.includes("ti_brace")) {
    return {
      familyId: fam || "hotend_brace",
      label: "Hotend Bottom Brace",
      stage: "hotend_brace",
      isStructural: true,
      heatWeight: 0.85,
      roleNote: "Lower secondary hotend support brace — eliminates cantilever heatbreak whip.",
      explodeCadM: [0, -0.028, -0.024],
    };
  }

  // --- PCBs, Probes, LEDs, Fittings ---
  if (nm.includes("nitehawk") || nm.includes("pcb") || nm.includes("electronics")) {
    return {
      familyId: fam || "toolhead_pcb",
      label: nm.includes("beacon")
        ? "Beacon Probe PCB"
        : nm.includes("nitehawk")
          ? "Nitehawk Toolhead PCB"
          : "Toolhead Control PCB",
      stage: "cooling_shroud",
      isStructural: false,
      heatWeight: 0.1,
      roleNote: "Toolhead electronics board (non-structural rider; adds upper moving mass).",
      explodeCadM: [0, -0.036, 0.026],
    };
  }
  if (nm.includes("beacon") || nm.includes("probe") || nm.includes("endstop")) {
    const isProbeMount = nm.includes("mount") || nm.includes("bracket");
    return {
      familyId: fam || (isProbeMount ? "probe_mount" : "probe"),
      label: nm.includes("beacon")
        ? isProbeMount
          ? "Beacon Probe Mount"
          : "Beacon Eddy-Current Probe"
        : nm.includes("endstop")
          ? "Toolhead Endstop Switch"
          : isProbeMount
            ? "Z-Probe Mount Bracket"
            : "Z-Probe Sensor",
      stage: "cooling_shroud",
      isStructural: false,
      heatWeight: 0.12,
      roleNote: isProbeMount
        ? "Printed probe mounting bracket (non-structural rider)."
        : "Bed-leveling probe mounted to toolhead (non-structural rider).",
      explodeCadM: [0, 0.022, -0.022],
    };
  }
  if (nm.includes("ptfe")) {
    return {
      familyId: fam || "ptfe_tube",
      label: "PTFE Filament Guide Tube",
      stage: "cooling_shroud",
      isStructural: false,
      heatWeight: 0.08,
      roleNote: "Low-friction PTFE filament guide tube (non-structural rider).",
      explodeCadM: [0, 0, 0.032],
    };
  }
  if (nm.includes("led") || nm.includes("logo") || nm.includes("ecas") || nm.includes("apex_clip")) {
    return {
      familyId: fam || "accessory",
      label: humanizeMeshName(fam || "accessory"),
      stage: "cooling_shroud",
      isStructural: false,
      heatWeight: 0.08,
      roleNote: "Non-structural toolhead accessory / trim.",
      explodeCadM: [0, -0.044, 0.016],
    };
  }

  // --- Cooling Shrouds, Cowls, Faceplates, Ducts, Fans, Shields ---
  if (
    nm.includes("shroud") ||
    nm.includes("cowl") ||
    nm.includes("faceplate") ||
    nm.includes("accent") ||
    nm.includes("fan_cover")
  ) {
    return {
      familyId: fam || "cooling_shroud",
      label: humanizeMeshName(fam || "cooling_shroud"),
      stage: "cooling_shroud",
      isStructural: false,
      heatWeight: 0.14,
      roleNote: "Front cooling shroud / aesthetic cover (non-structural mass rider).",
      explodeCadM: [0, -0.052, 0.0],
    };
  }
  if (nm.includes("duct") || nm.includes("cpap_tube") || nm.includes("cpap_inlet") || nm.includes("cpap_clamp")) {
    const xPush = nm.includes("left") ? -0.028 : nm.includes("right") ? 0.028 : 0.0;
    return {
      familyId: fam || "cooling_duct",
      label: humanizeMeshName(fam || "cooling_duct"),
      stage: "cooling_shroud",
      isStructural: false,
      heatWeight: 0.12,
      roleNote: "Part-cooling airflow duct / CPAP channel (non-structural mass rider).",
      explodeCadM: [xPush, -0.042, -0.014],
    };
  }
  if (nm.includes("fan") || nm.includes("delta_") || nm.includes("blades")) {
    return {
      familyId: fam || "cooling_fan",
      label: nm.includes("2510") || nm.includes("hef")
        ? "Hotend Heatsink Fan"
        : nm.includes("3628")
          ? "Part Cooling Blower Fan"
          : humanizeMeshName(fam || "cooling_fan"),
      stage: "cooling_shroud",
      isStructural: false,
      heatWeight: 0.12,
      roleNote: "Cooling fan assembly (non-structural moving mass).",
      explodeCadM: [0, -0.046, 0.008],
      explodeRadialM: 0.016,
    };
  }
  if (nm.includes("shield") || nm.includes("sock")) {
    return {
      familyId: fam || "thermal_shield",
      label: nm.includes("sock") ? "Silicone Heater Sock" : "Nozzle Heat Shield",
      stage: "cooling_shroud",
      isStructural: false,
      heatWeight: 0.08,
      roleNote: "Thermal insulation shield / silicone sock (non-structural rider).",
      explodeCadM: [0, -0.024, -0.048],
    };
  }

  // --- Carriage Mount / Backplate / Belt Clamps ---
  if (
    nm.includes("carriage") ||
    nm.includes("mgn12_mount") ||
    nm.includes("mgn9_mount") ||
    nm.includes("backplate") ||
    nm.includes("back_plate") ||
    nm.includes("back_piece") ||
    nm.includes("belt_plate") ||
    nm.includes("belt_clamp") ||
    nm.includes("belt_clip")
  ) {
    const sideSuffix =
      centerCadMm && (nm.includes("mgn12_mount") || nm.includes("belt_clip"))
        ? cx > 4
          ? " (Right)"
          : cx < -4
            ? " (Left)"
            : ""
        : "";
    const sideId = sideSuffix ? (cx > 4 ? "_right" : "_left") : "";
    return {
      familyId: `${fam || "carriage_mount"}${sideId}`,
      label: `${humanizeMeshName(fam || "carriage_mount")}${sideSuffix}`,
      stage: "backplate",
      isStructural: true,
      heatWeight: 0.95,
      roleNote: "Primary structural interface bolting the toolhead to the MGN carriage & XY belts.",
      explodeCadM: [cx > 4 ? 0.018 : cx < -4 ? -0.018 : 0, -0.016, 0],
    };
  }

  // --- Upper Extruder, Motor, Gears ---
  if (nm.includes("motor") || nm.includes("stepper")) {
    return {
      familyId: fam || "extruder_motor",
      label: nm.includes("cw2") ? "Clockwork 2 NEMA14 Motor" : "Extruder Stepper Motor",
      stage: "extruder_upper",
      isStructural: false,
      heatWeight: 0.45,
      roleNote: "Upper direct-drive stepper motor — major driver of Y-axis pitch inertia.",
      explodeCadM: [0, 0.008, 0.052],
    };
  }
  if (nm.includes("gear")) {
    return {
      familyId: fam || "extruder_gears",
      label: nm.startsWith("arch_") ? "Sherpa Micro Drive Gears" : "Extruder Drive Gears",
      stage: "extruder_upper",
      isStructural: false,
      heatWeight: 0.22,
      roleNote: "Filament drive gears & shafts inside the extruder (non-structural moving mass).",
      explodeCadM: [0, 0.014, 0.044],
    };
  }
  if (
    nm.includes("extruder") ||
    nm.includes("orbiter") ||
    nm.includes("sherpa") ||
    nm.includes("cw2") ||
    nm.includes("top_plate")
  ) {
    return {
      familyId: fam || "extruder_body",
      label: nm.includes("sherpa_housing")
        ? "Sherpa Micro Housing"
        : humanizeMeshName(fam || "extruder_body"),
      stage: "extruder_upper",
      isStructural: true,
      heatWeight: 0.65,
      roleNote: "Direct-drive extruder housing / drive assembly mounted above the rail.",
      explodeCadM: [0, -0.008, 0.036],
    };
  }

  // --- Hotend Mount / Collar / Adapter / Top Crown ---
  if (
    nm.includes("hotend_adapter") ||
    nm.includes("dragon_mount") ||
    nm.includes("hotend_mount") ||
    nm.includes("bottom_plate") ||
    nm.includes("top_cup") ||
    nm.includes("ti_crown") ||
    nm.includes("clamp") ||
    nm.includes("collar")
  ) {
    return {
      familyId: fam || "hotend_mount",
      label: humanizeMeshName(fam || "hotend_mount"),
      stage: "hotend_clamp",
      isStructural: true,
      heatWeight: 0.85,
      roleNote: "Rigid hotend clamping flange / mounting crown transferring nozzle loads into chassis.",
      explodeCadM: [0, 0, -0.018],
    };
  }

  // --- Hotend Sub-Parts (Nozzle, Heater Block, Cartridges, Heatbreak, Standoffs, Heatsink) ---
  if (nm.includes("nozzle")) {
    return {
      familyId: fam || "nozzle_tip",
      label: humanizeMeshName(fam || "nozzle_tip"),
      stage: "heater_nozzle",
      isStructural: false,
      heatWeight: 0.35,
      roleNote: "Print nozzle tip — experiences the maximum total displacement at the bottom of the lever arm.",
      explodeCadM: [0, 0, -0.058],
    };
  }
  if (nm.includes("heater_cartridge") || nm.includes("thermistor") || nm.includes("leads") || nm.includes("heater_tab") || nm === "heater" || nm.includes("supervolcano_heater")) {
    return {
      familyId: fam || "heater_element",
      label: humanizeMeshName(fam || "heater_element"),
      stage: "heater_nozzle",
      isStructural: false,
      heatWeight: 0.18,
      roleNote: "Heater cartridge / thermistor sensor (non-structural thermal mass).",
      explodeCadM: [0.024, 0, -0.038],
    };
  }
  if (nm.includes("block") || nm.includes("melt_core") || nm.includes("copper_core") || nm.includes("hotblock") || nm.includes("mze")) {
    return {
      familyId: fam || "heater_block",
      label: humanizeMeshName(fam || "heater_block"),
      stage: "heater_nozzle",
      isStructural: false,
      heatWeight: 0.3,
      roleNote: "Rigid copper heater / melt block — its tip mass drives cantilever bending in the heatbreak above it.",
      explodeCadM: [0, 0, -0.038],
    };
  }
  if (
    nm.includes("heatbreak") ||
    nm.includes("throat") ||
    nm.includes("standoff") ||
    nm.includes("ti_posts") ||
    nm === "dav_ti" ||
    nm.includes("monocoque_shell") ||
    nm.includes("chube_body")
  ) {
    return {
      familyId: fam || "heatbreak_neck",
      label: humanizeMeshName(fam || "heatbreak_neck"),
      stage: "hotend_neck",
      isStructural: true,
      heatWeight: 1.0,
      roleNote: "Primary load-bearing hotend neck / titanium structural shell — concentrates peak cantilever bending.",
      explodeCadM: [0, 0, -0.022],
    };
  }
  if (nm.includes("heatsink") || nm.includes("spring")) {
    return {
      familyId: fam || "hotend_heatsink",
      label: humanizeMeshName(fam || "hotend_heatsink"),
      stage: "hotend_neck",
      isStructural: true,
      heatWeight: 0.48,
      roleNote: "Upper hotend heatsink body — stiffer than the slender heatbreak neck below it.",
      explodeCadM: [0, 0, -0.004],
    };
  }

  // --- Spatial disambiguation for multi-part body groups (e.g. StealthBurner's 17 `sb_body_*` solids,
  //     VzPrinthead's `part_*` / `solid`, XOL `xol_body`) using real 3D CAD coordinates (mm) ---
  if (
    defaultStage === "chassis" &&
    centerCadMm &&
    (nm === "sb_body" || nm === "xol_body" || nm === "solid" || nm === "part" || nm.startsWith("body"))
  ) {
    // 1. Rear X-carriage mount halves & belt tie sitting near the MGN rail (Y >= -8 mm, Z <= +15 mm)
    if (cy >= -8 && cz <= 15) {
      const side = cx > 4 ? "Right" : cx < -4 ? "Left" : "Rear";
      const sideKey = side.toLowerCase();
      return {
        familyId: `${fam || "th"}_carriage_${sideKey}`,
        label: `X-Carriage Frame (${side})`,
        stage: "backplate",
        isStructural: true,
        heatWeight: 0.95,
        roleNote:
          "Bolted X-carriage frame half clamping directly to the MGN linear carriage & XY belts.",
        explodeCadM: [cx > 4 ? 0.022 : cx < -4 ? -0.022 : 0, -0.014, cz < -18 ? -0.018 : 0],
      };
    }
    // 2. Lower hotend mount carrier / clamshell (Z < +12 mm, forward of carriage)
    if (cz < 12 && cy < -8) {
      const isFront = cy < -33;
      return {
        familyId: `${fam || "th"}_hotend_carrier_${isFront ? "front" : "rear"}`,
        label: isFront ? "Hotend Mount Front Clamshell" : "Hotend Mount Rear Carrier",
        stage: isFront ? "chassis_clamshell" : "chassis_carrier",
        isStructural: true,
        heatWeight: isFront ? 0.72 : 0.95,
        roleNote: isFront
          ? "Front printed clamshell clamping the hotend collar against the rear carrier."
          : "Primary structural hotend carrier transferring nozzle bending loads to the X-carriage.",
        explodeCadM: [0, isFront ? -0.038 : -0.014, -0.024],
      };
    }
    // 3. Upper rear cable cover / strain relief (Z >= +15 mm, behind rail Y > +2 mm)
    if (cz >= 15 && cy > 2) {
      return {
        familyId: `${fam || "th"}_rear_cable_cover`,
        label: "Extruder Rear Cable Cover",
        stage: "cooling_shroud",
        isStructural: false,
        heatWeight: 0.14,
        roleNote: "Rear wiring / strain-relief cover behind the extruder motor (non-structural rider).",
        explodeCadM: [cx < -5 ? -0.024 : 0, 0.036, 0.032],
      };
    }
    // 4. Right-side PCB / toolhead electronics cover (Z >= +15 mm, X > +12 mm)
    if (cz >= 15 && cx > 12) {
      return {
        familyId: `${fam || "th"}_pcb_cover`,
        label: "Toolhead PCB / Side Cover",
        stage: "cooling_shroud",
        isStructural: false,
        heatWeight: 0.16,
        roleNote: "Side electronics / PCB housing cover (non-structural upper mass rider).",
        explodeCadM: [0.036, -0.008, 0.028],
      };
    }
    // 5. Upper Clockwork 2 / Direct-Drive Extruder body, motor plate & guidler (Z >= +15 mm)
    if (cz >= 15) {
      const isMotorPlate = cy > -14;
      const isLatchOrGuidler = cz >= 54 || cx < -8;
      return {
        familyId: isMotorPlate
          ? `${fam || "th"}_motor_plate`
          : isLatchOrGuidler
            ? `${fam || "th"}_extruder_latch`
            : `${fam || "th"}_extruder_main_body`,
        label: isMotorPlate
          ? "CW2 / Extruder Motor Plate"
          : isLatchOrGuidler
            ? "CW2 Extruder Latch & Guidler"
            : "CW2 / Extruder Main Body",
        stage: isMotorPlate
          ? "chassis_clamshell"
          : isLatchOrGuidler
            ? "cooling_shroud"
            : "chassis_extruder",
        isStructural: !isLatchOrGuidler,
        heatWeight: isMotorPlate ? 0.68 : isLatchOrGuidler ? 0.22 : 0.58,
        roleNote: isMotorPlate
          ? "Structural extruder motor plate carrying NEMA14 pitch inertia above the X-rail."
          : isLatchOrGuidler
            ? "Filament tension latch / guidler arm (non-structural upper mass)."
            : "Direct-drive extruder main body seated above the X-carriage.",
        explodeCadM: [
          isLatchOrGuidler ? -0.024 : 0,
          isMotorPlate ? 0.006 : -0.024,
          isLatchOrGuidler ? 0.056 : 0.036,
        ],
      };
    }
  }

  // --- Main Structural Chassis / Body ---
  return {
    familyId: fam || (defaultStage === "hotend_neck" ? "hotend_body" : "toolhead_chassis"),
    label: fam ? humanizeMeshName(fam) : defaultStage === "hotend_neck" ? "Hotend Body" : "Toolhead Chassis",
    stage: defaultStage,
    isStructural: true,
    heatWeight: 1.0,
    roleNote:
      defaultStage === "hotend_neck"
        ? "Structural hotend body carrying nozzle acceleration loads."
        : "Primary structural toolhead chassis carrying carriage-to-hotend bending & torsion.",
    explodeCadM: defaultStage === "hotend_neck" ? [0, 0, -0.018] : [0, 0, 0],
  };
}

export function classifyToolheadSubmeshStage(
  meshName: string,
  defaultStage: "chassis" | "hotend_neck"
): ToolheadPartStage {
  return describeToolheadSubmesh(meshName, defaultStage).stage;
}

/**
 * Explode individual sub-meshes inside a CAD toolhead or hotend GLB in CAD local space (SI meters)
 * so every physical component separates clearly while retaining live FEA & Modal elastic deformation.
 */
export function applyCadSubmeshExplode(
  cadGroup: THREE.Group,
  defaultStage: "chassis" | "hotend_neck"
): void {
  const rootBox = new THREE.Box3().setFromObject(cadGroup);
  const rootCenter = rootBox.isEmpty()
    ? new THREE.Vector3(0, 0, 0)
    : rootBox.getCenter(new THREE.Vector3());

  const tmpBox = new THREE.Box3();
  const tmpCenter = new THREE.Vector3();
  const radial = new THREE.Vector3();

  cadGroup.traverse((obj) => {
    if (!(obj instanceof THREE.Mesh) || !obj.visible) return;
    let centerCadMm: [number, number, number] | undefined;
    if (obj.geometry) {
      if (!obj.geometry.boundingBox) obj.geometry.computeBoundingBox();
      if (obj.geometry.boundingBox) {
        obj.geometry.boundingBox.getCenter(tmpCenter);
        centerCadMm = [tmpCenter.x * 1000, tmpCenter.y * 1000, tmpCenter.z * 1000];
      }
    }
    const desc = describeToolheadSubmesh(obj.name || "", defaultStage, centerCadMm);
    let dx = desc.explodeCadM[0];
    let dy = desc.explodeCadM[1];
    let dz = desc.explodeCadM[2];

    if (desc.explodeRadialM && desc.explodeRadialM > 0 && obj.geometry) {
      obj.geometry.computeBoundingBox();
      if (obj.geometry.boundingBox) {
        tmpBox.copy(obj.geometry.boundingBox);
        tmpBox.getCenter(tmpCenter);
        radial.subVectors(tmpCenter, rootCenter);
        const len = radial.length();
        if (len > 1e-4) {
          radial.multiplyScalar(desc.explodeRadialM / len);
          dx += radial.x;
          dy += radial.y;
          dz += radial.z;
        }
      }
    }

    if (Math.abs(dx) > 1e-6 || Math.abs(dy) > 1e-6 || Math.abs(dz) > 1e-6) {
      obj.position.x += dx;
      obj.position.y += dy;
      obj.position.z += dz;
    }
  });
}

export interface ResolvedToolheadPartVisual {
  partId: string;
  stage: ToolheadPartStage;
  heatScore01: number;
  heatColor: THREE.Color;
  heatEmissive: THREE.Color;
  heatEmissiveIntensity: number;
  inspectInfo: InspectedPartInfo;
}

/**
 * Find the best matching backend static + modal toolhead part for a given stage or CAD mesh name,
 * and compute its individual part identity, differential heat color, and rich Part Inspector payload.
 */
export function resolveToolheadPartVisual(params: {
  stage: ToolheadPartStage;
  meshName?: string;
  centerCadMm?: [number, number, number];
  defaultStage?: "chassis" | "hotend_neck";
  staticParts: ToolheadPartStaticData[];
  modalParts?: ToolheadPartModalData[];
  isModalMode: boolean;
  isHeatMode: boolean;
  fallbackInspect: InspectedPartInfo;
}): ResolvedToolheadPartVisual {
  const {
    stage,
    meshName,
    centerCadMm,
    defaultStage = "chassis",
    staticParts,
    modalParts,
    isModalMode,
    isHeatMode,
    fallbackInspect,
  } = params;

  const subDesc = meshName ? describeToolheadSubmesh(meshName, defaultStage, centerCadMm) : null;
  const effStage: ToolheadPartStage = subDesc?.stage ?? stage;
  const nmLow = (meshName || "").toLowerCase();

  // Prefer exact CAD 3D recovery sub-part match if available, then stage match
  const matchStatic =
    (nmLow
      ? staticParts.find(
          (p) =>
            p.source === "cad_3d_recovery" &&
            (nmLow.includes(p.id.replace(/^cad_(th|he)_/, "").toLowerCase()) ||
              p.name.toLowerCase().includes(nmLow))
        )
      : undefined) ??
    staticParts.find((p) => p.stage === effStage) ??
    (effStage === "chassis" ? staticParts.find((p) => p.stage === "chassis_carrier") : undefined) ??
    (effStage === "chassis_carrier" || effStage === "chassis_clamshell" || effStage === "chassis_extruder"
      ? staticParts.find((p) => p.stage === "chassis")
      : undefined) ??
    (effStage === "backplate" ? staticParts.find((p) => p.stage === "chassis" || p.stage === "chassis_carrier") : undefined) ??
    (effStage === "hotend_brace" ? staticParts.find((p) => p.stage === "hotend_neck") : undefined) ??
    staticParts.find((p) => p.stage === "chassis" || p.stage === "chassis_carrier");

  const matchModal =
    (matchStatic ? modalParts?.find((m) => m.id === matchStatic.id) : undefined) ??
    modalParts?.find((m) => m.stage === effStage) ??
    (effStage === "chassis" ? modalParts?.find((m) => m.stage === "chassis_carrier") : undefined) ??
    (effStage === "backplate" ? modalParts?.find((m) => m.stage === "chassis" || m.stage === "chassis_carrier") : undefined);

  const isPartStructural = subDesc ? subDesc.isStructural : (matchStatic?.is_structural ?? true);
  const heatW = subDesc ? subDesc.heatWeight : 1.0;

  // Compute normalized heat intensity [0..1] for differential per-part coloring
  let heat01 = 0.08;
  if (!isPartStructural) {
    // Non-structural riders (shroud, fans, PCBs, fasteners, silicone sock, heater block)
    // stay cool slate-blue in FEA Stress mode, or show mild kinetic shake tint in Modal mode
    heat01 = isModalMode
      ? Math.min(0.45, Math.max(0.06, ((matchModal?.kinetic_share_pct ?? 0) / 45) * 0.42 * Math.max(heatW, 0.4)))
      : 0.06 * heatW;
  } else if (isModalMode && matchModal) {
    const rawModal01 = Math.min(1.0, Math.max(0.08, (matchModal.modal_flex_pct / 50) ** 0.72));
    heat01 = Math.max(0.08, Math.min(1.0, rawModal01 * heatW));
  } else if (matchStatic) {
    const compScore = Math.min(1.0, (matchStatic.compliance_share_pct / 42) ** 0.72);
    const stressRatio =
      matchStatic.stress_mpa != null && matchStatic.yield_mpa && matchStatic.yield_mpa > 0
        ? Math.min(1.0, (matchStatic.stress_mpa / matchStatic.yield_mpa) * 4.0)
        : 0;
    heat01 = Math.max(0.08, Math.min(1.0, Math.max(compScore, stressRatio) * heatW));
  }

  const heatColor =
    !isPartStructural && !isModalMode ? new THREE.Color(0x334155) : stressColor(heat01);
  const heatEmissive = isHeatMode ? heatColor.clone().multiplyScalar(0.35) : new THREE.Color(0x000000);
  const heatEmissiveIntensity = isHeatMode ? 0.14 + heat01 * 0.42 : 0;

  const partId = matchStatic?.id ?? `toolhead_${effStage}`;
  const uniqueKey = subDesc ? `th_sub_${subDesc.familyId}` : `th_part_${partId}`;
  const displayName = subDesc ? subDesc.label : (matchStatic?.name ?? fallbackInspect.name);

  const inspectInfo: InspectedPartInfo = matchStatic
    ? {
        key: uniqueKey,
        name: displayName,
        subsystem: "toolhead",
        partGroupId: "toolhead_mount",
        role: "toolhead",
        material: matchStatic.material,
        profile:
          matchStatic.source === "cad_3d_recovery"
            ? "3D CAD Tet4 Recovery"
            : isPartStructural
              ? `Structural · ${matchStatic.name}`
              : "Non-Structural Rider",
        complianceSharePct: isPartStructural
          ? Number((matchStatic.compliance_share_pct * heatW).toFixed(1))
          : undefined,
        localFlexUm: isPartStructural
          ? Number((matchStatic.local_flex_um * heatW).toFixed(3))
          : 0,
        nodeDispUm: matchStatic.total_disp_um,
        bracketTwistMrad:
          isPartStructural && matchStatic.twist_mrad > 0
            ? Number((matchStatic.twist_mrad * heatW).toFixed(3))
            : undefined,
        partStiffnessNPerMm: isPartStructural ? (matchStatic.stiffness_n_per_mm ?? undefined) : undefined,
        unitMassG: matchStatic.mass_g > 0 ? matchStatic.mass_g : undefined,
        stressMpa:
          isPartStructural && matchStatic.stress_mpa != null
            ? Number((matchStatic.stress_mpa * heatW).toFixed(2))
            : undefined,
        yieldMpa: isPartStructural ? (matchStatic.yield_mpa ?? undefined) : undefined,
        modalFlexPct: isPartStructural && matchModal ? Number((matchModal.modal_flex_pct * heatW).toFixed(1)) : 0,
        kineticSharePct: matchModal?.kinetic_share_pct,
        partSource: matchStatic.source,
        isStructural: isPartStructural,
        toolheadPartKey: `th_part_${partId}`,
        whyItHurts: subDesc ? subDesc.roleNote : matchStatic.why_it_hurts,
        actionHint: matchStatic.action_hint,
      }
    : {
        ...fallbackInspect,
        key: uniqueKey,
        name: displayName,
        profile: isPartStructural ? "Structural Sub-Part" : "Non-Structural Rider",
        complianceSharePct:
          isPartStructural && fallbackInspect.complianceSharePct != null
            ? Number((fallbackInspect.complianceSharePct * heatW).toFixed(1))
            : undefined,
        stressMpa:
          isPartStructural && fallbackInspect.stressMpa != null
            ? Number((fallbackInspect.stressMpa * heatW).toFixed(2))
            : undefined,
        isStructural: isPartStructural,
        toolheadPartKey: `th_part_${partId}`,
        whyItHurts: subDesc ? subDesc.roleNote : fallbackInspect.whyItHurts,
      };

  return {
    partId,
    stage: effStage,
    heatScore01: heat01,
    heatColor,
    heatEmissive,
    heatEmissiveIntensity,
    inspectInfo,
  };
}

export type IntraPartFieldMetric = "combined" | "stress" | "flex";

/**
 * Precomputed 3D Tet4 intra-part field interpolator for fast per-vertex FEA coloring.
 */
export interface PreparedIntraPartField {
  pts: Float32Array; // length 3*K in CAD mm
  vals: Float32Array; // length K normalized [0..1]
  count: number;
  maxStressMpa: number;
  maxFlexUm: number;
}

/**
 * Look up an IntraPartField from printer-wide `part_fields` / `mode_part_fields`,
 * gracefully falling back to legacy `toolhead_part_fields` / `toolhead_mode_fields`.
 */
export function resolveIntraPartField(params: {
  partFields?: Record<string, CadIntraPartFieldData> | null;
  legacyToolheadFields?: Record<string, CadIntraPartFieldData> | null;
  candidateKeys: Array<string | undefined | null>;
  assetId?: string | null;
}): { key: string; field: CadIntraPartFieldData } | null {
  const { partFields, legacyToolheadFields, candidateKeys, assetId } = params;
  const dicts = [partFields, legacyToolheadFields].filter(
    (d): d is Record<string, CadIntraPartFieldData> => Boolean(d)
  );
  if (dicts.length === 0) return null;

  const cleanedKeys = candidateKeys.filter((k): k is string => Boolean(k && k.length > 0));
  for (const dict of dicts) {
    for (const k of cleanedKeys) {
      const hit = dict[k];
      if (hit && hit.points_mm && hit.points_mm.length >= 4) {
        return { key: k, field: hit };
      }
      const cadHit = dict[`cad_${k}`];
      if (cadHit && cadHit.points_mm && cadHit.points_mm.length >= 4) {
        return { key: `cad_${k}`, field: cadHit };
      }
    }
  }

  if (assetId) {
    const targetLow = assetId.toLowerCase();
    for (const dict of dicts) {
      for (const [k, f] of Object.entries(dict)) {
        if (!f || !f.points_mm || f.points_mm.length < 4) continue;
        if (f.asset_id && f.asset_id.toLowerCase() === targetLow) {
          return { key: k, field: f };
        }
      }
    }
  }
  return null;
}

export interface SubpartStrainSummary {
  index: number;
  name: string;
  strainEnergyPct: number;
  maxStressMpa: number;
  maxFlexUm: number;
  pointCount: number;
}

/**
 * Compute per-subpart strain energy share and peak stress/flex from an IntraPartField.
 */
export function computeSubpartStrainBreakdown(
  field: CadIntraPartFieldData | null | undefined
): SubpartStrainSummary[] {
  if (!field) return [];
  const rawSubparts = field.subparts;
  const subIdx = field.subpart_idx;
  const stresses = field.stress_mpa ?? [];
  const flexes = field.flex_um ?? [];
  const strainNorm = field.strain_norm ?? [];
  const flexNorm = field.flex_norm ?? [];

  // Case 1: Backend already provided rich subpart objects with strain_energy_pct
  if (rawSubparts && rawSubparts.length > 0) {
    const hasPrecomputed = rawSubparts.some(
      (sp) => typeof sp !== "string" && typeof sp.strain_energy_pct === "number"
    );
    if (hasPrecomputed) {
      return rawSubparts
        .map((sp, idx) => {
          if (typeof sp === "string") {
            return {
              index: idx,
              name: sp,
              strainEnergyPct: 0,
              maxStressMpa: 0,
              maxFlexUm: 0,
              pointCount: 0,
            };
          }
          return {
            index: idx,
            name: sp.label || sp.name || `Subpart ${idx + 1}`,
            strainEnergyPct: sp.strain_energy_pct ?? 0,
            maxStressMpa: sp.max_stress_mpa ?? 0,
            maxFlexUm: sp.max_flex_um ?? 0,
            pointCount: sp.point_count ?? 0,
          };
        })
        .sort((a, b) => b.strainEnergyPct - a.strainEnergyPct);
    }
  }

  // Case 2: Aggregate from per-point subpart_idx + stress_mpa / strain_norm
  if (!subIdx || subIdx.length === 0) return [];
  const buckets = new Map<
    number,
    { energyWeight: number; maxStress: number; maxFlex: number; count: number }
  >();
  let totalEnergy = 0;

  for (let i = 0; i < subIdx.length; i++) {
    const idx = subIdx[i];
    if (idx == null || idx < 0) continue;
    const s = stresses[i] ?? strainNorm[i] ?? 0;
    const f = flexes[i] ?? flexNorm[i] ?? 0;
    const w = s * s + 1e-12;
    totalEnergy += w;
    let b = buckets.get(idx);
    if (!b) {
      b = { energyWeight: 0, maxStress: 0, maxFlex: 0, count: 0 };
      buckets.set(idx, b);
    }
    b.energyWeight += w;
    if (s > b.maxStress) b.maxStress = s;
    if (f > b.maxFlex) b.maxFlex = f;
    b.count += 1;
  }

  if (buckets.size === 0) return [];
  const denom = Math.max(totalEnergy, 1e-12);
  const out: SubpartStrainSummary[] = [];
  for (const [idx, b] of buckets.entries()) {
    const rawEntry = rawSubparts?.[idx];
    const name =
      typeof rawEntry === "string"
        ? rawEntry
        : rawEntry?.label || rawEntry?.name || `Region ${idx + 1}`;
    out.push({
      index: idx,
      name,
      strainEnergyPct: (b.energyWeight / denom) * 100,
      maxStressMpa: b.maxStress,
      maxFlexUm: b.maxFlex,
      pointCount: b.count,
    });
  }
  return out.sort((a, b) => b.strainEnergyPct - a.strainEnergyPct);
}

export function prepareIntraPartField(
  field: CadIntraPartFieldData | undefined,
  isModalMode: boolean,
  metric: IntraPartFieldMetric = "combined"
): PreparedIntraPartField | null {
  if (!field || !field.points_mm || field.points_mm.length < 4) return null;
  const k = field.points_mm.length;
  const pts = new Float32Array(k * 3);
  const raw = new Float32Array(k);

  const stresses = field.stress_mpa ?? [];
  const flexes = field.flex_um ?? [];
  let maxStress = field.max_stress_mpa ?? 0;
  let maxFlex = field.max_flex_um ?? 0;
  for (let i = 0; i < k; i++) {
    if ((stresses[i] ?? 0) > maxStress) maxStress = stresses[i] ?? 0;
    if ((flexes[i] ?? 0) > maxFlex) maxFlex = flexes[i] ?? 0;
  }

  if (isModalMode && field.strain_norm && field.flex_norm) {
    for (let i = 0; i < k; i++) {
      const p = field.points_mm[i];
      pts[i * 3] = p[0];
      pts[i * 3 + 1] = p[1];
      pts[i * 3 + 2] = p[2];
      const sn = field.strain_norm[i] ?? 0;
      const fn = field.flex_norm[i] ?? 0;
      if (metric === "stress") {
        raw[i] = Math.min(1, Math.max(0, Math.pow(sn, 0.65)));
      } else if (metric === "flex") {
        raw[i] = Math.min(1, Math.max(0, Math.pow(fn, 0.72)));
      } else {
        raw[i] = Math.min(1, Math.max(0, 0.58 * Math.pow(sn, 0.65) + 0.42 * Math.pow(fn, 0.72)));
      }
    }
    return { pts, vals: raw, count: k, maxStressMpa: maxStress, maxFlexUm: maxFlex };
  }

  const sSorted = [...stresses].sort((a, b) => a - b);
  const fSorted = [...flexes].sort((a, b) => a - b);
  const sRef = Math.max(sSorted[Math.floor(0.95 * (sSorted.length - 1))] ?? 0, 1e-4);
  const fRef = Math.max(fSorted[Math.floor(0.95 * (fSorted.length - 1))] ?? 0, 1e-4);

  for (let i = 0; i < k; i++) {
    const p = field.points_mm[i];
    pts[i * 3] = p[0];
    pts[i * 3 + 1] = p[1];
    pts[i * 3 + 2] = p[2];
    const sn = Math.min(1, (stresses[i] ?? 0) / sRef);
    const fn = Math.min(1, (flexes[i] ?? 0) / fRef);
    if (metric === "stress") {
      raw[i] = Math.min(1, Math.max(0, Math.pow(sn, 0.62)));
    } else if (metric === "flex") {
      raw[i] = Math.min(1, Math.max(0, Math.pow(fn, 0.72)));
    } else {
      raw[i] = Math.min(1, Math.max(0, 0.62 * Math.pow(sn, 0.62) + 0.38 * Math.pow(fn, 0.72)));
    }
  }
  return { pts, vals: raw, count: k, maxStressMpa: maxStress, maxFlexUm: maxFlex };
}

/**
 * Create a per-vertex color callback that paints continuous intra-part flex & bending-strain
 * contours directly onto a structural CAD mesh (using exact 3D Tet4 recovery when available,
 * or an analytical clamped-root-to-tip cantilever gradient fallback).
 */
export function createIntraPartVertexColorFn(params: {
  isStructural: boolean;
  baseHeat01: number;
  isModalMode: boolean;
  defaultStage: "chassis" | "hotend_neck";
  preparedField?: PreparedIntraPartField | null;
  port0Mm?: [number, number, number];
  port1Mm?: [number, number, number];
}): ((xMm: number, yMm: number, zMm: number, outColor: THREE.Color) => void) | undefined {
  const {
    isStructural,
    baseHeat01,
    isModalMode,
    defaultStage,
    preparedField,
    port0Mm = [0, 0, 0],
    port1Mm = defaultStage === "hotend_neck" ? [0, 0, -60] : [0, -22, 0],
  } = params;

  if (!isStructural) return undefined;

  // Dynamic visual gain so structural parts with meaningful flex span blue -> cyan -> green -> yellow -> red
  const gain = Math.min(1.0, Math.max(0.45, 0.35 + baseHeat01 * 0.85));

  if (preparedField && preparedField.count >= 4) {
    const { pts, vals, count } = preparedField;
    return (xMm: number, yMm: number, zMm: number, outColor: THREE.Color) => {
      // 4-nearest inverse-distance weighted (IDW) interpolation in CAD mm space
      let d0 = 1e18, d1 = 1e18, d2 = 1e18, d3 = 1e18;
      let v0 = 0, v1 = 0, v2 = 0, v3 = 0;
      for (let i = 0; i < count; i++) {
        const dx = xMm - pts[i * 3];
        const dy = yMm - pts[i * 3 + 1];
        const dz = zMm - pts[i * 3 + 2];
        const d2Sq = dx * dx + dy * dy + dz * dz;
        if (d2Sq < d0) {
          d3 = d2; v3 = v2;
          d2 = d1; v2 = v1;
          d1 = d0; v1 = v0;
          d0 = d2Sq; v0 = vals[i];
        } else if (d2Sq < d1) {
          d3 = d2; v3 = v2;
          d2 = d1; v2 = v1;
          d1 = d2Sq; v1 = vals[i];
        } else if (d2Sq < d2) {
          d3 = d2; v3 = v2;
          d2 = d2Sq; v2 = vals[i];
        } else if (d2Sq < d3) {
          d3 = d2Sq; v3 = vals[i];
        }
      }
      const eps = 4.0; // 2 mm smoothing radius squared
      const w0 = 1 / (d0 + eps);
      const w1 = 1 / (d1 + eps);
      const w2 = 1 / (d2 + eps);
      const w3 = 1 / (d3 + eps);
      const interp = (w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3) / (w0 + w1 + w2 + w3);
      const t = Math.min(1.0, Math.max(0.04, interp * gain));
      stressColorInto(t, outColor);
    };
  }

  // Analytical cantilever load-path fallback for assets not yet re-baked with 3D field clouds
  const dxP = port1Mm[0] - port0Mm[0];
  const dyP = port1Mm[1] - port0Mm[1];
  const dzP = port1Mm[2] - port0Mm[2];
  const lenP = Math.max(Math.hypot(dxP, dyP, dzP), 12.0);
  const ux = dxP / lenP;
  const uy = dyP / lenP;
  const uz = dzP / lenP;

  return (xMm: number, yMm: number, zMm: number, outColor: THREE.Color) => {
    const rx = xMm - port0Mm[0];
    const ry = yMm - port0Mm[1];
    const rz = zMm - port0Mm[2];
    const sRaw = (rx * ux + ry * uy + rz * uz) / lenP;
    const s = Math.min(1.2, Math.max(0, sRaw));
    const rDist = Math.hypot(rx, ry, rz);

    // Bending moment peaks just outside the clamped bolt pad (root fillet / heatbreak throat),
    // while relative elastic flex grows smoothly along the cantilever toward Port 1.
    let localVal: number;
    if (defaultStage === "hotend_neck") {
      const rootStrain = Math.exp(-Math.pow((s - 0.22) / 0.28, 2));
      const tipFlex = Math.pow(Math.min(1, s), 1.35);
      localVal = isModalMode ? 0.45 * rootStrain + 0.55 * tipFlex : 0.72 * rootStrain + 0.28 * tipFlex;
    } else {
      const clampFade = Math.min(1, Math.max(0, (rDist - 5.0) / 10.0));
      const neckStrain = clampFade * Math.exp(-Math.pow((s - 0.42) / 0.42, 2));
      const overhangFlex = clampFade * Math.pow(Math.min(1, s), 1.2);
      localVal = isModalMode ? 0.42 * neckStrain + 0.58 * overhangFlex : 0.65 * neckStrain + 0.35 * overhangFlex;
    }
    const t = Math.min(1.0, Math.max(0.05, localVal * gain));
    stressColorInto(t, outColor);
  };
}
