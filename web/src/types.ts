export interface MaterialItem {
  id: string;
  name: string;
  youngs_modulus_pa: number;
  density_kg_m3: number;
  yield_strength_pa: number;
  color_hex: string;
}

export interface ProfileItem {
  id: string;
  name: string;
  shape: string;
  width_m: number;
  height_m: number;
}

export interface PresetItem {
  id: string;
  name: string;
  architecture: string;
  description: string;
}

export interface FilamentItem {
  id: string;
  name: string;
  print_temp_c: number;
  bed_temp_c: number;
  glass_transition_c: number;
}

export interface PartPresetItem {
  id: string;
  name: string;
}

export interface StepperCatalogItem {
  preset_id: string;
  name: string;
  holding_torque_nm: number;
  rated_current_a: number;
  run_current_a: number;
  phase_resistance_ohm: number;
  inductance_mh: number;
  rotor_inertia_kg_m2: number;
  motor_mass_g: number;
  steps_per_rev: number;
}

export type ExtrusionFillingType =
  | "none"
  | "pu_foam"
  | "silica_sand"
  | "lead_shot"
  | "epoxy_granite"
  | "uhpc_grout";

export type BracketMaterialType =
  | "abs_printed"
  | "pet_cf_printed"
  | "pa12_cf_printed"
  | "ppa_cf_printed"
  | "pps_cf_printed"
  | "aluminum_6061_t6"
  | "aluminum_7075_t6";

export type ToolheadMountMaterial = BracketMaterialType;

export type ToolheadConstructionType =
  | "multi_piece_clamshell"
  | "compact_cage"
  | "plate_sandwich_cf"
  | "extruder_as_chassis"
  | "cnc_monoblock"
  | "cantilever_bowden";

export type XRailMountType = "front" | "top" | "bottom" | "dual_top_bottom";

export type YRailMountType = "top" | "inner" | "bottom";

export type CornerJointType =
  | "rigid"
  | "l_bracket"
  | "blind_joint"
  | "gusset_plate"
  | "corner_cube";

export type RailPreloadType = "ZF" | "Z0" | "Z1" | "ZA" | "ZB";

export type RailQualityType =
  | "budget_generic"
  | "midgrade_matched"
  | "hiwin_genuine"
  | "precision_thk_nsk";

export type RailBlockType = "C" | "H";

export type RailBackerMaterialType = "none" | "steel_4140" | "titanium_grade5";

export type BearingClearanceClass = "C2" | "CN" | "C3";

export type BedMountType =
  | "kinematic_3pt"
  | "trident_3pt_standoffs"
  | "whopping_pochard_kinematic"
  | "rigid_4pt"
  | "floating_4pt"
  | "silicone_4pt"
  | "springs_4pt";

export type FrameFeetType =
  | "rigid_bolted"
  | "rubber_feet"
  | "anti_vib_pads"
  | "paver_on_foam";

export type TopBraceType =
  | "none"
  | "cnc_halo_ring"
  | "rear_v_and_front_corners"
  | "corner_45_bars";

export type FrontBraceType =
  | "none"
  | "cnc_portal_frame"
  | "header_and_skirt"
  | "corner_45_bars";

export interface BraceCatalogItem {
  id: string;
  label: string;
  short_label: string;
  kind: string;
  thickness_mm: number;
  description: string;
  cost_usd: number;
}

export type ZProbeType = "static_eddy" | "klicky_inductive" | "cnc_tap" | "printed_tap";

export type BeltRoutingType = "stock_corexy" | "monolith_short";

export type BeltType =
  | "gt2_6mm"
  | "gt2_9mm"
  | "gt2_12mm"
  | "gt2_15mm"
  | "gt2_dual_9mm"
  | "3gt_9mm"
  | "3gt_12mm"
  | "3gt_15mm"
  | "at3_10mm"
  | "at3_16mm"
  | "at3_gen3_16mm"
  | "htd3m_15mm"
  | "at5_16mm";

export type BeltValidationTier = "proven" | "experimental" | "theoretical";

export interface BeltCatalogItem {
  id: BeltType;
  label: string;
  profile_family: string;
  pitch_mm: number;
  width_mm: number;
  hardware_width_mm: number;
  default_core: "fiberglass" | "steel" | "aramid";
  body_material: "epdm" | "tpu";
  ea_default_core_n: number;
  linear_mass_kg_m: number;
  max_working_tension_n: number;
  tooth_shear_k_n_per_m_mm: number;
  min_pulley_teeth: number;
  min_back_idler_mm: number;
  max_temp_c: number;
  cost_usd_per_m: number;
  validation_tier: BeltValidationTier;
  theoretical_summary: string;
  practical_issues: string[];
}

export type ZRailProfileType =
  | "mgn12_rail"
  | "mgn9_rail"
  | "steel_rod_8mm"
  | "steel_rod_10mm"
  | "steel_rod_12mm"
  | "none";

export type ZLeadscrewType =
  | "auto"
  | "tr8x4_pom"
  | "tr8x8_pom"
  | "tr8x8_brass"
  | "sfu1204_bk10"
  | "gt2_6mm_z_belt"
  | "gt2_9mm_z_belt";

export type HotendMountStyle =
  | "groove_mount"
  | "rigid_4bolt_m3"
  | "dragon_4bolt_m25"
  | "chube_6bolt_m3_18pcd";

export type AdjointLens = "print_impact" | "compliance" | "modal" | "mass";

export type SubsystemKey = "toolhead" | "gantry" | "frame" | "bed";

export type DriveType = "open_loop_stepper" | "closed_loop_stepper" | "foc_servo";

export interface BedMountCatalogItem {
  id: BedMountType;
  name: string;
  short_name: string;
  mount_points?: number;
  k_axial_n_per_mm?: number;
  k_lateral_n_per_mm?: number;
  cost_usd?: number;
  notes?: string;
}

export interface FrameFeetCatalogItem {
  id: FrameFeetType;
  name: string;
  short_name: string;
  k_axial_n_per_mm?: number;
  k_lateral_n_per_mm?: number;
  notes?: string;
}

export interface RailMountOption<T extends string = string> {
  id: T;
  name: string;
  short_name: string;
  notes: string;
}

export interface ExtrusionFillingInfo {
  id: ExtrusionFillingType;
  name: string;
  short_name: string;
  density_kg_m3: number;
  youngs_modulus_pa: number;
  bonded: boolean;
  damping_boost: number;
  cost_per_kg_usd: number;
  color_hex: string;
  notes: string;
}

export interface GantryBracketInfo {
  id: BracketMaterialType;
  name: string;
  short_name: string;
  youngs_modulus_gpa: number;
  hdt_c: number;
  creep_tension_limit_n: number;
  cnc_machined: boolean;
  kit_cost_usd: number;
  carriage_mass_delta_g: number;
  color_hex: string;
}

export interface GantryWeightClassItem {
  id: string;
  name: string;
  short_name: string;
  carriage_mass_g: number;
  suggested_bracket_material: string;
  notes: string;
}

export interface ZProbeCatalogItem {
  id: ZProbeType;
  name: string;
  short_name: string;
  added_mass_g?: number;
  notes?: string;
}

export interface BeltRoutingCatalogItem {
  id: BeltRoutingType;
  name: string;
  short_name: string;
  notes?: string;
}

export interface ToolheadConstructionCatalogItem {
  id: ToolheadConstructionType;
  name: string;
  short_name: string;
  body_stiffness_base_n_per_mm: number;
  torsion_mult: number;
  mass_delta_g?: number;
  notes: string;
}

export interface ToolheadPresetItem {
  preset_id: string;
  name: string;
  total_mass_g: number;
  cog_offset_mm: [number, number, number];
  nozzle_offset_mm: [number, number, number];
  mount_material: ToolheadMountMaterial;
  body_construction?: ToolheadConstructionType | null;
  internal_stiffness_n_per_mm?: number;
  z_probe_type?: ZProbeType;
}

export type ToolheadPresetData = ToolheadPresetItem;

export interface HotendPresetItem {
  id: string;
  name: string;
  short_name?: string;
  mount_style: string;
  mass_g?: number;
  overall_length_mm: number;
  heater_power_w: number;
  block_material: string;
  block_mass_g: number;
  nozzle_material: string;
  melt_zone_length_mm: number;
  heatbreak_material: string;
  heatbreak_length_mm: number;
  heatbreak_wall_mm: number;
  heatbreak_id_mm: number;
  heatsink_thermal_resistance_k_w: number;
  structural_stiffness_n_per_mm: number;
  max_temp_c: number;
  cost_usd?: number;
  notes?: string;
}

export interface ZDriveMechanismItem {
  id: string;
  name: string;
  kind: "leadscrew" | "ballscrew" | "belt";
  lead_mm?: number;
  diameter_mm?: number;
  backlash_um?: number;
  notes?: string;
}

export interface ZRailProfileItem {
  id: string;
  name: string;
  mass_kg_per_m?: number;
}

export interface CatalogResponse {
  materials: MaterialItem[];
  profiles: ProfileItem[];
  filaments: FilamentItem[];
  part_presets: PartPresetItem[];
  presets: PresetItem[];
  steppers?: StepperCatalogItem[];
  toolhead_presets?: ToolheadPresetItem[];
  toolhead_constructions?: ToolheadConstructionCatalogItem[];
  extrusion_fillings?: ExtrusionFillingInfo[];
  gantry_brackets?: GantryBracketInfo[];
  gantry_weight_classes?: GantryWeightClassItem[];
  x_rail_mounts?: RailMountOption<XRailMountType>[];
  y_rail_mounts?: RailMountOption<YRailMountType>[];
  bed_mounts?: BedMountCatalogItem[];
  frame_feet?: FrameFeetCatalogItem[];
  z_probes?: ZProbeCatalogItem[];
  belts?: BeltCatalogItem[];
  belt_routings?: BeltRoutingCatalogItem[];
  hotend_presets?: HotendPresetItem[];
  z_drive_mechanisms?: ZDriveMechanismItem[];
  z_rail_profiles?: ZRailProfileItem[];
  top_braces?: BraceCatalogItem[];
  front_braces?: BraceCatalogItem[];
  rail_qualities?: Array<{
    id: RailQualityType;
    name: string;
    short_name: string;
    stiffness_mult: number;
    drag_mult: number;
    cost_mult: number;
    notes: string;
  }>;
}

export type CatalogData = CatalogResponse;

export interface SolidVoxelData {
  center_mm: [number, number, number];
  disp_mm: [number, number, number];
  von_mises_mpa: number;
  fixed: boolean;
  loaded: boolean;
}

export interface SolidFEAResponse {
  part_name: string;
  material_id: string;
  is_fdm_anisotropic: boolean;
  voxel_size_mm: number;
  num_elements: number;
  num_nodes: number;
  num_dofs: number;
  mass_g: number;
  applied_force_n: [number, number, number];
  max_displacement_um: number;
  stiffness_n_per_mm: number;
  max_von_mises_mpa: number;
  max_z_tensile_mpa: number;
  yield_strength_mpa: number;
  z_layer_strength_mpa: number;
  safety_factor: number;
  failure_mode: string;
  voxels: SolidVoxelData[];
}

export interface TetSurfaceMeshData {
  vertices_mm: [number, number, number][];
  triangles: [number, number, number][];
  displacements_mm: [number, number, number][];
  von_mises_mpa: number[];
  fixed_mask: boolean[];
  loaded_mask: boolean[];
}

export interface TetFEAResponse {
  backend: string;
  element_type: string;
  part_name: string;
  material_id: string;
  is_fdm_anisotropic: boolean;
  mesh_size_mm: number;
  num_vertices: number;
  num_elements: number;
  num_dofs: number;
  mass_g: number;
  volume_mm3: number;
  applied_force_n: [number, number, number];
  max_displacement_um: number;
  stiffness_n_per_mm: number;
  max_von_mises_mpa: number;
  max_z_tensile_mpa: number;
  yield_strength_mpa: number;
  z_layer_strength_mpa: number;
  safety_factor: number;
  failure_mode: string;
  surface_mesh: TetSurfaceMeshData;
}

export type SolidFeaEngine = "tet10" | "hex8";

export interface TopologyHistoryPoint {
  iteration: number;
  compliance_mj: number;
  volume_fraction: number;
  mass_g: number;
  max_change: number;
}

export interface TopologyVoxelData {
  center_mm: [number, number, number];
  disp_mm: [number, number, number];
  von_mises_mpa: number;
  density: number;
  kept: boolean;
  fixed: boolean;
  loaded: boolean;
}

export interface TopologyOptResponse {
  part_name: string;
  material_id: string;
  is_fdm_anisotropic: boolean;
  voxel_size_mm: number;
  target_volume_fraction: number;
  achieved_volume_fraction: number;
  penalization: number;
  filter_radius_voxels: number;
  iterations_run: number;
  baseline_elements: number;
  optimized_elements: number;
  baseline_mass_g: number;
  optimized_mass_g: number;
  mass_reduction_pct: number;
  baseline_compliance_mj: number;
  optimized_compliance_mj: number;
  baseline_stiffness_n_per_mm: number;
  optimized_stiffness_n_per_mm: number;
  specific_stiffness_gain_pct: number;
  max_displacement_um: number;
  max_von_mises_mpa: number;
  safety_factor: number;
  failure_mode: string;
  applied_force_n: [number, number, number];
  history: TopologyHistoryPoint[];
  voxels: TopologyVoxelData[];
}

export interface TopoOptRequestParams {
  preset: string;
  materialId: string;
  forceN: [number, number, number];
  targetVolumeFraction: number;
  iterations: number;
  preserveBoundaries: boolean;
  fdmAnisotropic: boolean;
}

export interface CutListItem {
  part_code: string;
  name: string;
  profile_id: string;
  material_id: string;
  role: string;
  length_mm: number;
  quantity: number;
  total_length_mm: number;
  unit_mass_g: number;
  total_mass_g: number;
  unit_cost_usd: number;
  total_cost_usd: number;
  notes: string;
}

export interface BOMLineItem {
  sku: string;
  category: string;
  name: string;
  specification: string;
  quantity: number;
  unit: string;
  unit_cost_usd: number;
  total_cost_usd: number;
  total_mass_g: number;
}

export interface StockNestingSummary {
  profile_id: string;
  material_id: string;
  stock_length_mm: number;
  bars_required: number;
  used_length_mm: number;
  waste_length_mm: number;
  waste_pct: number;
  bar_cuts_mm: number[][];
}

export interface PrinterBOMResponse {
  printer_name: string;
  architecture: string;
  cut_list: CutListItem[];
  stock_nesting: StockNestingSummary[];
  items: BOMLineItem[];
  category_costs_usd: Record<string, number>;
  total_cost_usd: number;
  total_mass_kg: number;
  total_fasteners: number;
  total_belt_length_m: number;
  total_extrusion_length_m: number;
}

export interface CustomBracketOverrideData {
  label: string;
  stiffness_n_per_mm: number;
  mass_delta_g: number;
  source: string;
}

export interface PrinterSpecData {
  name: string;
  description: string;
  architecture: "corexy" | "cartesian_bedslinger" | "delta";
  frame: {
    width_mm: number;
    depth_mm: number;
    height_mm: number;
    profile: string;
    upright_profile?: string | null;
    material: string;
    bracing:
      | "none"
      | "corner_gussets"
      | "top_cross"
      | "rear_cross"
      | "side_cross"
      | "rear_and_sides"
      | "full_truss";
    brace_profile: string;
    brace_material: string;
    enclosed_panels: boolean;
    panel_mounting: "none" | "clip_enclosure" | "bolted_structural";
    panel_material: string;
    panel_thickness_mm: number;
    top_panel?: boolean;
    bolt_panels_to_gantry?: boolean;
    gantry_top_clearance_mm?: number;
    trident_backbone?: boolean;
    top_brace?: TopBraceType;
    front_brace?: FrontBraceType;
    extrusion_filling?: ExtrusionFillingType;
    corner_joint_type?: CornerJointType;
    feet_type?: FrameFeetType;
    chamber_temp_c?: number;
  };
  gantry: {
    x_beam_profile: string;
    x_beam_material: string;
    x_rail_profile: string;
    y_rail_profile: string;
    rail_material: string;
    x_rail_mount?: XRailMountType;
    y_rail_mount?: YRailMountType;
    dual_x_rails: boolean;
    rail_preload?: RailPreloadType;
    rail_quality?: RailQualityType;
    rail_block_type?: RailBlockType;
    rail_bolt_pitch_mm?: number;
    rail_backer_material?: RailBackerMaterialType;
    rail_backer_thickness_mm?: number;
    drive_mode: "2wd" | "awd";
    belt_routing?: BeltRoutingType;
    double_shear_pulleys: boolean;
    bracket_material?: BracketMaterialType;
    custom_bracket_override?: CustomBracketOverrideData | null;
    carriage_mass_g: number;
    belt: {
      type: BeltType;
      core: "fiberglass" | "steel" | "aramid";
      pulley_teeth: number;
      pitch_mm: number;
      pretension_n: number;
      wrap_angle_deg?: number;
      idler_shaft_diameter_mm?: number;
      back_idler_diameter_mm?: number | null;
      bearing_clearance_class?: BearingClearanceClass;
    };
  };
  kinematics: {
    build_volume_mm: [number, number, number];
    max_velocity_mm_s: number;
    max_accel_mm_s2: number;
    max_z_velocity_mm_s: number;
    max_z_accel_mm_s2: number;
    square_corner_velocity_mm_s: number;
    input_shaper:
      | "auto"
      | "none"
      | "zv"
      | "mzv"
      | "ei"
      | "2hump_ei"
      | "zvdd"
      | "smooth_zv"
      | "smooth_ei";
    drive_type?: DriveType;
    servo_vel_bw_hz?: number;
    servo_pos_bw_hz?: number;
    notch_filter_hz?: number | null;
    delta_arm_length_mm: number;
    delta_radius_mm: number;
    xy_stepper: {
      preset_id?: string;
      name: string;
      holding_torque_nm: number;
      rated_current_a: number;
      run_current_a?: number;
      phase_resistance_ohm?: number;
      inductance_mh: number;
      supply_voltage_v: number;
      rotor_inertia_kg_m2: number;
      motor_mass_g: number;
      steps_per_rev: number;
    };
  };
  toolhead: {
    preset_id?: string;
    name: string;
    total_mass_g: number;
    cog_offset_mm: [number, number, number];
    nozzle_offset_mm?: [number, number, number];
    mount_material?: ToolheadMountMaterial;
    body_construction?: ToolheadConstructionType | null;
    carriage_mount_material?: string | null;
    body_material?: string | null;
    hotend_mount_material?: string | null;
    carriage_plate_thickness_mm?: number;
    body_wall_thickness_mm?: number;
    hotend_mount_thickness_mm?: number;
    internal_stiffness_n_per_mm?: number;
    z_probe_type?: ZProbeType;
    envelope_mm: [number, number, number];
    hotend: {
      preset_id?: string | null;
      name: string;
      mass_g?: number;
      mount_style?: HotendMountStyle;
      structural_stiffness_n_per_mm?: number;
      overall_length_mm?: number;
      heater_power_w: number;
      block_material: string;
      block_mass_g: number;
      nozzle_material: string;
      nozzle_diameter_mm: number;
      melt_zone_length_mm: number;
      heatbreak_material: string;
      heatbreak_length_mm: number;
      heatbreak_wall_mm: number;
      heatbreak_id_mm: number;
      heatsink_thermal_resistance_k_w: number;
      part_cooling_cfm: number;
      max_temp_c: number;
      ambient_temp_c: number;
    };
    extruder: {
      type: "direct_drive" | "bowden";
      gear_ratio: number;
      max_push_force_n: number;
      filament_diameter_mm: number;
    };
  };
  bed: {
    size_mm: [number, number];
    plate_material: string;
    plate_thickness_mm: number;
    heater_power_w: number;
    mass_g: number;
    z_drive:
      | "leadscrew_triple"
      | "belted_quad_z"
      | "leadscrew_dual"
      | "leadscrew_dual_belt_sync"
      | "leadscrew_single";
    z_rail_profile?: ZRailProfileType;
    z_leadscrew_type?: ZLeadscrewType;
    target_temp_c?: number;
    mount_type?: BedMountType;
  };
}

export interface GeometryNode {
  id: string;
  position_mm: [number, number, number];
  fixed: boolean;
  lumped_mass_g: number;
  role: string;
}

export interface GeometryMember {
  id: string;
  start_id: string;
  end_id: string;
  start_mm: [number, number, number];
  end_mm: [number, number, number];
  profile: string;
  width_mm: number;
  height_mm: number;
  shape: string;
  material: string;
  color_hex: string;
  role: string;
  mass_g: number;
}

export interface ShellPanelGeometryData {
  id: string;
  bay: string;
  node_ids: [string, string, string, string];
  thickness_mm: number;
}

export interface ShellPanelStressData {
  id: string;
  bay: string;
  node_ids: [string, string, string, string];
  von_mises_mpa: number;
}

export interface DeepVerifySummary {
  fundamental_freq_hz: number;
  recommended_max_accel_mm_s2: number;
  combined_max_deflection_um: number;
  panel_drum_freq_hz: number;
  panel_max_von_mises_mpa: number;
  num_shell_elements: number;
  solve_time_ms: number;
}

export interface DeepVerifyResponse {
  fast_mesh_n: number;
  deep_mesh_n: number;
  fast_summary: DeepVerifySummary;
  deep_summary: DeepVerifySummary;
  convergence: {
    freq_diff_pct: number;
    deflection_diff_pct: number;
    accel_diff_pct: number;
    converged: boolean;
  };
  deep_simulation: SimulationResponse;
}

export interface ModeCauseItem {
  id: string;
  label: string;
  subsystem: SubsystemKey | "drive";
  kind: "stiffness" | "mass";
  share_pct: number;
  est_hz_gain: number;
  detail: string;
}

export interface ModeShapeData {
  mode_number: number;
  frequency_hz: number;
  dominant_axis: string;
  excited_axis?: string;
  mode_label?: string;
  what_is_shaking?: string;
  how_to_fix?: string;
  severity?: "critical" | "moderate" | "stiff";
  subsystem_share_pct?: {
    toolhead: number;
    gantry: number;
    frame: number;
    bed: number;
  };
  nozzle_whip_ratio?: number;
  toolhead_flop_pct?: number;
  relative_nozzle_bed_amp?: number;
  print_impact_score?: number;
  print_impact_level?: "critical" | "high" | "moderate" | "low";
  dominant_problem_area?: SubsystemKey;
  top_causes?: ModeCauseItem[];
  limiting_axis_share_pct?: number;
  x_axis_share_pct?: number;
  y_axis_share_pct?: number;
  is_limiting_axis_mode?: boolean;
  is_primary_limiting_mode?: boolean;
  toolhead_participation_xyz: [number, number, number];
  nodal_vectors: Record<string, [number, number, number]>;
  nodal_rotations?: Record<string, [number, number, number]>;
}

export interface AccelBottleneckShares {
  frame: number;
  belt: number;
  motor: number;
  bracket: number;
}

export interface BeltSubBreakdownData {
  k_belt_cord_n_per_mm: number;
  k_tooth_mesh_n_per_mm: number;
  k_idler_shaft_n_per_mm: number;
  k_belt_total_n_per_mm: number;
  belt_cord_share_pct: number;
  tooth_mesh_share_pct: number;
  idler_shaft_share_pct: number;
}

export interface OverallLimitingCauseItem {
  id: string;
  label: string;
  category: "belt" | "motor" | "bracket" | "structural" | "mass";
  subsystem: SubsystemKey | "drive";
  share_pct: number;
  est_hz_gain: number;
  est_accel_gain_mm_s2: number;
  detail: string;
}

export interface LimitingMovingMassInfo {
  axis: "X" | "Y";
  mass_g: number;
  label: string;
  est_hz_gain_25pct_lighter: number;
}

export interface LimitingModalSummaryData {
  limiting_axis: "X" | "Y";
  other_axis: "X" | "Y";
  effective_freq_hz: number;
  other_axis_effective_freq_hz: number;
  recommended_shaper: string;
  max_accel_mm_s2: number;
  other_axis_max_accel_mm_s2: number;
  limiting_kind: "structural" | "belt" | "motor" | "bracket";
  limiting_kind_label: string;
  headline: string;
  belt_included_note: string;
  structural_freq_hz: number;
  belt_freq_hz: number;
  motor_freq_hz: number;
  bracket_freq_hz: number;
  drive_freq_hz: number;
  nozzle_whip_amp_ratio?: number;
  primary_structural_mode_number: number;
  primary_structural_mode_label: string;
  primary_structural_mode_freq_hz: number;
  primary_structural_mode_share_pct: number;
  compliance_share_pct: AccelBottleneckShares;
  belt_sub_breakdown?: BeltSubBreakdownData;
  top_overall_causes: OverallLimitingCauseItem[];
  moving_mass_info?: LimitingMovingMassInfo;
}

export interface AccelBottleneckData {
  limiting_axis: "X" | "Y";
  primary_source: "frame" | "belt" | "motor" | "bracket";
  primary_label: string;
  primary_share_pct: number;
  shares_pct: AccelBottleneckShares;
  x_shares_pct: AccelBottleneckShares;
  y_shares_pct: AccelBottleneckShares;
  recommendation: string;
}

export interface ShaperData {
  axis: string;
  effective_freq_hz: number;
  structural_freq_hz: number;
  belt_freq_hz: number;
  motor_freq_hz?: number;
  bracket_freq_hz?: number;
  drive_freq_hz?: number;
  primary_bottleneck?: string;
  compliance_share_pct?: AccelBottleneckShares;
  recommended_shaper: string;
  max_accel_mm_s2: number;
  estimated_ringing_ratio: number;
  klipper_graph?: AxisShaperGraph;
}

export type ShaperRecommendationData = ShaperData;

export interface KlipperShaperCandidate {
  name: string; // "ZV" | "MZV" | "EI" | "2HUMP_EI" | "3HUMP_EI"
  freq_hz: number;
  vibrations_pct: number;
  smoothing: number;
  max_accel_mm_s2: number;
  score: number;
  attenuation_curve: number[];
}

export interface AxisShaperGraph {
  axis: "X" | "Y";
  freqs_hz: number[];
  raw_psd: number[];
  shaped_psd: number[];
  peak_freq_hz: number;
  damping_ratio: number;
  recommended_shaper: string;
  recommended_freq_hz: number;
  recommended_max_accel_mm_s2: number;
  candidates: KlipperShaperCandidate[];
}

export interface ParetoPointData {
  id: string;
  label: string;
  changes: string[];
  total_cost_usd: number;
  safe_max_accel_mm_s2: number;
  nozzle_deflection_um: number;
  fundamental_freq_hz: number;
  max_flow_mm3_s: number;
  total_mass_kg: number;
  composite_score: number;
  spec?: PrinterSpecData;
}

export interface ParetoOptResponse {
  backend: string;
  evaluations: number;
  population_size: number;
  generations: number;
  baseline: ParetoPointData;
  pareto_front: ParetoPointData[];
}

export interface ToolpathSegment {
  index: number;
  layer_index: number;
  start_mm: [number, number, number];
  end_mm: [number, number, number];
  is_extrusion: boolean;
  peak_speed_mm_s: number;
  volumetric_flow_mm3_s: number;
  ringing_severity: number;
  defect: "ok" | "travel" | "ringing" | "underextrusion" | "step_skip" | "cooling";
}

export interface GeometryExtrusionFillingData {
  id: string;
  name: string;
  short_name: string;
  bonded: boolean;
  added_mass_kg: number;
  damping_ratio: number;
  color_hex: string;
  notes: string;
}

export interface GeometryGantryBracketsData {
  material_id: string;
  name: string;
  short_name: string;
  youngs_modulus_gpa: number;
  hdt_c: number;
  creep_tension_limit_n: number;
  cnc_machined: boolean;
  kit_cost_usd: number;
  carriage_mass_delta_g: number;
  color_hex: string;
}

export interface GeometryLinearRailsData {
  x_rail_profile: string;
  y_rail_profile: string;
  rail_material: string;
  x_rail_mount: XRailMountType;
  y_rail_mount: YRailMountType;
  dual_x_rails: boolean;
  rail_quality?: RailQualityType;
  x_rail_width_mm: number;
  x_rail_height_mm: number;
  y_rail_width_mm: number;
  y_rail_height_mm: number;
  x_beam_ei_y_boost_pct: number;
  x_beam_ei_z_boost_pct: number;
  y_rail_ei_x_boost_pct: number;
  total_rail_mass_g: number;
}

export interface ZRailGeometryData {
  id: string;
  start_mm: [number, number, number];
  end_mm: [number, number, number];
  profile_id: string;
  carriage_z_mm: number;
  normal: [number, number, number];
}

export interface ZLeadscrewGeometryData {
  id: string;
  kind: "leadscrew" | "ballscrew" | "belt";
  mechanism_id: string;
  start_mm: [number, number, number];
  end_mm: [number, number, number];
  nut_z_mm: number;
  motor_pos_mm: [number, number, number];
  diameter_mm: number;
}

export interface ToolheadStageStiffnessesData {
  rail_carriage_n_per_mm?: number;
  backplate_only_n_per_mm?: number;
  carriage_mount_n_per_mm?: number;
  body_n_per_mm?: number;
  hotend_clamp_n_per_mm?: number;
  hotend_total_n_per_mm?: number;
  combined_toolhead_n_per_mm?: number;
}

export interface ToolheadStructureGeometryData {
  body_construction?: ToolheadConstructionType | null;
  construction_mass_delta_g?: number;
  carriage_mount_material: string;
  body_material: string;
  hotend_mount_material: string;
  carriage_plate_thickness_mm?: number;
  body_wall_thickness_mm?: number;
  hotend_mount_thickness_mm?: number;
  thickness_mass_delta_g?: number;
  hotend_mass_g?: number;
  toolhead_body_mass_g?: number;
  assembled_toolhead_mass_g?: number;
  hotend_preset_id: string | null;
  hotend_mount_style: string;
  hotend_stiffness_n_per_mm: number;
  hotend_overall_length_mm: number;
  stage_stiffnesses_n_per_mm?: ToolheadStageStiffnessesData;
}

export interface ToolheadFlexBreakdownData {
  rail_and_xbeam_pct: number;
  carriage_and_body_pct: number;
  hotend_and_nozzle_pct: number;
  effective_tip_stiffness_n_per_mm: number;
  hotend_stiffness_n_per_mm: number;
  carriage_mount_stiffness_n_per_mm: number;
  dominant_flex_source: "rail_xbeam" | "carriage_body" | "hotend_nozzle";
}

export interface GeometryMassSummaryData {
  frame_mass_kg: number;
  total_mass_kg: number;
  x_moving_mass_g: number;
  y_moving_mass_g: number;
  x_beam_mass_g?: number;
  x_rail_mass_g?: number;
  x_backer_mass_g?: number;
  y_carriage_joints_mass_g?: number;
  estimated_material_cost_usd: number;
}

export interface GeometryData {
  architecture: string;
  drive_mode?: "2wd" | "awd";
  double_shear_pulleys?: boolean;
  extrusion_filling?: GeometryExtrusionFillingData;
  gantry_brackets?: GeometryGantryBracketsData;
  linear_rails?: GeometryLinearRailsData;
  panels?: {
    enabled: boolean;
    mounting: string;
    material_id: string;
    material_name: string;
    thickness_mm: number;
    color_hex: string;
    top_panel?: boolean;
    bolted_to_gantry?: boolean;
    is_flying_gantry?: boolean;
    effective_shear_gt_kn_mm?: number;
    buckling_factor?: number;
    areal_mass_kg_m2?: number;
    total_panel_mass_kg?: number;
  };
  gantry_top_clearance_mm?: number;
  trident_backbone?: {
    enabled: boolean;
    active: boolean;
    material_name: string;
    thickness_mm: number;
    mass_kg: number;
    cost_usd: number;
    z_thrust_boost_factor: number;
    notes: string;
  };
  top_brace?: {
    id: string;
    active: boolean;
    label: string;
    material_name: string;
    thickness_mm: number;
    flange_width_mm: number;
    cutout_w_mm: number;
    cutout_d_mm: number;
    mass_kg: number;
    cost_usd: number;
  };
  front_brace?: {
    id: string;
    active: boolean;
    label: string;
    material_name: string;
    thickness_mm: number;
    flange_width_mm: number;
    header_height_mm: number;
    skirt_height_mm: number;
    cutout_w_mm: number;
    cutout_h_mm: number;
    mass_kg: number;
    cost_usd: number;
  };
  frame_dims_mm: [number, number, number];
  build_volume_mm: [number, number, number];
  build_origin_mm: [number, number, number];
  bed_size_mm: [number, number];
  bed_center_mm: [number, number, number];
  bed_node_id?: string;
  bed_frame_node_id?: string;
  bed_mount_type?: BedMountType;
  bed_z_drive?: string;
  feet_type?: FrameFeetType;
  bed_support_points_mm?: [number, number, number][];
  bed_mount_points_mm?: [number, number, number][];
  bed_subframe_z_mm?: number;
  bed_mount_anchor_nodes?: string[];
  z_rails?: ZRailGeometryData[];
  z_leadscrews?: ZLeadscrewGeometryData[];
  toolhead_structure?: ToolheadStructureGeometryData;
  toolhead_node_id: string;
  nozzle_node_id?: string;
  toolhead_cog_node_id?: string;
  cog_offset_mm?: [number, number, number];
  nozzle_offset_mm?: [number, number, number];
  mount_material?: string;
  toolhead_envelope_mm: [number, number, number];
  nodes: GeometryNode[];
  members: GeometryMember[];
  panel_mesh_n?: number;
  shell_panels?: ShellPanelGeometryData[];
  mass_summary: GeometryMassSummaryData;
}

export interface StaticFeaData {
  gravity_sag_toolhead_um: number;
  accel_deflection_x_um: number;
  accel_deflection_y_um: number;
  combined_max_deflection_um: number;
  nozzle_absolute_deflection_um?: number;
  bed_deflection_um?: number;
  beam_center_deflection_um?: number;
  nozzle_twist_amplification_pct?: number;
  nozzle_twist_angle_mrad?: number;
  toolhead_stiffness_x_n_per_mm: number;
  toolhead_stiffness_y_n_per_mm: number;
  toolhead_stiffness_z_n_per_mm: number;
  toolhead_flex_breakdown?: ToolheadFlexBreakdownData;
  max_von_mises_mpa: number;
  panel_max_von_mises_mpa?: number;
  peak_stress_part_label?: string;
  top_flex_explanation?: string;
  min_safety_factor: number;
  peak_inertial_force_x_n: number;
  peak_inertial_force_y_n: number;
  nodal_displacements_mm: Record<string, [number, number, number]>;
  nodal_displacements_x_mm?: Record<string, [number, number, number]>;
  nodal_displacements_y_mm?: Record<string, [number, number, number]>;
  nodal_displacements_xy_mm?: Record<string, [number, number, number]>;
  shell_panel_stresses?: ShellPanelStressData[];
  member_stresses: Array<{
    member_id: string;
    role: string;
    axial_force_n: number;
    max_bending_moment_nm: number;
    von_mises_mpa: number;
    yield_mpa: number;
    safety_factor: number;
    utilization: number;
  }>;
}

export interface ModalFeaData {
  fundamental_freq_hz: number;
  nozzle_pitch_twist_freq_hz?: number;
  panel_drum_freq_hz?: number;
  frame_damping_ratio?: number;
  recommended_max_accel_mm_s2: number;
  worst_print_mode_number?: number;
  accel_bottleneck?: AccelBottleneckData;
  limiting_modal_summary?: LimitingModalSummaryData;
  shaper_x: ShaperData | null;
  shaper_y: ShaperData | null;
  shaper_graphs?: { x: AxisShaperGraph; y: AxisShaperGraph };
  modes: ModeShapeData[];
}

export interface KinematicsData {
  architecture: string;
  drive_mode?: string;
  xy_motor_count?: number;
  build_volume_liters: number;
  frame_volume_liters: number;
  packaging_efficiency_pct: number;
  max_motor_rpm: number;
  available_torque_at_vmax_nm: number;
  required_torque_x_nm: number;
  required_torque_y_nm: number;
  torque_safety_margin: number;
  max_safe_accel_motor_mm_s2: number;
  step_skip_risk: boolean;
  belt_peak_tension_n: number;
  belt_slip_risk: boolean;
  belt_pluck_freq_150mm_hz?: number;
  belt_max_working_tension_n?: number;
  belt_overtension_risk?: boolean;
  shaft_bending_risk?: boolean;
  frame_belt_preload_n?: number;
  motor_run_current_a?: number;
  motor_stiffness_n_per_mm?: number;
  belt_stiffness_n_per_mm?: number;
  bracket_material?: string;
  bracket_stiffness_n_per_mm?: number;
  bracket_creep_risk?: boolean;
  custom_bracket_active?: boolean;
  custom_bracket_label?: string;
  drive_stiffness_n_per_mm?: number;
  motor_voltage_efficiency_pct?: number;
  motor_heat_w_each?: number;
  motor_overcurrent_risk?: boolean;
  tooth_mesh_stiffness_n_per_mm?: number;
  idler_shaft_stiffness_n_per_mm?: number;
  idler_shaft_deflection_um?: number;
  bearing_play_um?: number;
  bracket_modulus_retention_pct?: number;
  belt_routing?: BeltRoutingType;
  steel_core_fatigue_risk?: boolean;
  belt_bend_fatigue_risk?: boolean;
  belt_creep_risk?: boolean;
  belt_min_pulley_teeth?: number;
  belt_min_back_idler_mm?: number;
  belt_back_idler_mm?: number;
  belt_body_material?: string;
  belt_thermal_retention_pct?: number;
  belt_label?: string;
  belt_validation_tier?: BeltValidationTier;
  belt_theoretical_summary?: string;
  belt_practical_issues?: string[];
  pulley_circumference_mm?: number;
  vfa_belt_tooth_speed_range_mm_s?: [number, number];
  delta_workspace_valid: boolean;
}

export interface ThermalStructuralReport {
  chamber_temp_c: number;
  delta_t_k: number;
  gantry_bimetallic_bow_um: number;
  gantry_backer_reduction_pct: number;
  bed_target_temp_c: number;
  bed_mount_type: string;
  bed_gradient_bow_um: number;
  bed_buckling_taco_um: number;
  frame_z_drift_um: number;
  bracket_modulus_retention_pct: number;
  bracket_tg_warning: boolean;
  total_thermal_z_error_um: number;
  rail_newmark_gamma: number;
  rail_carriage_roll_nm_per_rad: number;
  rail_carriage_radial_n_per_um: number;
}

export interface WorkspaceFRFData {
  grid_n: number;
  u_coords: number[];
  v_coords: number[];
  freq_x_grid_hz: number[][];
  freq_y_grid_hz: number[][];
  min_axis_freq_grid_hz: number[][];
  xy_cross_coupling_grid_pct: number[][];
  center_freq_hz: number;
  min_freq_hz: number;
  max_freq_hz: number;
  worst_corner_drop_pct: number;
  max_cross_coupling_pct: number;
  reflected_rotor_mass_g: number;
  drive_type: string;
  midband_instability_risk: boolean;
}

export interface TimeDomainMetrologyData {
  corner_rounding_um: number;
  corner_overshoot_um: number;
  ringing_peak_to_peak_um: number;
  ringing_unshaped_um: number;
  ringing_reduction_pct: number;
  ringing_decay_mm: number;
  corner_trace: {
    ideal_x_mm: number[];
    ideal_y_mm: number[];
    actual_x_mm: number[];
    actual_y_mm: number[];
    unshaped_x_mm: number[];
    unshaped_y_mm: number[];
  };
  ballbar_radius_mm: number;
  ballbar_speed_mm_s: number;
  circularity_gxy_um: number;
  mean_radial_lag_um: number;
  quadrant_glitch_um: number;
  ballbar_angles_deg: number[];
  ballbar_radial_error_um: number[];
}

export interface RCSAData {
  sensor_peak_freq_hz: number;
  nozzle_tip_peak_freq_hz: number;
  nozzle_to_sensor_amp_ratio: number;
  hotend_bending_stiffness_n_per_mm: number;
  whip_risk: boolean;
  warning_message: string;
}

export interface SensitivityEntryData {
  hz_per_100g: number;
  hz_per_usd: number;
  stiffness_sens_hz: number;
  mass_sens_hz: number;
  compliance_share_pct: number;
  heat_score: number;
  heat_score_compliance?: number;
  heat_score_modal?: number;
  heat_score_mass?: number;
}

export interface LumpedMassSensitivityData {
  heat_score?: number;
  heat_score_compliance?: number;
  heat_score_modal?: number;
  heat_score_mass?: number;
  hz_per_100g: number;
  mass_sens_hz: number;
}

export interface SubsystemBreakdownEntry {
  id: string;
  label: string;
  compliance_share_pct: number;
  modal_stiff_hz: number;
  modal_mass_hz: number;
  severity: "critical" | "high" | "moderate" | "low";
}

export interface SensitivityStiffenTarget {
  id: string;
  role: string;
  label?: string;
  why_it_hurts?: string;
  est_hz_gain?: number;
  hz_per_usd: number;
  compliance_share_pct: number;
  action_hint: string;
}

export interface SensitivityLightenTarget {
  id: string;
  role: string;
  label?: string;
  why_it_hurts?: string;
  hz_gain_per_100g_saved?: number;
  hz_per_100g: number;
  mass_sens_hz: number;
  action_hint: string;
}

export interface SensitivityData {
  summary_diagnosis?: string;
  member_sensitivities: Record<string, SensitivityEntryData>;
  panel_bay_sensitivities: Record<string, SensitivityEntryData>;
  lumped_mass_sensitivities?: Record<string, LumpedMassSensitivityData>;
  subsystem_breakdown?: Record<SubsystemKey, SubsystemBreakdownEntry>;
  primary_problem_subsystem?: SubsystemKey;
  top_stiffen_targets: SensitivityStiffenTarget[];
  top_lighten_targets: SensitivityLightenTarget[];
}

export interface BedPlateFeaData {
  grid_nx: number;
  grid_ny: number;
  grid_x_rel: number[];
  grid_y_rel: number[];
  mount_points_rel: [number, number][];
  z_gravity_um: number[];
  z_thermal_um: number[];
  z_total_um: number[];
  mode1_shape_z: number[];
  gravity_sag_um: number;
  thermal_peak_to_valley_um: number;
  taco_buckling_um: number;
  total_flatness_error_um: number;
  probe_center_deflection_um: number;
  probe_corner_deflection_um: number;
  plate_first_mode_hz: number;
  plate_first_mode_type: "elastic_drum" | "mount_bounce" | "cantilever_flap";
  plate_modes_hz: number[];
  mesh_bed_leveling_required: boolean;
  first_layer_risk: "low" | "moderate" | "high" | "critical";
  summary: string;
}

export interface WeightShaperGainItem {
  id: string;
  label: string;
  current_mass_g: number;
  example_cut_g: number;
  hz_per_100g: number;
  est_hz_gain: number;
  est_accel_gain_mm_s2: number;
  affects_axis: "X" | "Y" | "XY";
  note: string;
}

export interface WeightShaperGainsSummary {
  limiting_axis: "X" | "Y";
  limiting_shaper_hz: number;
  limiting_max_accel_mm_s2: number;
  limiting_moving_mass_g: number;
  total_potential_cut_g: number;
  total_potential_hz_gain: number;
  total_potential_accel_gain_mm_s2: number;
  items: WeightShaperGainItem[];
}

export interface SimulationResponse {
  printer_name: string;
  architecture: string;
  spec: PrinterSpecData;
  yaml_text: string;
  scores: {
    structural: number;
    dynamics: number;
    thermal: number;
    print_quality: number;
    efficiency: number;
    composite: number;
  };
  diagnostics: string[];
  geometry: GeometryData;
  collision: {
    has_collision: boolean;
    clearance_x_mm: number;
    clearance_y_mm: number;
    clearance_z_mm: number;
    issues: string[];
  };
  static_fea: StaticFeaData;
  modal_fea: ModalFeaData;
  kinematics: KinematicsData;
  thermal_structural?: ThermalStructuralReport;
  bed_plate_fea?: BedPlateFeaData | null;
  workspace_frf?: WorkspaceFRFData;
  time_domain_metrology?: TimeDomainMetrologyData;
  rcsa?: RCSAData;
  sensitivity?: SensitivityData;
  weight_shaper_gains?: WeightShaperGainsSummary;
  thermal: {
    filament_id: string;
    filament_name: string;
    glass_transition_c: number;
    target_temp_c: number;
    heatup_time_215c_s: number;
    bed_heatup_time_100c_s: number;
    steady_heat_loss_w: number;
    heatbreak_thermal_resistance_k_w: number;
    cold_side_temp_c: number;
    heat_creep_risk: boolean;
    max_volumetric_flow_mm3_s: number;
    flow_limiting_factor: string;
    max_linear_speed_02x04_mm_s: number;
    extruder_pressure_margin: number;
    max_overhang_deg: number;
    bridge_sag_40mm_mm: number;
    heatup_curve: Array<{ time_s: number; temp_c: number }>;
    axial_profile: Array<{ z_mm: number; temp_c: number; zone: string }>;
  };
  virtual_print: {
    total_time_s: number;
    total_moves: number;
    extrusion_moves: number;
    layer_count: number;
    filament_used_mm: number;
    filament_used_g: number;
    avg_print_speed_mm_s: number;
    peak_print_speed_mm_s: number;
    peak_volumetric_flow_mm3_s: number;
    max_ringing_severity: number;
    underextrusion_segments: number;
    ringing_segments: number;
    step_skip_segments: number;
    cooling_warning_layers: number;
    out_of_bounds_issues: string[];
    print_quality_score: number;
    segments: ToolpathSegment[];
  };
}

export type ViewMode =
  | "assembly"
  | "fea_stress"
  | "modal"
  | "sensitivity"
  | "toolpath"
  | "solid_part"
  | "topology";

export type FeaLoadCase = "xy" | "x" | "y" | "gravity";

export type CompareMode = "last" | "baseline";

export type ModalColorMode = "relative" | "global";

export interface StudioStatePayload {
  client_id: string;
  active_preset_id: string | null;
  view_mode: ViewMode;
  selected_mode_idx: number;
  fea_load_case: FeaLoadCase;
  fea_scale: number;
  sensitivity_lens: string;
  modal_color_mode: string;
  toolhead_rel_pos: [number, number, number];
  filament_id: string;
  inspected_part_id: string | null;
  spec: PrinterSpecData | null;
}

export interface StudioCaptureSnapshotCommand {
  id: string;
  type: "capture_snapshot";
  frames?: number;
  delay_ms?: number;
}

export interface StudioUpdateStateCommand {
  id: string;
  type: "update_state";
  spec?: PrinterSpecData | null;
  view_mode?: ViewMode | null;
  selected_mode_idx?: number | null;
  fea_load_case?: FeaLoadCase | null;
  fea_scale?: number | null;
  toolhead_rel_pos?: [number, number, number] | null;
  filament_id?: string | null;
  sensitivity_lens?: AdjointLens | null;
  modal_color_mode?: ModalColorMode | null;
}

export type StudioCommand = StudioCaptureSnapshotCommand | StudioUpdateStateCommand;

export interface StudioCommandsResponse {
  commands: StudioCommand[];
}

export interface StudioSnapshotPayload {
  command_id: string;
  client_id: string;
  view_mode: ViewMode;
  selected_mode_idx: number;
  frames_base64: string[];
}

export interface HistorySnapshot {
  sim: SimulationResponse;
  bom: PrinterBOMResponse | null;
  spec: PrinterSpecData;
  filamentId: string;
  label: string | null;
}

export interface AdxlAxisCalibrationData {
  axis: "X" | "Y";
  source_format: "klipper_psd_csv" | "raw_accel_csv";
  freqs_hz: number[];
  measured_psd_norm: number[];
  simulated_psd_norm: number[];
  measured_peak_hz: number;
  simulated_peak_hz: number;
  peak_delta_hz: number;
  peak_error_pct: number;
  measured_damping_ratio: number;
  simulated_damping_ratio: number;
  measured_shaper: string;
  measured_shaper_freq_hz: number;
  measured_max_accel_mm_s2: number;
  simulated_shaper: string;
  simulated_shaper_freq_hz: number;
  simulated_max_accel_mm_s2: number;
  spectral_correlation: number;
  sample_count: number;
  diagnostics: string[];
}

export interface AdxlCalibrationResponse {
  printer_name: string;
  axes: Partial<Record<"X" | "Y", AdxlAxisCalibrationData>>;
  overall_summary: string;
  recommended_adjustments: string[];
}

export interface AdxlFitChange {
  field: string;
  label: string;
  before: string | number | boolean | null;
  after: string | number | boolean | null;
  reason: string;
}

export interface AdxlFitResponse {
  printer_name: string;
  before_mean_abs_error_pct: number;
  after_mean_abs_error_pct: number;
  before_axes: Partial<Record<"X" | "Y", AdxlAxisCalibrationData>>;
  after_axes: Partial<Record<"X" | "Y", AdxlAxisCalibrationData>>;
  fitted_spec: PrinterSpecData;
  changes: AdxlFitChange[];
  summary: string;
}
