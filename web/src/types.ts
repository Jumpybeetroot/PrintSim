export interface MaterialItem {
  id: string;
  name: string;
  youngs_modulus_pa: number;
  density_kg_m3: number;
  yield_strength_pa: number;
  color_hex: string;
  is_slm_metal?: boolean;
  e3_ratio?: number;
  z_strength_ratio?: number;
  bolted_joint_stiffness_factor?: number;
  surface_roughness_ra_um?: number | null;
  porosity_pct?: number;
}

export interface ProfileItem {
  id: string;
  name: string;
  shape: string;
  width_m: number;
  height_m: number;
  model_status?: ModelStatus;
  model_status_label?: string;
  wip_notes?: string;
}

export type ModelStatus = "complete" | "wip_specific" | "generic_standin";

export interface PresetItem {
  id: string;
  name: string;
  short_name?: string;
  architecture: string;
  base_kit?: "siboor" | "ldo" | "diy" | "generic" | string;
  category?: ModelStatus | string;
  category_label?: string;
  model_status?: ModelStatus;
  model_status_label?: string;
  wip_notes?: string;
  hardware_kit?: string;
  summary?: string;
  highlights?: string[];
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
  nema_size?: number;
  body_length_mm?: number;
  insulation_class?: "B" | "F" | "H";
  has_encoder?: boolean;
  unit_cost_usd?: number;
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
  | "pc_pbt_gf_printed"
  | "ppa_cf_printed"
  | "pps_cf_printed"
  | "slm_aluminum"
  | "slm_ti6al4v"
  | "slm_maraging_steel"
  | "slm_316l_steel"
  | "slm_17_4ph_steel"
  | "sheet_metal_steel"
  | "aluminum_6061_t6"
  | "aluminum_7075_t6";

export type GantryHardwareKitType =
  | "auto"
  | "voron_stock_printed"
  | "ldo_voron_kit"
  | "siboor_cnc_awd"
  | "siboor_cnc_2wd"
  | "monolith_ldo_cnc_awd"
  | "monolith_ldo_cnc_2wd"
  | "vzbot_330_awd"
  | "vzbot_330_2wd"
  | "procedural";

export interface GantryHardwareKitSettings {
  drive_mode?: "2wd" | "awd";
  belt_routing?: BeltRoutingType;
  bracket_material?: BracketMaterialType;
  double_shear_pulleys?: boolean;
  x_rail_profile?: string;
  y_rail_profile?: string;
  x_rail_mount?: XRailMountType;
  y_rail_mount?: YRailMountType;
  rail_block_type?: RailBlockType;
  belt_type?: BeltType;
  belt_pretension_n?: number;
  carriage_mass_g?: number;
  idler_axial_preload?: IdlerAxialPreloadType;
  z_joint_material_flying?: ZJointMaterialType;
  sibling_2wd_kit?: GantryHardwareKitType;
  sibling_awd_kit?: GantryHardwareKitType;
}

export interface GantryHardwareKitItem {
  id: GantryHardwareKitType | string;
  name: string;
  short_name: string;
  description?: string;
  default_bracket_material?: string;
  drive_mode?: string;
  kit_settings?: GantryHardwareKitSettings | null;
  model_status?: ModelStatus;
  model_status_label?: string;
  wip_notes?: string;
}

export interface CadGantryStationAsset {
  asset_id: string;
  visual_glb_url: string;
  mass_g?: number;
  stiffness_n_per_mm?: number | null;
  port0_mm?: [number, number, number];
  port1_mm?: [number, number, number];
}

export type ChamberHeaterType = "none" | "monolith_100w_ptc" | "monolith_500w_ptc";

export type CpapBlowerType = "none" | "ws7040_24v" | "wm9290_24v" | "wm7575_220vac";

export type StepperThermalGasketType = "none" | "ptfe_2mm";

export type IdlerAxialPreloadType = "auto" | "none" | "disc_spring_m5";

export type ZJointType = "single" | "double";

export type ZJointMaterialType = "printed" | "hybrid" | "slm" | "milled";

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

export type CFRailMountingType = "auto" | "single_wall_bare" | "two_wall_clamped_inserts";

export type BearingClearanceClass = "C2" | "CN" | "C3";

export type BedMountType =
  | "kinematic_3pt"
  | "cnc_kelvin_magnetic"
  | "captive_belleville_kinematic"
  | "dual_leaf_ti_kinematic"
  | "spherical_3pt_ratrig"
  | "craz_kinematic_ge5c"
  | "trident_3pt_standoffs"
  | "whopping_pochard_kinematic"
  | "rigid_4pt"
  | "floating_4pt"
  | "silicone_4pt"
  | "silicone_4pt_clamped"
  | "springs_4pt";

export type FrameFeetType =
  | "rigid_bolted"
  | "rubber_feet"
  | "anti_vib_pads"
  | "paver_on_foam"
  | "suspended_lines";

export type SuspensionLineType =
  | "paracord_550"
  | "uhmwpe_braid_200lb"
  | "nylon_mono_100lb"
  | "steel_cable_1_5mm"
  | "bungee_cord_6mm";

export type FrameBallastLocationType = "base" | "top" | "distributed";

export interface SuspensionLineCatalogItem {
  id: SuspensionLineType | string;
  name: string;
  short_name: string;
  stiffness_ea_n: number;
  break_strength_n: number;
  design_factor: number;
  loss_factor: number;
  notes?: string;
}

export interface MountRigidBodyMode {
  frequency_hz: number;
  kind: string;
  label: string;
  mount_strain_share_pct: number;
  in_shaper_band: boolean;
  adxl_peak_ratio_x: number;
  adxl_peak_ratio_y: number;
}

export interface MountAxisDynamics {
  moving_mass_kg: number;
  reduced_moving_mass_kg: number;
  frame_recoil_ratio: number;
  frame_recoil_um_per_mm: number;
  recoil_freq_gain_pct: number;
  retained_mount_compliance_um_per_n: number;
}

export interface MountingDynamics {
  feet_type: FrameFeetType;
  mount_damping_ratio: number;
  total_mass_kg: number;
  ballast_mass_kg: number;
  ballast_location: FrameBallastLocationType;
  rigid_body_modes: MountRigidBodyMode[];
  lowest_sway_hz: number | null;
  x: MountAxisDynamics;
  y: MountAxisDynamics;
  notes: string[];
  suspension?: {
    line: string;
    line_length_mm: number;
    line_count: number;
    line_tension_n: number;
    line_break_strength_n: number;
    safety_factor: number;
    design_factor: number;
    static_stretch_mm: number;
    pendulum_hz_analytic: number;
    swing_resonant_line_length_mm: number;
  };
}

export interface BeltDynamics {
  installed_pretension_n: number;
  operating_pretension_n: number;
  thermal_tension_delta_n: number;
  pretension_set_temp_c: number | null;
  pluck_hz_150mm_installed: number;
  pluck_hz_150mm_operating: number;
  tension_mismatch_pct: number;
  coupling_kappa: number;
  tension_a_n?: number;
  tension_b_n?: number;
  pluck_hz_150mm_a?: number;
  pluck_hz_150mm_b?: number;
  effective_coupling?: number;
  coupled_x_hz?: number;
  coupled_y_hz?: number;
  cross_peak_in_x?: number;
  cross_peak_in_y?: number;
}

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
  | "gt2_10mm"
  | "gt3_10mm"
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
  | "tr8x4_wobblex"
  | "tr8x8_pom"
  | "tr8x8_brass"
  | "sfu1204_bk10"
  | "sfu1204_wobblex"
  | "sfu1605_fk12"
  | "gt2_6mm_z_belt"
  | "gt2_9mm_z_belt"
  | "gt2_9mm_planetary_z"
  | "gt2_9mm_worm_z"
  | "kevinakasam_belted_z";

export type HotendMountStyle =
  | "groove_mount"
  | "rigid_4bolt_m3"
  | "dragon_4bolt_m25"
  | "chube_6bolt_m3_18pcd";

export type AdjointLens = "print_impact" | "compliance" | "modal" | "mass";

export type SubsystemKey = "toolhead" | "gantry" | "frame" | "bed";

export type DriveType = "open_loop_stepper" | "closed_loop_stepper" | "foc_servo";

export type InputShaperType =
  | "auto"
  | "none"
  | "zv"
  | "mzv"
  | "ei"
  | "2hump_ei"
  | "3hump_ei"
  | "zvdd"
  | "smooth_zv"
  | "smooth_ei"
  | "smooth_flatness";

export type AwdSyncMode = "none" | "torque_mirror" | "position_equalizer";

export interface BedMountCatalogItem {
  id: BedMountType;
  name: string;
  short_name: string;
  mount_points?: number;
  k_axial_n_per_mm?: number;
  k_lateral_n_per_mm?: number;
  k_restraint_thermal_n_per_m?: number;
  damping_delta?: number;
  standoff_height_mm?: number;
  cost_usd?: number;
  notes?: string;
  model_status?: ModelStatus;
  model_status_label?: string;
  wip_notes?: string;
}

export interface FrameFeetCatalogItem {
  id: FrameFeetType;
  name: string;
  short_name: string;
  k_axial_n_per_mm?: number;
  k_lateral_n_per_mm?: number;
  damping_delta?: number;
  base_mass_kg?: number;
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
  manufacturing_process?: "fdm" | "slm" | "sheet_metal" | "cnc";
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
  compliance_factor?: number;
  cog_y_shift_mm?: number;
  notes?: string;
  model_status?: ModelStatus;
  model_status_label?: string;
  wip_notes?: string;
}

export interface BeltRoutingCatalogItem {
  id: BeltRoutingType;
  name: string;
  short_name: string;
  span_factor?: number;
  idler_stiffness_factor?: number;
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
  body_material?: string;
  body_construction?: ToolheadConstructionType | null;
  internal_stiffness_n_per_mm?: number;
  z_probe_type?: ZProbeType;
  extruder_type?: "direct_drive" | "bowden";
  extruder_gear_ratio?: number;
  default_hotend_preset_id?: string;
  part_cooling_cfm?: number;
  envelope_mm?: [number, number, number];
  cpap_blower_id?: CpapBlowerType;
  cpap_hose_id_mm?: number;
  shared_heatsink_cpap_duct?: boolean;
  belt_plane_z_offset_mm?: number;
  offset_datum_rail_mount?: "front" | "top";
  default_hardware_summary?: string;
  model_status?: ModelStatus;
  model_status_label?: string;
  wip_notes?: string;
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
  model_status?: ModelStatus;
  model_status_label?: string;
  wip_notes?: string;
}

export interface ZDriveMechanismItem {
  id: string;
  name: string;
  kind: "leadscrew" | "ballscrew" | "belt";
  lead_mm?: number;
  diameter_mm?: number;
  k_nut_n_per_um?: number;
  k_thrust_n_per_um?: number;
  k_drive_n_per_um_per_corner?: number | null;
  backlash_um?: number;
  wobble_decoupled?: boolean;
  self_locking?: boolean;
  unpowered_drop_risk?: boolean;
  gear_ratio?: number;
  cost_per_axis_usd?: number;
  notes?: string;
  model_status?: ModelStatus;
  model_status_label?: string;
  wip_notes?: string;
}

export interface ZRailProfileItem {
  id: string;
  name: string;
  mass_kg_per_m?: number;
  model_status?: ModelStatus;
  model_status_label?: string;
  wip_notes?: string;
}

export interface CornerJointCatalogItem {
  id: CornerJointType;
  name?: string;
  k_t_n_per_mm: number;
  k_r_inplane_nm_rad: number;
  k_r_torsion_nm_rad: number;
  damping_delta: number;
}

export interface RailPreloadCatalogItem {
  id: RailPreloadType;
  name?: string;
  preload_fraction: number;
}

export interface BearingClearanceCatalogItem {
  id: BearingClearanceClass;
  name?: string;
  radial_play_um: number;
}

export interface HotendMountStyleCatalogItem {
  id: HotendMountStyle;
  name?: string;
  clamp_stiffness_base_n_per_mm: number;
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
  gantry_hardware_kits?: GantryHardwareKitItem[];
  gantry_weight_classes?: GantryWeightClassItem[];
  x_rail_mounts?: RailMountOption<XRailMountType>[];
  y_rail_mounts?: RailMountOption<YRailMountType>[];
  bed_mounts?: BedMountCatalogItem[];
  frame_feet?: FrameFeetCatalogItem[];
  suspension_lines?: SuspensionLineCatalogItem[];
  z_probes?: ZProbeCatalogItem[];
  belts?: BeltCatalogItem[];
  belt_routings?: BeltRoutingCatalogItem[];
  hotend_presets?: HotendPresetItem[];
  z_drive_mechanisms?: ZDriveMechanismItem[];
  z_rail_profiles?: ZRailProfileItem[];
  frame_bracing?: Array<{
    id: string;
    label: string;
    short_label: string;
    kind: string;
    cad_asset_family: string | null;
    physics_model: string;
    description: string;
    cost_usd: number;
  }>;
  top_braces?: BraceCatalogItem[];
  front_braces?: BraceCatalogItem[];
  corner_joints?: CornerJointCatalogItem[];
  rail_preloads?: RailPreloadCatalogItem[];
  bearing_clearances?: BearingClearanceCatalogItem[];
  hotend_mount_styles?: HotendMountStyleCatalogItem[];
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
  realized_wall_thickness_mm?: number;
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
  realized_wall_thickness_mm?: number;
  surface_mesh: TetSurfaceMeshData;
}

export interface BracketDesignResponse {
  bracket_type: string;
  params_mm: {
    arm_length_mm: number;
    height_mm: number;
    wall_thickness_mm: number;
    rib_thickness_mm: number;
    bolt_hole_mm: number;
    fillet_mm: number;
  };
  realized_wall_thickness_mm?: number;
  material_id: string;
  engine: string;
  mass_g: number;
  volume_cm3: number;
  max_displacement_um: number;
  stiffness_n_per_mm: number;
  specific_stiffness_n_per_mm_per_g: number;
  max_von_mises_mpa: number;
  safety_factor: number;
  failure_mode: string;
  fdm_anisotropic: boolean;
  num_elements: number;
  num_dofs: number;
  exported_stl_path?: string | null;
  exported_step_path?: string | null;
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

export interface ModelOverridesData {
  corner_joint?: {
    k_t_n_per_mm?: number | null;
    k_r_inplane_nm_rad?: number | null;
    k_r_torsion_nm_rad?: number | null;
    damping_delta?: number | null;
  } | null;
  extrusion_filling?: {
    density_kg_m3?: number | null;
    youngs_modulus_gpa?: number | null;
    bonded?: boolean | null;
    damping_boost?: number | null;
  } | null;
  frame_feet?: {
    k_axial_n_per_mm?: number | null;
    k_lateral_n_per_mm?: number | null;
    damping_delta?: number | null;
    base_mass_kg?: number | null;
  } | null;
  belt?: {
    stiffness_ea_n?: number | null;
    tooth_shear_k_n_per_m_mm?: number | null;
    linear_mass_g_per_m?: number | null;
    max_working_tension_n?: number | null;
  } | null;
  gantry_bracket?: {
    stiffness_n_per_mm?: number | null;
    carriage_mass_delta_g?: number | null;
    creep_tension_limit_n?: number | null;
    damping_delta?: number | null;
  } | null;
  rail_preload?: {
    preload_fraction?: number | null;
  } | null;
  rail_quality?: {
    stiffness_mult?: number | null;
    drag_mult?: number | null;
  } | null;
  belt_routing?: {
    span_factor?: number | null;
    idler_stiffness_factor?: number | null;
  } | null;
  bearing_clearance?: {
    radial_play_um?: number | null;
  } | null;
  z_probe?: {
    compliance_factor?: number | null;
    cog_y_shift_mm?: number | null;
  } | null;
  toolhead_construction?: {
    body_stiffness_base_n_per_mm?: number | null;
    torsion_mult?: number | null;
  } | null;
  hotend_clamp?: {
    clamp_stiffness_base_n_per_mm?: number | null;
  } | null;
  bed_mount?: {
    k_axial_n_per_mm?: number | null;
    k_lateral_n_per_mm?: number | null;
    k_restraint_thermal_n_per_mm?: number | null;
    damping_delta?: number | null;
    standoff_height_mm?: number | null;
  } | null;
  z_drive_mechanism?: {
    k_nut_n_per_um?: number | null;
    k_thrust_n_per_um?: number | null;
    k_drive_n_per_um_per_corner?: number | null;
    backlash_um?: number | null;
  } | null;
}

export type ModelOverrideGroupKey = keyof ModelOverridesData;

export interface EffectiveParamGroup {
  baseline: Record<string, number | boolean>;
  effective: Record<string, number | boolean>;
  overridden_fields: string[];
}

export type EffectiveModelParamsData = Partial<Record<ModelOverrideGroupKey, EffectiveParamGroup>>;

export type ConfidenceTier = "exact" | "calibrated" | "estimate" | "heuristic";

export interface ModelConfidenceOptionItem {
  id: string;
  label: string;
  trust_pct: number;
  tier: ConfidenceTier;
  active: boolean;
  summary_values: string;
  basis: string;
  caveat: string;
}

export interface ModelConfidenceEntry {
  id: string;
  subsystem: "frame" | "gantry" | "toolhead" | "hotend" | "bed" | "solver";
  subsystem_label: string;
  name: string;
  active_selection: string;
  active_values: string;
  trust_pct: number;
  tier: ConfidenceTier;
  override_group?: ModelOverrideGroupKey | null;
  overridden: boolean;
  basis: string;
  caveat: string;
  options_breakdown?: ModelConfidenceOptionItem[];
  cad_superelement_active?: boolean;
  cad_superelement_id?: string | null;
  cad_trust_pct?: number | null;
}

export interface ModelConfidenceReport {
  disclaimer_title: string;
  disclaimer_body: string;
  overall_trust_pct: number;
  entries: ModelConfidenceEntry[];
}

export type BracingType =
  | "none"
  | "cast_corner_brackets"
  | "cnc_corner_gussets"
  | "rear_flat_bar_x"
  | "side_flat_bar_x"
  | "rear_and_side_flat_bars"
  | "gussets_and_flat_bars";

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
    bracing: BracingType;
    brace_profile: string;
    brace_material: string;
    enclosed_panels: boolean;
    panel_mounting: "none" | "clip_enclosure" | "bolted_structural";
    panel_material: string;
    panel_thickness_mm: number;
    top_panel?: boolean;
    bolt_panels_to_gantry?: boolean;
    gantry_top_clearance_mm?: number;
    gantry_on_top_frame?: boolean;
    side_panel_material?: string | null;
    panel_overrides?: Partial<
      Record<
        "rear" | "left" | "right" | "top",
        { material?: string | null; thickness_mm?: number | null }
      >
    >;
    trident_backbone?: boolean;
    top_brace?: TopBraceType;
    front_brace?: FrontBraceType;
    extrusion_filling?: ExtrusionFillingType;
    corner_joint_type?: CornerJointType;
    feet_type?: FrameFeetType;
    suspension_line?: SuspensionLineType;
    suspension_length_mm?: number;
    ballast_mass_kg?: number;
    ballast_location?: FrameBallastLocationType;
    chamber_temp_c?: number;
    chamber_heater_id?: ChamberHeaterType;
    chamber_insulation_mm?: number;
    rear_bottom_extrusion?: boolean;
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
    cf_rail_mounting?: CFRailMountingType;
    drive_mode: "2wd" | "awd";
    belt_routing?: BeltRoutingType;
    double_shear_pulleys: boolean;
    bracket_material?: BracketMaterialType;
    hardware_kit?: GantryHardwareKitType;
    custom_bracket_override?: CustomBracketOverrideData | null;
    carriage_mass_g: number;
    stepper_thermal_gasket?: StepperThermalGasketType;
    idler_axial_preload?: IdlerAxialPreloadType;
    z_joint_type?: ZJointType;
    z_joint_material?: ZJointMaterialType;
    belt: {
      type: BeltType;
      core: "fiberglass" | "steel" | "aramid";
      pulley_teeth: number;
      pitch_mm: number;
      pretension_n: number;
      pretension_set_temp_c?: number | null;
      tension_mismatch_pct?: number;
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
    input_shaper: InputShaperType;
    drive_type?: DriveType;
    servo_vel_bw_hz?: number;
    servo_pos_bw_hz?: number;
    notch_filter_hz?: number | null;
    lpv_workspace_scheduling?: boolean;
    awd_sync_mode?: AwdSyncMode;
    awd_equalizer_bw_hz?: number;
    notch_depth_db?: number;
    notch_q?: number;
    pin_rotor_ff?: boolean;
    toolhead_imu_active_damping?: boolean;
    mode_space_torque_ff?: boolean;
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
      nema_size?: number;
      body_length_mm?: number;
      insulation_class?: "B" | "F" | "H";
      has_encoder?: boolean;
    };
  };
  toolhead: {
    preset_id?: string;
    name: string;
    total_mass_g: number;
    cog_offset_mm: [number, number, number];
    nozzle_offset_mm?: [number, number, number];
    offset_datum_rail_mount?: "front" | "top";
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
    use_hotend_bottom_brace?: boolean;
    nozzle_shield_mode?: "auto" | "stock" | "microwaved" | "none";
    cpap_blower_id?: CpapBlowerType;
    cpap_hose_id_mm?: number;
    cpap_pwm_pct?: number;
    shared_heatsink_cpap_duct?: boolean;
    belt_plane_z_offset_mm?: number;
    umbilical_type?: "standard" | "can_umbilical" | "cable_chain" | "cpap_15mm" | "cpap_18mm" | "custom";
    umbilical_mass_g?: number | null;
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
  model_overrides?: ModelOverridesData | null;
  use_cad_superelements?: boolean;
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
  bed_flatness_error_um?: number;
  bed_first_mode_hz?: number;
  solve_time_ms: number;
}

export interface SignoffCheckItem {
  id: string;
  title: string;
  status: "pass" | "caution" | "fail";
  value: string;
  detail: string;
}

export interface SignoffConfidenceBand {
  nominal: number;
  p05: number;
  p95: number;
  ci90_half_width: number;
  ci90_pct: number;
}

export interface SignoffUncertaintyDriver {
  group_id: string;
  label: string;
  trust_pct: number;
  tier: string;
  overridden: boolean;
  sigma_param_pct: number;
  delta_freq_hz: number;
  delta_accel_mm_s2: number;
  delta_deflection_um: number;
  variance_share_pct: number;
  /** Share of nozzle-bed flex variance (variance_share_pct is the resonance share). */
  deflection_variance_share_pct?: number;
  how_to_tighten: string;
}

export interface SignoffWorkspaceFea {
  grid_n: number;
  coords: number[];
  /** False for architectures whose model ignores XY (delta): the grid repeats the centre solve. */
  xy_position_modeled?: boolean;
  num_fea_solves: number;
  solve_time_ms: number;
  freq_x_grid_hz: number[][];
  freq_y_grid_hz: number[][];
  min_axis_grid_hz: number[][];
  safe_accel_grid_mm_s2: number[][];
  deflection_grid_um: number[][];
  z_sweep: Array<{
    z_rel: number;
    shaper_x_hz: number;
    shaper_y_hz: number;
    min_axis_hz: number;
    safe_accel_mm_s2: number;
    deflection_um: number;
  }>;
  /** Softest XY cell re-solved at the Z extremes (empty when that cell is the centre). */
  worst_xy_z_sweep?: Array<{
    u_rel: number;
    v_rel: number;
    z_rel: number;
    shaper_x_hz: number;
    shaper_y_hz: number;
    min_axis_hz: number;
    safe_accel_mm_s2: number;
    deflection_um: number;
  }>;
  center_min_axis_hz: number;
  center_safe_accel_mm_s2: number;
  worst_case_min_axis_hz: number;
  worst_case_safe_accel_mm_s2: number;
  worst_case_deflection_um: number;
  worst_freq_pos: [number, number, number];
  worst_accel_pos: [number, number, number];
  worst_deflection_pos: [number, number, number];
  worst_freq_drop_pct: number;
  worst_accel_drop_pct: number;
  spatial_spread_pct: number;
}

export interface SignoffUncertaintyReport {
  overall_trust_pct: number;
  num_groups_perturbed: number;
  /** Labels of uncertain groups a max_uncertainty_groups cap left unperturbed. */
  skipped_groups?: string[];
  /** What the parametric band does and does not propagate. */
  coverage_note?: string;
  num_fea_solves: number;
  solve_time_ms: number;
  bands: {
    shaper_x_hz: SignoffConfidenceBand;
    shaper_y_hz: SignoffConfidenceBand;
    min_axis_hz: SignoffConfidenceBand;
    fundamental_freq_hz: SignoffConfidenceBand;
    recommended_max_accel_mm_s2: SignoffConfidenceBand;
    combined_max_deflection_um: SignoffConfidenceBand;
  };
  drivers: SignoffUncertaintyDriver[];
}

export interface DeepVerifyResponse {
  fast_mesh_n: number;
  deep_mesh_n: number;
  fast_bed_mesh_n?: number;
  deep_bed_mesh_n?: number;
  fast_summary: DeepVerifySummary;
  deep_summary: DeepVerifySummary;
  convergence: {
    freq_diff_pct: number;
    deflection_diff_pct: number;
    accel_diff_pct: number;
    converged: boolean;
  };
  deep_simulation: SimulationResponse;
  // Sign-Off Verification superset fields (present from /api/verify/signoff)
  verdict?: "pass" | "caution" | "fail";
  checks?: SignoffCheckItem[];
  total_solve_time_ms?: number;
  total_fea_solves?: number;
  workspace_fea?: SignoffWorkspaceFea;
  uncertainty?: SignoffUncertaintyReport;
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

export interface ToolheadPartModalData {
  id: string;
  name: string;
  stage: string;
  is_structural: boolean;
  source: "cad_3d_recovery" | "stage_fea";
  modal_flex_pct: number;
  kinetic_share_pct: number;
  rel_flex_norm: number;
  disp_norm: number;
}

export interface ToolheadPartStaticData {
  id: string;
  name: string;
  stage: string;
  is_structural: boolean;
  source: "cad_3d_recovery" | "stage_fea";
  material: string;
  mass_g: number;
  stiffness_n_per_mm: number | null;
  compliance_share_pct: number;
  toolhead_share_pct: number;
  local_flex_um: number;
  total_disp_um: number;
  twist_mrad: number;
  stress_mpa: number | null;
  yield_mpa: number | null;
  why_it_hurts: string;
  action_hint: string;
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
  toolhead_part_flex?: ToolheadPartModalData[];
  toolhead_mode_fields?: Record<string, CadIntraPartFieldData>;
  mode_part_fields?: Record<string, IntraPartField>;
}

export type ModeShape = ModeShapeData;

export interface IntraPartSubpartInfo {
  name: string;
  label?: string;
  strain_energy_pct?: number;
  max_stress_mpa?: number;
  max_flex_um?: number;
  mean_stress_mpa?: number;
  point_count?: number;
}

export interface IntraPartField {
  source: string;
  asset_id?: string;
  label?: string;
  node_ids?: string[];
  points_mm: [number, number, number][];
  flex_um?: number[];
  stress_mpa?: number[];
  max_flex_um?: number;
  max_stress_mpa?: number;
  flex_norm?: number[];
  strain_norm?: number[];
  subpart_idx?: number[];
  subparts?: Array<string | IntraPartSubpartInfo>;
}

export type CadIntraPartFieldData = IntraPartField;

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

export interface StepResponseData {
  time_ms: number[];
  unshaped: number[];
  shaped: number[];
  overshoot_unshaped_pct: number;
  overshoot_shaped_pct: number;
  settle_time_unshaped_ms: number;
  settle_time_shaped_ms: number;
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
  step_response?: StepResponseData;
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
  corner_smoothing_um?: number;
  defect:
    | "ok"
    | "travel"
    | "ringing"
    | "smoothing"
    | "underextrusion"
    | "step_skip"
    | "belt_slip"
    | "cooling";
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
  hardware_kit?: string;
  hardware_kit_requested?: string;
  hardware_kit_label?: string;
  cad_backed?: boolean;
  cad_xy_fea_active?: boolean;
  cad_stiffness_n_per_mm?: number | null;
  cad_assets?: Record<string, CadGantryStationAsset>;
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
  x_beam_span_mm?: number | null;
  x_beam_ei_y_boost_pct: number;
  x_beam_ei_z_boost_pct: number;
  x_beam_gj_boost_pct?: number;
  x_rail_newmark_gamma?: number;
  cf_rail_mounting?: string;
  x_beam_insert_mass_g?: number;
  y_rail_ei_x_boost_pct: number;
  total_rail_mass_g: number;
  x_carriage_block_mass_g?: number;
  x_carriage_block_delta_g?: number;
  x_carriage_k_rad_n_per_um?: number;
  x_carriage_k_rx_nm_rad?: number;
  x_carriage_k_ry_nm_rad?: number;
  y_carriage_block_mass_g?: number;
  y_carriage_block_delta_g?: number;
  y_carriage_k_rx_nm_rad?: number;
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

export interface CadSuperelementSummary {
  id: string;
  category: string;
  asset_id: string;
  name: string;
  element_type: string;
  num_elements: number;
  port_stiffness_n_per_mm: { kx: number; ky: number; kz: number };
  port0_mm?: [number, number, number] | null;
  port1_mm?: [number, number, number] | null;
  visual_url?: string | null;
  cad_trust_pct?: number;
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
  effective_cog_offset_mm?: [number, number, number];
  accessory_mass_delta_g?: number;
  nozzle_shield_mode?: "stock" | "microwaved" | "none";
  nozzle_shield_options?: Array<"stock" | "microwaved" | "none">;
  hotend_preset_id: string | null;
  hotend_mount_style: string;
  hotend_stiffness_n_per_mm: number;
  hotend_overall_length_mm: number;
  stage_stiffnesses_n_per_mm?: ToolheadStageStiffnessesData;
  use_cad_superelements?: boolean;
  use_hotend_bottom_brace?: boolean;
  cad_hotend_brace_available?: boolean;
  cad_toolhead_self_contained?: boolean;
  cad_split_carriage_compatible?: boolean;
  cad_hotend_self_contained?: boolean;
  cad_hotend_braced?: boolean;
  cad_superelements_active?: CadSuperelementSummary[];
  hotend_visual_glb_url?: string | null;
  toolhead_visual_glb_url?: string | null;
  carriage_mount_visual_glb_url?: string | null;
  toolhead_visual_ports_mm?: {
    port0_mm: [number, number, number];
    port1_mm: [number, number, number];
  } | null;
}

export interface ToolheadFlexSubCause {
  id: string;
  label: string;
  share_pct?: number;
  stiffness_n_per_mm?: number | null;
  severity?: "critical" | "high" | "moderate" | "low" | string;
  detail?: string;
  action_hint?: string;
}

export interface ToolheadFlexBreakdownData {
  rail_and_xbeam_pct: number;
  carriage_and_body_pct: number;
  hotend_and_nozzle_pct: number;
  effective_tip_stiffness_n_per_mm: number;
  hotend_stiffness_n_per_mm: number;
  carriage_mount_stiffness_n_per_mm: number;
  dominant_flex_source:
    | "rail_xbeam"
    | "carriage_body"
    | "rail_carriage"
    | "toolhead_body"
    | "hotend_nozzle";
  rail_carriage_pct?: number;
  toolhead_body_pct?: number;
  rail_carriage_stiffness_n_per_mm?: number;
  body_stiffness_n_per_mm?: number;
  effective_cog_offset_mm?: [number, number, number];
  nozzle_drop_mm?: number;
  assembled_mass_g?: number;
  sub_causes?: ToolheadFlexSubCause[];
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
    faces?: Partial<
      Record<
        "rear" | "left" | "right" | "top",
        {
          material_id: string;
          material_name: string;
          thickness_mm: number;
          color_hex: string;
        }
      >
    >;
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
  frame_outer_dims_mm?: [number, number, number];
  build_volume_mm: [number, number, number];
  build_origin_mm: [number, number, number];
  bed_size_mm: [number, number];
  bed_center_mm: [number, number, number];
  bed_node_id?: string;
  bed_frame_node_id?: string;
  bed_mount_type?: BedMountType;
  bed_z_drive?: string;
  feet_type?: FrameFeetType;
  suspension?: {
    line: string;
    length_mm: number;
    line_tension_n: number;
    lines: { anchor_mm: [number, number, number]; attach_mm: [number, number, number]; attach_node_id: string; anchor_node_id: string }[];
  } | null;
  ballast_mass_kg?: number;
  ballast_location?: FrameBallastLocationType;
  bed_support_points_mm?: [number, number, number][];
  bed_mount_points_mm?: [number, number, number][];
  bed_subframe_z_mm?: number;
  bed_mount_anchor_nodes?: string[];
  z_rails?: ZRailGeometryData[];
  z_leadscrews?: ZLeadscrewGeometryData[];
  toolhead_structure?: ToolheadStructureGeometryData;
  frame_bracing?: {
    type: string;
    active: boolean;
    cad_count: number;
    cad_asset_id: string | null;
    cad_asset_name?: string;
    unit_mass_g?: number;
    unit_cost_usd?: number;
    element_type?: string;
    num_elements?: number;
    kr_inplane_nm_rad?: number;
    kr_torsion_nm_rad?: number;
    corner_kr_inplane_nm_rad?: number;
    corner_kr_torsion_nm_rad?: number;
    cad_mass_kg: number;
    flat_bar_mass_kg: number;
    total_mass_kg: number;
    cost_usd: number;
    visuals: Array<{
      id: string;
      kind: string;
      asset_id: string;
      visual_glb_url: string;
      origin_m: [number, number, number];
      basis_x: [number, number, number];
      basis_y: [number, number, number];
      basis_z: [number, number, number];
      node_a: string;
      node_b: string;
    }>;
  };
  x_beam_visual?: {
    asset_id: string;
    visual_glb_url: string;
    length_mm: number;
  } | null;
  x_rail_visual?: {
    asset_id: string;
    visual_glb_url: string;
    length_mm: number;
  } | null;
  y_rail_visual?: {
    asset_id: string;
    visual_glb_url: string;
    length_mm: number;
  } | null;
  z_rail_visual?: {
    asset_id: string;
    visual_glb_url: string;
    length_mm: number;
  } | null;
  carriage_visuals?: {
    x?: { asset_id: string; visual_glb_url: string; port1_z_mm: number } | null;
    y?: { asset_id: string; visual_glb_url: string; port1_z_mm: number } | null;
    z?: { asset_id: string; visual_glb_url: string; port1_z_mm: number } | null;
  } | null;
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

export interface SuperelementStaticMetrics {
  id: string;
  part_group_id: string;
  mass_g: number;
  flex_xy_um: number;
  flex_x_um: number;
  flex_y_um: number;
  flex_gravity_um: number;
  twist_xy_mrad: number;
  twist_x_mrad: number;
  twist_y_mrad: number;
  twist_gravity_mrad: number;
  force_xy_n: number;
  force_x_n: number;
  force_y_n: number;
  force_gravity_n: number;
  moment_xy_nm: number;
  moment_x_nm: number;
  moment_y_nm: number;
  moment_gravity_nm: number;
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
  toolhead_parts?: ToolheadPartStaticData[];
  toolhead_part_fields?: Record<string, CadIntraPartFieldData>;
  part_fields?: Record<string, IntraPartField>;
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
  superelement_metrics?: Record<string, SuperelementStaticMetrics>;
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

export type StaticFEAResult = StaticFeaData;

export interface AdxlNozzleAxisDiagnostic {
  carriage_peak_hz: number;
  nozzle_peak_hz: number;
  nozzle_to_carriage_ratio: number;
  carriage_recommended_shaper: string;
  carriage_recommended_freq_hz: number;
  carriage_max_accel_mm_s2: number;
  nozzle_recommended_shaper: string;
  nozzle_recommended_freq_hz: number;
  nozzle_max_accel_mm_s2: number;
  carriage_raw_psd: number[];
  has_significant_divergence: boolean;
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
  mounting_dynamics?: MountingDynamics | null;
  belt_dynamics?: BeltDynamics | null;
  adxl_vs_nozzle_diagnostic?: {
    x: AdxlNozzleAxisDiagnostic;
    y: AdxlNozzleAxisDiagnostic;
  };
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
  motor_est_temp_c?: number;
  motor_max_winding_temp_c?: number;
  motor_thermal_risk?: boolean;
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
  motor_max_accel_x_mm_s2?: number;
  motor_max_accel_y_mm_s2?: number;
  motor_max_accel_mm_s2?: number;
  belt_max_accel_mm_s2?: number;
  motor_max_velocity_mm_s?: number;
  shaper_l1_accel_factor?: number;
  delta_workspace_valid: boolean;
}

export interface MotionLimitsData {
  shaper_max_accel_mm_s2: number;
  motor_max_accel_mm_s2: number;
  belt_max_accel_mm_s2: number;
  achievable_max_accel_mm_s2: number;
  accel_limiting_factor: "motor" | "shaper" | "belt";
  motor_max_velocity_mm_s: number;
  flow_max_velocity_mm_s: number;
  configured_max_velocity_mm_s: number;
  configured_max_accel_mm_s2: number;
  speed_limiting_factor: "motor_voltage" | "hotend_flow";
  shaper_l1_accel_factor: number;
}

export interface ThermalStructuralReport {
  chamber_temp_c: number;
  delta_t_k: number;
  gantry_bimetallic_bow_um: number;
  gantry_backer_reduction_pct: number;
  gantry_bow_direction?: "toward_rail" | "away_from_rail" | "neutral" | string;
  bed_target_temp_c: number;
  bed_mount_type: string;
  bed_gradient_bow_um: number;
  bed_buckling_taco_um: number;
  frame_z_drift_um: number;
  bracket_modulus_retention_pct: number;
  bracket_tg_warning: boolean;
  x_beam_modulus_retention_pct?: number;
  x_beam_shear_retention_pct?: number;
  x_beam_tg_warning?: boolean;
  x_beam_has_tg?: boolean;
  total_thermal_z_error_um: number;
  rail_newmark_gamma: number;
  rail_carriage_roll_nm_per_rad: number;
  rail_carriage_radial_n_per_um: number;
  motor_mount_temp_c?: number;
  bearing_temp_c?: number;
  bearing_cte_clearance_um?: number;
  bearing_cte_pressfit_warning?: boolean;
  toolhead_belt_cog_z_offset_mm?: number;
  toolhead_nozzle_y_lever_mm?: number;
  toolhead_x_roll_warning?: boolean;
}

export interface ChamberCpapReport {
  chamber_heater_id: string;
  chamber_heater_power_w: number;
  chamber_insulation_mm: number;
  enclosure_ua_w_per_k: number;
  chamber_steady_temp_c: number;
  chamber_heatup_time_60c_s: number | null;
  cpap_blower_id: string;
  cpap_hose_id_mm: number;
  cpap_pwm_pct: number;
  cpap_effective_pwm_pct: number;
  cpap_hose_flow_factor: number;
  cpap_delivered_dm3_s: number;
  cpap_delivered_cfm: number;
  cpap_exhaust_temp_rise_c: number;
  cpap_outlet_air_temp_c: number;
  cpap_hose_moving_mass_g: number;
  kalico_shared_duct_active: boolean;
  cpap_cooling_notes: string[];
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
  is_spatial_spread_pct?: number;
  max_cross_coupling_pct: number;
  reflected_rotor_mass_g: number;
  drive_type: string;
  midband_instability_risk: boolean;
}

export interface ZigzagTraceData {
  ideal_x_mm: number[];
  ideal_y_mm: number[];
  actual_x_mm: number[];
  actual_y_mm: number[];
  unshaped_x_mm: number[];
  unshaped_y_mm: number[];
  perp_error_um: number[];
}

export interface TimeDomainMetrologyData {
  corner_rounding_um: number;
  corner_overshoot_um: number;
  ringing_peak_to_peak_um: number;
  ringing_unshaped_um: number;
  ringing_reduction_pct: number;
  ringing_decay_mm: number;
  printed_path_error_um?: number;
  shaper_path_smoothing_um?: number;
  cross_track_error_um?: number;
  along_track_lag_um?: number;
  corner_trace: {
    ideal_x_mm: number[];
    ideal_y_mm: number[];
    actual_x_mm: number[];
    actual_y_mm: number[];
    unshaped_x_mm: number[];
    unshaped_y_mm: number[];
    shaped_cmd_x_mm?: number[];
    shaped_cmd_y_mm?: number[];
    wall_dist_mm?: number[];
    wall_dev_unshaped_um?: number[];
    wall_dev_shaped_um?: number[];
  };
  zigzag_path_error_um?: number;
  zigzag_shaper_smoothing_um?: number;
  zigzag_residual_ringing_um?: number;
  zigzag_unshaped_ringing_um?: number;
  zigzag_trace?: ZigzagTraceData;
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
  part_group_sensitivities?: Record<string, SensitivityStiffenTarget>;
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
  mount_settlement_um?: number;
  rigid_tilt_um?: number;
  plate_flatness_um?: number;
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

export interface VirtualPrintReport {
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
  max_corner_smoothing_um?: number;
  underextrusion_segments: number;
  ringing_segments: number;
  smoothing_segments?: number;
  step_skip_segments: number;
  belt_slip_segments?: number;
  cooling_warning_layers: number;
  out_of_bounds_issues: string[];
  print_quality_score: number;
  segments: ToolpathSegment[];
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
  motion_limits?: MotionLimitsData;
  bom_total_cost_usd?: number;
  bom_total_mass_kg?: number;
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
    chamber_cpap?: ChamberCpapReport;
  };
  virtual_print: VirtualPrintReport;
  effective_model_params?: EffectiveModelParamsData;
  model_confidence?: ModelConfidenceReport;
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

export type ModalScaleMode = "impact" | "shape";

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
