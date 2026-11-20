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

export interface CatalogResponse {
  materials: MaterialItem[];
  profiles: ProfileItem[];
  filaments: FilamentItem[];
  part_presets: PartPresetItem[];
  presets: PresetItem[];
}

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

export interface PrinterSpecData {
  name: string;
  description: string;
  architecture: "corexy" | "cartesian_bedslinger" | "delta";
  frame: {
    width_mm: number;
    depth_mm: number;
    height_mm: number;
    profile: string;
    material: string;
    bracing: "none" | "corner_gussets" | "rear_cross" | "full_truss";
    brace_profile: string;
    brace_material: string;
    enclosed_panels: boolean;
    panel_mounting: "none" | "clip_enclosure" | "bolted_structural";
    panel_material: string;
    panel_thickness_mm: number;
  };
  gantry: {
    x_beam_profile: string;
    x_beam_material: string;
    x_rail_profile: string;
    y_rail_profile: string;
    rail_material: string;
    dual_x_rails: boolean;
    drive_mode: "2wd" | "awd";
    double_shear_pulleys: boolean;
    carriage_mass_g: number;
    belt: {
      type: "gt2_6mm" | "gt2_9mm" | "gt2_12mm";
      core: "fiberglass" | "steel" | "aramid";
      pulley_teeth: number;
      pitch_mm: number;
      pretension_n: number;
    };
  };
  kinematics: {
    build_volume_mm: [number, number, number];
    max_velocity_mm_s: number;
    max_accel_mm_s2: number;
    max_z_velocity_mm_s: number;
    max_z_accel_mm_s2: number;
    square_corner_velocity_mm_s: number;
    input_shaper: "auto" | "none" | "zv" | "mzv" | "ei" | "2hump_ei";
    delta_arm_length_mm: number;
    delta_radius_mm: number;
    xy_stepper: {
      name: string;
      holding_torque_nm: number;
      rated_current_a: number;
      inductance_mh: number;
      supply_voltage_v: number;
      rotor_inertia_kg_m2: number;
      motor_mass_g: number;
      steps_per_rev: number;
    };
  };
  toolhead: {
    name: string;
    total_mass_g: number;
    cog_offset_mm: [number, number, number];
    envelope_mm: [number, number, number];
    hotend: {
      name: string;
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
    z_drive: string;
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

export interface ModeShapeData {
  mode_number: number;
  frequency_hz: number;
  dominant_axis: string;
  toolhead_participation_xyz: [number, number, number];
  nodal_vectors: Record<string, [number, number, number]>;
}

export interface ShaperData {
  axis: string;
  effective_freq_hz: number;
  structural_freq_hz: number;
  belt_freq_hz: number;
  recommended_shaper: string;
  max_accel_mm_s2: number;
  estimated_ringing_ratio: number;
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
  geometry: {
    architecture: string;
    drive_mode?: "2wd" | "awd";
    double_shear_pulleys?: boolean;
    panels?: {
      enabled: boolean;
      mounting: string;
      material_id: string;
      material_name: string;
      thickness_mm: number;
      color_hex: string;
    };
    frame_dims_mm: [number, number, number];
    build_volume_mm: [number, number, number];
    build_origin_mm: [number, number, number];
    bed_size_mm: [number, number];
    bed_center_mm: [number, number, number];
    toolhead_node_id: string;
    toolhead_envelope_mm: [number, number, number];
    nodes: GeometryNode[];
    members: GeometryMember[];
    mass_summary: {
      frame_mass_kg: number;
      total_mass_kg: number;
      x_moving_mass_g: number;
      y_moving_mass_g: number;
      estimated_material_cost_usd: number;
    };
  };
  collision: {
    has_collision: boolean;
    clearance_x_mm: number;
    clearance_y_mm: number;
    clearance_z_mm: number;
    issues: string[];
  };
  static_fea: {
    gravity_sag_toolhead_um: number;
    accel_deflection_x_um: number;
    accel_deflection_y_um: number;
    combined_max_deflection_um: number;
    toolhead_stiffness_x_n_per_mm: number;
    toolhead_stiffness_y_n_per_mm: number;
    toolhead_stiffness_z_n_per_mm: number;
    max_von_mises_mpa: number;
    min_safety_factor: number;
    peak_inertial_force_x_n: number;
    peak_inertial_force_y_n: number;
    nodal_displacements_mm: Record<string, [number, number, number]>;
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
  };
  modal_fea: {
    fundamental_freq_hz: number;
    recommended_max_accel_mm_s2: number;
    shaper_x: ShaperData | null;
    shaper_y: ShaperData | null;
    modes: ModeShapeData[];
  };
  kinematics: {
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
    delta_workspace_valid: boolean;
  };
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

export type ViewMode = "assembly" | "fea_stress" | "modal" | "toolpath" | "solid_part" | "topology";

export type CompareMode = "last" | "baseline";

export interface HistorySnapshot {
  sim: SimulationResponse;
  bom: PrinterBOMResponse | null;
  spec: PrinterSpecData;
  filamentId: string;
  label: string | null;
}
