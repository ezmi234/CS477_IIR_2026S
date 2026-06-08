import math
import os


JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

HOME_JOINTS = [0.0, -math.pi / 2.0, 1.0, -math.pi / 3.0, -math.pi / 2.0, 0.0]
OBSERVE_JOINTS = [0.0, -1.35, 1.45, -1.75, -math.pi / 2.0, 0.0]
REFERENCE_GRASP_JOINTS = [
    0.16456877764159317,
    -1.4026709127086092,
    1.6609417499881791,
    -1.829975190229056,
    -1.570936444921175,
    0.0030041786525445556,
]

PLACE_CONFIGS = {
    'left_storage': {
        'range_x': [-0.124, 0.117],
        'range_y': [0.381, 0.717],
        'release_z': 0.34,
        'release_z_from_grasp': True,
        'storage_base_z': 0.06,
        'object_clearance_z': 0.15,
        'use_grasp_orientation': True,
        'approach_z': 0.15,
        'retreat_z': 0.20,
        'approach_duration': 2.8,
        'release_duration': 2.0,
        'open_timeout': 2.5,
        'post_release_sleep': 0.8,
        'retreat_duration': 3.0,
        'return_home_after_place': True,
        'return_home_duration': 4.0,
    },
    'right_storage': {
        'range_x': [-0.124, 0.117],
        'range_y': [-0.717, -0.381],
        'release_z': 0.34,
        'release_z_from_grasp': True,
        'storage_base_z': 0.06,
        'object_clearance_z': 0.15,
        'use_grasp_orientation': True,
        'approach_z': 0.15,
        'retreat_z': 0.20,
        'approach_duration': 2.8,
        'release_duration': 2.0,
        'open_timeout': 2.5,
        'post_release_sleep': 0.8,
        'retreat_duration': 3.0,
        'return_home_after_place': True,
        'return_home_duration': 4.0,
    },
    'shelf': {
        'x': 0.86,
        'y_slots': [-0.30],
        'release_z': 0.50,
        'approach_x_offset': -0.16,
        'retreat_x_offset': -0.28,
        'retreat_lift_z': 0.02,
        'approach_duration': 2.8,
        'release_duration': 2.0,
        'open_timeout': 2.5,
        'post_release_sleep': 0.8,
        'retreat_duration': 3.0,
        'return_home_after_place': True,
        'return_home_duration': 4.0,
    },
}

BOOKSHELF_WRIST_FLIP_OBJECTS = {'banana', 'hammer'}

DEBUG_GROUND_TRUTH_Z_OFFSETS = {
    'book': -0.392,
    'eraser': -0.52,
    'soap': -0.52,
    'soap2': -0.52,
    'snacks': -0.52,
    'biscuits': -0.52,
    'glue': -0.52,
    'mustard_bottle': -0.52,
    'sticky_notes': -0.52,
    'coke_can': -0.56,
    'meat_can': -0.56,
    'banana': -0.58,
    'strawberry': -0.545,
    'hammer': -0.58,
}

HARDCODED_PICK_TARGETS = {
    'coke_can': (0.548, 0.248, 0.011),
    'meat_can': (0.499, 0.000, -0.008),
    'banana': (0.598, -0.110, -0.055),
    'strawberry': (0.495, -0.148, -0.037),
    'hammer': (0.500, 0.150, -0.057),
}

GRIPPER_CLOSE_POSITIONS = {
    'coke_can': 0.42,
    'meat_can': 0.38,
    'strawberry': 0.70,
    'banana': 0.90,
    'hammer': 0.75,
}

PICK_APPROACH_HEIGHTS = {
    'banana': 0.22,
}

PICK_LIFT_HEIGHTS = {
    'banana': 0.32,
    'hammer': 0.35,
}

PLACE_APPROACH_HEIGHTS = {
    'hammer': 0.10,
}

PICK_POSITION_OFFSETS = {
    'strawberry': (0.0, -0.012, 0.0),
    'hammer': (-0.04, 0.06, 0.0),
}

OBJECT_PICK_OVERRIDES = {
    'meat_can': {
        'approach_duration': 2.3,
        'descent_duration': 1.8,
        'lift_duration': 2.2,
        'close_timeout': 2.5,
    },
    'coke_can': {
        'approach_duration': 2.3,
        'descent_duration': 1.8,
        'lift_duration': 2.2,
        'close_timeout': 2.5,
    },
    'strawberry': {
        'approach_duration': 2.4,
        'descent_duration': 1.8,
        'lift_duration': 2.2,
        'close_timeout': 2.5,
    },
    'banana': {
        'approach_height': 0.18,
        'lift_height': 0.30,
        'grasp_z_offset': -0.010,
        # When the pose already comes from /vision/selected_grasp_base, do not
        # descend again below it. The vision server already clamps bad depth.
        # The RGB-D banana affordance returns a point on the visible body.
        # Descend a little below that surface seed and allow a lower clamp;
        # otherwise the gripper often closes just above the thin banana.
        'vision_grasp_z_offset': -0.018,
        'vision_grasp_z_min': -0.080,
        'vision_grasp_z_max': -0.025,
        'close_force': 1.0,
        'close_timeout': 3.5,
        'approach_duration': 3.0,
        'descent_duration': 2.4,
        'lift_duration': 3.0,
        'post_close_sleep': 0.35,
        'use_vision_grasp_orientation': True,
    },
    'hammer': {
        'approach_height': 0.18,
        'lift_height': 0.34,
        'grasp_z_offset': -0.010,
        'vision_grasp_z_offset': 0.0,
        'vision_grasp_z_min': -0.045,
        'vision_grasp_z_max': 0.040,
        'close_timeout': 3.0,
        'approach_duration': 3.2,
        'descent_duration': 2.8,
        'lift_duration': 3.2,
        'post_close_sleep': 0.35,
        'use_vision_grasp_orientation': True,
    },
}

OBJECT_GRASP_PROFILES = {
    'banana': {
        'strategy': 'banana_mask_distance_transform',
        'close_pos': 0.90,
        'close_timeout': 3.5,
        'approach_height': 0.16,
        'grasp_z_offset': -0.010,
        'vision_grasp_z_offset': -0.018,
        'vision_grasp_z_min': -0.080,
        'vision_grasp_z_max': -0.025,
        'lift_height': 0.28,
        'yaw_mode': 'local_tangent',
        'risk': 'medium',
        'approach_duration': 3.0,
        'descent_duration': 2.4,
        'lift_duration': 3.0,
        'post_close_sleep': 0.35,
        'use_vision_grasp_orientation': True,
    },
    'meat_can': {
        'strategy': 'can_side_or_top_center',
        'close_pos': 0.38,
        'close_timeout': 2.5,
        'approach_height': 0.12,
        'grasp_z_offset': -0.010,
        'lift_height': 0.18,
        'yaw_mode': 'symmetric_default',
        'risk': 'low',
        'approach_duration': 2.3,
        'descent_duration': 1.8,
        'lift_duration': 2.2,
    },
    'coke_can': {
        'strategy': 'can_side_or_top_center',
        'close_pos': 0.42,
        'close_timeout': 2.5,
        'approach_height': 0.12,
        'grasp_z_offset': -0.010,
        'lift_height': 0.18,
        'yaw_mode': 'symmetric_default',
        'risk': 'low',
        'approach_duration': 2.3,
        'descent_duration': 1.8,
        'lift_duration': 2.2,
    },
    'strawberry': {
        'strategy': 'compact_top_center',
        'close_pos': 0.70,
        'close_timeout': 2.5,
        'approach_height': 0.10,
        'grasp_z_offset': -0.006,
        'lift_height': 0.14,
        'yaw_mode': 'symmetric_default',
        'risk': 'low',
        'approach_duration': 2.4,
        'descent_duration': 1.8,
        'lift_duration': 2.2,
    },
    'hammer': {
        'strategy': 'handle_grasp',
        'close_pos': 0.75,
        'close_timeout': 3.0,
        'approach_height': 0.16,
        'grasp_z_offset': -0.012,
        'vision_grasp_z_offset': 0.0,
        'vision_grasp_z_min': -0.045,
        'vision_grasp_z_max': 0.040,
        'lift_height': 0.22,
        'yaw_mode': 'handle_axis',
        'risk': 'high',
        'approach_duration': 3.2,
        'descent_duration': 2.8,
        'lift_duration': 3.2,
        'post_close_sleep': 0.35,
        'use_vision_grasp_orientation': True,
    },
}

UNCERTAIN_PICK_PROFILE = {
    'velocity_scale': 0.45,
    'acceleration_scale': 0.45,
    'approach_height': 0.14,
    'lift_height': 0.10,
    'close_force': 0.8,
    'require_lift_verification': True,
}

STORAGE_OBJECT_OVERRIDES = {
    'banana': {
        'approach_duration': 2.0,
        'release_duration': 1.2,
        'retreat_duration': 2.2,
    },
}

IK_SEEDS = [
    HOME_JOINTS,
    [0.1646, -1.3117, 1.9484, -2.2074, -1.5707, 0.1646],
    [0.1617, -1.4298, 1.3224, -1.4706, -1.5801, 0.1637],
    [0.1645, -1.4026, 1.6609, -1.8299, -1.5709, 0.0030],
    [1.3680, -1.4297, 1.2446, -1.3691, -1.5598, 2.1546],
    [-0.50, -1.30, 1.40, -1.60, -1.57, -0.50],
    [0.50, -1.30, 1.40, -1.60, -1.57, 0.50],
]

OBJECT_ALIASES = {
    'banana': ['banana'],
    'meat_can': ['meat can', 'meat_can', 'tin can', 'food can', 'can of meat', 'meat'],
    'coke_can': [
        'coke can', 'coke_can', 'cola can', 'red can', 'red soda can',
        'soda can', 'drink can', 'beverage can', 'coke', 'cola',
    ],
    'strawberry': ['strawberry'],
    'hammer': ['hammer'],
    'book': ['book'],
    'eraser': ['eraser'],
    'soap': ['soap'],
    'soap2': ['soap2'],
    'snacks': ['snack', 'snacks'],
    'biscuits': ['biscuits box', 'biscuit box', 'biscuits', 'biscuit'],
    'glue': ['glue'],
    'mustard_bottle': ['mustard bottle', 'mustard_bottle'],
    'sticky_notes': ['sticky notes', 'sticky_notes', 'sticky note'],
}

TASK_EXECUTION_PRIORITY = {
    'meat_can': 0,
    'coke_can': 1,
    'strawberry': 2,
    'banana': 3,
    'hammer': 4,
}

OBJECT_SUCCESS_PRIORITY = dict(TASK_EXECUTION_PRIORITY)

OBJECT_RISK = {
    'meat_can': 'low',
    'coke_can': 'low',
    'strawberry': 'low',
    'banana': 'medium',
    'hammer': 'high',
}

MIN_CONFIDENCE = {
    'meat_can': 0.040,
    'coke_can': 0.040,
    'strawberry': 0.040,
    'banana': 0.070,
    'hammer': 0.180,
}

MIN_GRASP_SCORE = {
    'meat_can': 0.12,
    'coke_can': 0.12,
    'strawberry': 0.10,
    'banana': 0.18,
    'hammer': 0.35,
}

MIN_FINAL_CANDIDATE_SCORE = {
    'meat_can': 0.12,
    'coke_can': 0.15,
    'strawberry': 0.15,
    'banana': 0.50,
    'hammer': 0.70,
}

TRIAL_TIME_LIMIT_SEC = 8.0 * 60.0
SKIP_HAMMER_AFTER_SEC = 6.0 * 60.0
LOW_RISK_ONLY_AFTER_SEC = 7.0 * 60.0

DESTINATION_ALIASES = {
    'left_storage': [
        'left storage', 'left_storage', 'storage box a', 'storage a',
        'storage a basket', 'box a', 'left basket', 'left',
    ],
    'right_storage': [
        'right storage', 'right_storage', 'storage box b', 'storage b',
        'storage b basket', 'box b', 'right basket', 'right',
    ],
    'shelf': ['bookshelf', 'book shelf', 'shelf'],
}


def _load_package_yaml(filename):
    try:
        import yaml
        from ament_index_python.packages import get_package_share_directory

        config_path = os.path.join(get_package_share_directory('team_1'), 'config', filename)
        if not os.path.exists(config_path):
            config_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', 'config', filename)
            )
        if not os.path.exists(config_path):
            return {}
        with open(config_path, 'r', encoding='utf-8') as stream:
            loaded = yaml.safe_load(stream) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_target_config():
    targets = _load_package_yaml('targets.yaml')
    for name, values in targets.items():
        if name in PLACE_CONFIGS and isinstance(values, dict):
            _deep_update(PLACE_CONFIGS[name], values)


def _apply_motion_config():
    motion = _load_package_yaml('motion.yaml')
    uncertain = motion.get('uncertain_pick_profile', {})
    if isinstance(uncertain, dict):
        _deep_update(UNCERTAIN_PICK_PROFILE, uncertain)
    profile_config = motion.get('motion_profile', {})
    if isinstance(profile_config, dict):
        mode = str(profile_config.get('mode', 'competition') or 'competition')
        mode_values = profile_config.get(mode, {})
        if isinstance(mode_values, dict):
            for name, values in mode_values.items():
                if not isinstance(values, dict):
                    continue
                key = str(name).strip().lower().replace(' ', '_')
                current = dict(OBJECT_GRASP_PROFILES.get(key, {}))
                for field in ('velocity_scale', 'acceleration_scale'):
                    if field in values:
                        try:
                            current[field] = float(values[field])
                        except (TypeError, ValueError):
                            pass
                OBJECT_GRASP_PROFILES[key] = current
    object_pick = motion.get('object_pick', {})
    if isinstance(object_pick, dict):
        for name, values in object_pick.items():
            if isinstance(values, dict):
                current = dict(OBJECT_PICK_OVERRIDES.get(name, {}))
                _deep_update(current, values)
                OBJECT_PICK_OVERRIDES[name] = current
    gripper = motion.get('gripper', {})
    if isinstance(gripper, dict):
        for name, values in gripper.items():
            if not isinstance(values, dict):
                continue
            if 'close_pos' in values:
                try:
                    GRIPPER_CLOSE_POSITIONS[name] = float(values['close_pos'])
                except (TypeError, ValueError):
                    pass
            if 'close_timeout' in values:
                current = dict(OBJECT_PICK_OVERRIDES.get(name, {}))
                try:
                    current['close_timeout'] = float(values['close_timeout'])
                    OBJECT_PICK_OVERRIDES[name] = current
                except (TypeError, ValueError):
                    pass


def _apply_grasp_profile_config():
    grasp = _load_package_yaml('grasp.yaml')
    profiles = grasp.get('object_grasp_profiles', {})
    if not isinstance(profiles, dict):
        return
    for name, values in profiles.items():
        if not isinstance(values, dict):
            continue
        key = str(name).strip().lower().replace(' ', '_')
        current = dict(OBJECT_GRASP_PROFILES.get(key, {}))
        _deep_update(current, values)
        OBJECT_GRASP_PROFILES[key] = current

        if 'close_pos' in current:
            try:
                GRIPPER_CLOSE_POSITIONS[key] = float(current['close_pos'])
            except (TypeError, ValueError):
                pass

        pick_override = dict(OBJECT_PICK_OVERRIDES.get(key, {}))
        for field in (
            'approach_height',
            'lift_height',
            'grasp_z_offset',
            'vision_grasp_z_offset',
            'vision_grasp_z_min',
            'vision_grasp_z_max',
            'close_timeout',
            'approach_duration',
            'descent_duration',
            'lift_duration',
            'post_close_sleep',
            'close_force',
            'use_vision_grasp_orientation',
            'velocity_scale',
            'acceleration_scale',
        ):
            if field in current:
                pick_override[field] = current[field]
        if pick_override:
            OBJECT_PICK_OVERRIDES[key] = pick_override


_apply_target_config()
_apply_motion_config()
_apply_grasp_profile_config()
