import math


JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

HOME_JOINTS = [0.0, -math.pi / 2.0, 1.0, -math.pi / 3.0, -math.pi / 2.0, 0.0]
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
        'approach_z': 0.08,
        'retreat_z': 0.08,
    },
    'right_storage': {
        'range_x': [-0.124, 0.117],
        'range_y': [-0.717, -0.381],
        'release_z': 0.34,
        'approach_z': 0.08,
        'retreat_z': 0.08,
    },
    'shelf': {
        'x': 0.925,
        'y_slots': [-0.215, -0.30, -0.385],
        'release_z': 0.21,
        'approach_x_offset': -0.20,
        'retreat_x_offset': -0.30,
        'retreat_lift_z': 0.03,
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
    'coke_can': math.radians(14),
    'meat_can': math.radians(19),
    'banana': math.radians(27),
    'strawberry': 0.8,
    'hammer': 0.8,
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
    'meat_can': ['meat can', 'meat_can', 'can of meat', 'meat'],
    'coke_can': ['coke can', 'coke_can', 'coke', 'cola'],
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
    'strawberry': 0,
}

DESTINATION_ALIASES = {
    'left_storage': [
        'left storage', 'left_storage', 'storage box a', 'storage a',
        'storage a basket', 'left basket', 'left',
    ],
    'right_storage': [
        'right storage', 'right_storage', 'storage box b', 'storage b',
        'storage b basket', 'right basket', 'right',
    ],
    'shelf': ['bookshelf', 'book shelf', 'shelf'],
}
