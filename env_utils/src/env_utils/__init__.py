"""Environment modelling utilities package.

Provides reusable helpers for handling Lanelet2 maps,
sensor calibration, global pose conversions, and dataset preparation.
"""

from .common import (
    DEFAULT_SENSOR,
    DEFAULT_POSE_KEY,
    DEFAULT_SRC_EPSG,
    PreparedEnvironment,
    add_dataset_arguments,
    apply_transformation,
    ego_pose_to_matrix,
    ensure_transformer_available,
    epsg_to_latlon,
    latlon_to_epsg,
    load_point_cloud,
    load_sample,
    load_transformation_matrix,
    matrix_to_quaternion,
    prepare_environment,
    transform_point_cloud,
)
from .lanelet_map import EgoPose, LaneletMapInterface, LaneletQueryResult

__all__ = [
    "DEFAULT_SENSOR",
    "DEFAULT_POSE_KEY",
    "DEFAULT_SRC_EPSG",
    "PreparedEnvironment",
    "add_dataset_arguments",
    "apply_transformation",
    "ego_pose_to_matrix",
    "ensure_transformer_available",
    "epsg_to_latlon",
    "latlon_to_epsg",
    "load_point_cloud",
    "load_sample",
    "load_transformation_matrix",
    "matrix_to_quaternion",
    "prepare_environment",
    "transform_point_cloud",
    "EgoPose",
    "LaneletMapInterface",
    "LaneletQueryResult",
]
