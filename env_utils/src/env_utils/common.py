"""Shared helpers for environment representation scripts."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

try:
    from pyproj import Transformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Transformer = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from .lanelet_map import EgoPose, LaneletMapInterface

DEFAULT_SENSOR = "lidar_os1"
DEFAULT_POSE_KEY = "pose"
DEFAULT_SRC_EPSG = "EPSG:25832"
WGS84_EPSG = "EPSG:4326"


@dataclass
class EgoPoseRaw:
    position: np.ndarray
    orientation_quat: np.ndarray
    timestamp: float | None = None


@dataclass
class PreparedEnvironment:
    sensor_points: np.ndarray
    ego_points: np.ndarray
    world_points: np.ndarray
    map_points: np.ndarray
    ego_pose_world: np.ndarray
    ego_pose_map: np.ndarray
    ego_pose: "EgoPose"
    map_interface: "LaneletMapInterface"
    origin_latlonalt: tuple[float, float, float]
    origin_eastnorthalt: tuple[float, float, float]
    sample: dict[str, Any]


def load_point_cloud(path: str | Path) -> np.ndarray:
    """Load a point cloud from .npy, .npz, .pcd (via open3d), or .csv (XYZ)."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            if "points" in data:
                points = data["points"]
            else:
                points = next(iter(data.values()))
    elif suffix == ".pcd":
        try:
            import open3d as o3d  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("open3d is required to load .pcd files") from exc
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            points = np.concatenate([points, colors], axis=1)
    elif suffix in {".csv", ".txt"}:
        points = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported point cloud format '{suffix}'.")

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Point cloud must have shape (N, >=3).")
    return points


def load_transformation_matrix(path: str | Path) -> np.ndarray:
    """Load a 4x4 homogeneous matrix from .npy, .npz, .json, or plain text."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            matrix = next(iter(data.values()))
    elif suffix in {".json", ".yaml", ".yml"}:
        data = _read_structured_file(path)
        matrix = np.asarray(data["matrix"], dtype=float)
    else:
        matrix = np.loadtxt(path)

    if matrix.shape != (4, 4):
        raise ValueError("Transformation matrix must be 4x4.")
    return matrix


def load_ego_pose(path: str | Path) -> EgoPoseRaw:
    """Load an ego pose description from JSON/YAML."""

    path = Path(path)
    data = _read_structured_file(path)

    position = np.asarray(data.get("position", [0.0, 0.0, 0.0]), dtype=float)
    if "orientation_quat" in data:
        quat = np.asarray(data["orientation_quat"], dtype=float)
    else:
        yaw = float(data.get("yaw", 0.0))
        pitch = float(data.get("pitch", 0.0))
        roll = float(data.get("roll", 0.0))
        quat = euler_to_quaternion(roll, pitch, yaw)

    timestamp = data.get("timestamp")
    return EgoPoseRaw(position=position, orientation_quat=quat, timestamp=timestamp)


def _read_structured_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("pyyaml is required to parse YAML ego poses.")
        return yaml.safe_load(path.read_text())
    raise ValueError(f"Unsupported structured file type '{suffix}'.")


def apply_transformation(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transformation to the point cloud."""

    homog = np.ones((points.shape[0], 4), dtype=float)
    homog[:, :3] = points[:, :3]
    transformed = homog @ matrix.T
    return transformed[:, :3]


def transform_point_cloud(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply transform while preserving additional attributes."""

    xyz = apply_transformation(points[:, :3], matrix)
    if points.shape[1] > 3:
        return np.concatenate([xyz, points[:, 3:]], axis=1)
    return xyz


def ego_pose_to_matrix(position: np.ndarray, orientation_quat: np.ndarray) -> np.ndarray:
    """Convert pose to a homogeneous transformation matrix."""

    rotation = quaternion_to_matrix(orientation_quat)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position[:3]
    return matrix


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix."""

    if quaternion.shape != (4,):
        raise ValueError("Quaternion must have shape (4,).")
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=float,
    )


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z], dtype=float)


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z)."""

    if matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    m = matrix
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quaternion = np.array([w, x, y, z], dtype=float)
    return quaternion / np.linalg.norm(quaternion)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Tait-Bryan ZYX rotation to matrix."""

    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def load_sample(sample_path: str | Path) -> dict[str, Any]:
    """Load a multi-sensor sample stored as pickle."""

    path = Path(sample_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    if not isinstance(sample, dict):
        raise TypeError("Sample pickle must contain a dictionary.")
    return sample


def get_sensor_points(sample: dict[str, Any], sensor_key: str = DEFAULT_SENSOR) -> np.ndarray:
    """Extract a point cloud for the requested sensor."""

    if sensor_key not in sample:
        available = ", ".join(sorted(sample.keys()))
        raise KeyError(f"Sensor '{sensor_key}' not found in sample. Available keys: {available}")
    points = np.asarray(sample[sensor_key])
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Sample entry '{sensor_key}' must have shape (N, >=3).")
    return points


def load_pose_matrix(sample: dict[str, Any], key: str = DEFAULT_POSE_KEY) -> np.ndarray:
    """Return the 4x4 pose matrix stored in the sample."""

    if key not in sample:
        available = ", ".join(sorted(sample.keys()))
        raise KeyError(f"Pose key '{key}' not found in sample. Available keys: {available}")
    matrix = np.asarray(sample[key], dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Pose matrix under key '{key}' must be 4x4.")
    return matrix


def load_sensor_extrinsics(calib_root: str | Path, sensor_name: str = DEFAULT_SENSOR) -> np.ndarray:
    """Load homogeneous transform from sensor to ego frame."""

    calib_root = Path(calib_root)
    pkl_path = calib_root / "demandaer_extrinsics.pkl"
    if pkl_path.exists():
        with pkl_path.open("rb") as handle:
            data = pickle.load(handle)
        if sensor_name not in data:
            available = ", ".join(sorted(data.keys()))
            raise KeyError(f"Sensor '{sensor_name}' not in '{pkl_path}'. Available: {available}")
        matrix = np.asarray(data[sensor_name], dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"Extrinsic matrix for '{sensor_name}' must be 4x4.")
        return matrix

    txt_path = calib_root / "extrinsics" / f"{sensor_name}.txt"
    if txt_path.exists():
        return _load_extrinsics_from_text(txt_path)

    raise FileNotFoundError(f"No extrinsic calibration found for sensor '{sensor_name}'.")


def _load_extrinsics_from_text(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=float)
    if values.size != 6:
        raise ValueError(f"Extrinsic text file '{path}' must contain 6 values (x y z roll pitch yaw).")
    translation = values[:3]
    roll, pitch, yaw = values[3:]
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rpy_to_matrix(roll, pitch, yaw)
    matrix[:3, 3] = translation
    return matrix


def ensure_transformer_available() -> None:
    if Transformer is None:
        raise RuntimeError("pyproj is required for EPSG transformations. Install via `pip install pyproj`.")


@lru_cache(maxsize=32)
def _get_transformer(src_epsg: str, dst_epsg: str) -> Transformer:
    ensure_transformer_available()
    return Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)


def epsg_to_latlon(position: np.ndarray, src_epsg: str = DEFAULT_SRC_EPSG) -> tuple[float, float, float]:
    """Convert projected coordinates to latitude/longitude."""

    transformer = _get_transformer(src_epsg, WGS84_EPSG)
    lon, lat = transformer.transform(float(position[0]), float(position[1]))
    alt = float(position[2])
    return float(lat), float(lon), alt


def latlon_to_epsg(lat: float, lon: float, alt: float, dst_epsg: str = DEFAULT_SRC_EPSG) -> tuple[float, float, float]:
    """Convert latitude/longitude to projected coordinates."""

    transformer = _get_transformer(WGS84_EPSG, dst_epsg)
    x, y = transformer.transform(float(lon), float(lat))
    return float(x), float(y), float(alt)


def prepare_environment(
    sample_path: str | Path,
    calib_root: str | Path,
    lanelet_map_path: str | Path,
    *,
    sensor_name: str = DEFAULT_SENSOR,
    pose_key: str = DEFAULT_POSE_KEY,
    src_epsg: str = DEFAULT_SRC_EPSG,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
    origin_alt: float | None = None,
) -> PreparedEnvironment:
    """Load sample, calibrations, and map interface for downstream tasks."""

    from .lanelet_map import EgoPose, LaneletMapInterface

    sample = load_sample(sample_path)
    sensor_points = get_sensor_points(sample, sensor_name)
    extrinsic = load_sensor_extrinsics(calib_root, sensor_name)
    pose_matrix_world = load_pose_matrix(sample, pose_key)

    ego_points = transform_point_cloud(sensor_points, extrinsic)
    world_points = transform_point_cloud(ego_points, pose_matrix_world)

    base_position_world = pose_matrix_world[:3, 3]
    lat_pose, lon_pose, alt_pose = epsg_to_latlon(base_position_world, src_epsg=src_epsg)

    origin_lat_val = origin_lat if origin_lat is not None else lat_pose
    origin_lon_val = origin_lon if origin_lon is not None else lon_pose
    origin_alt_val = origin_alt if origin_alt is not None else alt_pose

    map_interface = LaneletMapInterface(
        lanelet_map_path,
        origin=(origin_lat_val, origin_lon_val, origin_alt_val),
        projector="utm",
    )

    origin_easting, origin_northing, origin_altitude = latlon_to_epsg(
        origin_lat_val, origin_lon_val, origin_alt_val, dst_epsg=src_epsg
    )

    transformer_world_to_geo = _get_transformer(src_epsg, WGS84_EPSG)
    world_lon, world_lat = transformer_world_to_geo.transform(world_points[:, 0], world_points[:, 1])
    world_alt = world_points[:, 2] if world_points.shape[1] >= 3 else np.zeros(world_points.shape[0])

    map_xyz = np.array(
        [
            map_interface.project_gps(float(lat), float(lon), float(alt))
            for lat, lon, alt in zip(world_lat, world_lon, world_alt)
        ],
        dtype=float,
    )
    if world_points.shape[1] > 3:
        map_points = np.concatenate([map_xyz, world_points[:, 3:]], axis=1)
    else:
        map_points = map_xyz

    ego_map_position = map_interface.project_gps(lat_pose, lon_pose, alt_pose)
    pose_matrix_map = pose_matrix_world.copy()
    pose_matrix_map[:3, 3] = ego_map_position

    ego_pose = EgoPose(
        position=pose_matrix_map[:3, 3],
        orientation_quat=matrix_to_quaternion(pose_matrix_map[:3, :3]),
        timestamp=None,
    )

    return PreparedEnvironment(
        sensor_points=sensor_points,
        ego_points=ego_points,
        world_points=world_points,
        map_points=map_points,
        ego_pose_world=pose_matrix_world,
        ego_pose_map=pose_matrix_map,
        ego_pose=ego_pose,
        map_interface=map_interface,
        origin_latlonalt=(origin_lat_val, origin_lon_val, origin_alt_val),
        origin_eastnorthalt=(origin_easting, origin_northing, origin_altitude),
        sample=sample,
    )


def add_dataset_arguments(parser):
    """Attach common dataset arguments to an argparse parser."""

    parser.add_argument("--sample", type=Path, required=True, help="Pickle file containing multi-sensor sample.")
    parser.add_argument("--calib-root", type=Path, required=True, help="Directory with calibration files.")
    parser.add_argument("--lanelet-map", type=Path, required=True, help="Lanelet2 OSM map.")
    parser.add_argument("--sensor", default=DEFAULT_SENSOR, help="Sensor key inside the sample pickle.")
    parser.add_argument("--pose-key", default=DEFAULT_POSE_KEY, help="Key for the ego pose 4x4 matrix in the sample.")
    parser.add_argument(
        "--src-epsg",
        default=DEFAULT_SRC_EPSG,
        help="EPSG code describing the world frame used in the sample (default: EPSG:25832).",
    )
    parser.add_argument("--origin-lat", type=float, help="Optional override for Lanelet map origin latitude.")
    parser.add_argument("--origin-lon", type=float, help="Optional override for Lanelet map origin longitude.")
    parser.add_argument("--origin-alt", type=float, help="Optional override for Lanelet map origin altitude.")
    return parser
