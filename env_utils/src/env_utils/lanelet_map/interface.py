"""Lanelet2 map interface utilities.

The interface hides the Lanelet2 python bindings behind a light-weight façade that
provides query helpers commonly required in environment modelling labs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import warnings

import numpy as np


@dataclass
class EgoPose:
    """Minimal ego pose container."""

    position: np.ndarray  # shape: (3,)
    orientation_quat: np.ndarray  # (w, x, y, z)
    timestamp: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EgoPose":
        pos = np.asarray(data.get("position", [0.0, 0.0, 0.0]), dtype=float)
        ori = np.asarray(data.get("orientation_quat", [1.0, 0.0, 0.0, 0.0]), dtype=float)
        stamp = data.get("timestamp")
        return cls(position=pos, orientation_quat=ori, timestamp=stamp)


@dataclass
class LaneletQueryResult:
    """Query payload returned by :class:`LaneletMapInterface`."""

    ego_lanelet_id: Optional[int]
    nearby_lanelet_ids: List[int]
    centerline: np.ndarray  # shape: (N, 2)
    left_boundary: np.ndarray  # shape: (N, 2)
    right_boundary: np.ndarray  # shape: (N, 2)


_ROUTING_UNAVAILABLE = object()


class LaneletMapInterface:
    """Query-focused wrapper around the Lanelet2 python bindings.

    Parameters
    ----------
    map_path:
        Path to an `.osm` Lanelet2 map.
    projector:
        Projection to use. Supports `"utm"` (default) and `"mercator"`.
    origin:
        Optional origin tuple `(lat, lon, alt)` for the projector.
    """

    def __init__(
        self,
        map_path: str | Path,
        *,
        projector: str = "utm",
        origin: tuple[float, float, float] | None = None,
    ) -> None:
        try:
            import lanelet2
            from lanelet2.io import Origin, load
            from lanelet2.projection import MercatorProjector, UtmProjector
        except ImportError as exc:
            raise RuntimeError(
                "lanelet2 python bindings are required. Install via `pip install lanelet2`."
            ) from exc

        self._lanelet2 = lanelet2
        map_path = Path(map_path)
        if not map_path.exists():
            raise FileNotFoundError(map_path)

        origin = origin or (0.0, 0.0, 0.0)
        self._origin = origin
        projector = projector.lower()
        if projector not in {"utm", "mercator"}:
            raise ValueError(f"Unsupported projector '{projector}'. Use 'utm' or 'mercator'.")

        origin_obj = Origin(*origin)
        if projector == "utm":
            self._projector = UtmProjector(origin_obj)
        else:
            self._projector = MercatorProjector(origin_obj)

        self._map = load(str(map_path), self._projector)
        # Routing graph is optional; create lazily when needed.
        self._routing_graph = None

    @property
    def lanelet_map(self):
        """Return the underlying Lanelet2 map."""

        return self._map

    @property
    def origin(self) -> tuple[float, float, float]:
        """Return the map origin used for projection."""

        return self._origin

    @property
    def projector(self):
        """Expose the projection used to load the map."""

        return self._projector

    def routing_graph(self):
        """Return a cached Lanelet2 routing graph."""

        return self._ensure_routing_graph()

    def _ensure_routing_graph(self):
        if self._routing_graph is _ROUTING_UNAVAILABLE:
            return None
        if self._routing_graph is None:
            from lanelet2.routing import RoutingGraph
            import lanelet2.traffic_rules as tr

            locations = getattr(tr, "Locations", getattr(tr, "Location", None))
            participants = getattr(tr, "Participants", getattr(tr, "Participant", None))
            if locations is None or participants is None:
                raise RuntimeError("lanelet2 traffic rules API missing Locations/Participants enums.")

            germany = getattr(locations, "Germany", None)
            vehicle = getattr(participants, "Vehicle", None)
            if germany is None or vehicle is None:
                raise RuntimeError("lanelet2 traffic rules enums lack Germany/Vehicle definitions.")

            traffic_rules = None
            if hasattr(tr, "TrafficRulesFactory"):
                traffic_rules = tr.TrafficRulesFactory.create(germany, vehicle)
            else:
                creator = getattr(tr, "createTrafficRules", None)
                if callable(creator):
                    try:
                        traffic_rules = creator(germany, vehicle)
                    except TypeError as exc:
                        raise RuntimeError(
                            "lanelet2.createTrafficRules signature not supported; please update env_utils."
                        ) from exc

            if traffic_rules is not None:
                try:
                    self._routing_graph = RoutingGraph(self._map, traffic_rules)
                except TypeError:
                    self._routing_graph = RoutingGraph(self._map, traffic_rules, vehicle)
            else:
                warnings.warn(
                    "lanelet2 traffic rules could not be instantiated; routing graph unavailable",
                    RuntimeWarning,
                )
                self._routing_graph = _ROUTING_UNAVAILABLE
        return None if self._routing_graph is _ROUTING_UNAVAILABLE else self._routing_graph

    def query_lanelets(
        self,
        ego_pose: EgoPose,
        *,
        radius: float = 50.0,
        max_neighbors: int = 5,
    ) -> LaneletQueryResult:
        """Return the lanelets in the vicinity of the ego pose."""

        from lanelet2.core import BasicPoint2d
        from lanelet2.geometry import findNearest

        xy = ego_pose.position[:2]
        search_point = BasicPoint2d(float(xy[0]), float(xy[1]))
        nearest = findNearest(self._map.laneletLayer, search_point, max_neighbors)

        nearby_lanelets = []
        for distance, lanelet in nearest:
            if radius is not None and distance > radius:
                continue
            nearby_lanelets.append(lanelet)

        ego_lanelet = nearby_lanelets[0] if nearby_lanelets else None
        def poly_or_empty(ll, attr: str) -> np.ndarray:
            if ll is None:
                return np.empty((0, 2), dtype=float)
            linestring = self._get_linestring(ll, attr)
            return self._linestring_to_numpy(linestring)

        return LaneletQueryResult(
            ego_lanelet_id=ego_lanelet.id if ego_lanelet is not None else None,
            nearby_lanelet_ids=[lanelet.id for lanelet in nearby_lanelets],
            centerline=poly_or_empty(ego_lanelet, "centerline"),
            left_boundary=poly_or_empty(ego_lanelet, "leftBound"),
            right_boundary=poly_or_empty(ego_lanelet, "rightBound"),
        )

    def sample_route(
        self,
        ego_pose: EgoPose,
        *,
        downtrack: float = 60.0,
    ) -> Iterable[np.ndarray]:
        """Yield centerline points along a routed path starting from the ego pose lanelet."""

        query = self.query_lanelets(ego_pose, radius=5.0, max_neighbors=1)
        if query.ego_lanelet_id is None:
            return []

        routing_graph = self.routing_graph()
        if routing_graph is None:
            return [query.centerline] if len(query.centerline) else []

        lanelet = self._map.laneletLayer[query.ego_lanelet_id]
        route = routing_graph.getRoute(lanelet)
        if route is None:
            return [query.centerline]

        path = route.fullPath()
        accumulated = 0.0
        sampled: List[np.ndarray] = []
        for ll in path:
            centerline = self._linestring_to_numpy(self._get_linestring(ll, "centerline"))
            if not len(centerline):
                continue
            sampled.append(centerline)
            accumulated += self._length(centerline)
            if downtrack is not None and accumulated >= downtrack:
                break
        return sampled

    def project_gps(self, lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
        """Project a GPS coordinate to the map frame."""

        gps_point = self._lanelet2.core.GPSPoint(lat, lon, alt)
        projected = self._projector.forward(gps_point)
        return np.array([float(projected.x), float(projected.y), float(projected.z)], dtype=float)

    def plot_lanelet_neighborhood(
        self,
        ego_pose: EgoPose,
        *,
        radius: float = 50.0,
        ax=None,
        show: bool = True,
    ):
        """Visualize the ego lanelet neighborhood using matplotlib."""

        import matplotlib.pyplot as plt

        result = self.query_lanelets(ego_pose, radius=radius)
        ax = ax or plt.gca()

        # Plot all nearby lanelets
        for lanelet_id in result.nearby_lanelet_ids:
            lanelet = self._map.laneletLayer[lanelet_id]
            left = self._linestring_to_numpy(self._get_linestring(lanelet, "leftBound"))
            right = self._linestring_to_numpy(self._get_linestring(lanelet, "rightBound"))
            center = self._linestring_to_numpy(self._get_linestring(lanelet, "centerline"))
            if len(left):
                ax.plot(left[:, 0], left[:, 1], color="gray", linewidth=1.0)
            if len(right):
                ax.plot(right[:, 0], right[:, 1], color="gray", linewidth=1.0)
            if len(center):
                ax.plot(center[:, 0], center[:, 1], color="blue", linewidth=1.5, linestyle="--")

        # Ego pose marker
        ax.scatter([ego_pose.position[0]], [ego_pose.position[1]], color="red", s=50, label="ego")
        ax.set_aspect("equal")
        ax.set_title("Lanelet Neighborhood")
        ax.legend(loc="upper right")

        if show:
            plt.show()

        return ax

    @staticmethod
    def _linestring_to_numpy(linestring: Iterable[Any]) -> np.ndarray:
        points = [(float(pt.x), float(pt.y)) for pt in linestring]
        if not points:
            return np.empty((0, 2), dtype=float)
        return np.asarray(points, dtype=float)

    @staticmethod
    def _get_linestring(obj: Any, attr: str):
        element = getattr(obj, attr)
        return element() if callable(element) else element

    @staticmethod
    def _linestring_to_xyz(linestring: Iterable[Any]) -> np.ndarray:
        points = [(float(pt.x), float(pt.y), float(getattr(pt, "z", 0.0))) for pt in linestring]
        if not points:
            return np.empty((0, 3), dtype=float)
        return np.asarray(points, dtype=float)

    def extract_lanelet_geometry(
        self,
        lanelet,
        *,
        include_z: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return center, left, and right boundary polylines for a lanelet."""

        convert = self._linestring_to_xyz if include_z else self._linestring_to_numpy
        center = convert(self._get_linestring(lanelet, "centerline"))
        left = convert(self._get_linestring(lanelet, "leftBound"))
        right = convert(self._get_linestring(lanelet, "rightBound"))
        return center, left, right

    @staticmethod
    def _length(polyline: np.ndarray) -> float:
        if len(polyline) < 2:
            return 0.0
        diffs = np.diff(polyline, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
