"""Asynchronous CycloneDDS receiver for external FAST-LIO odometry.

get_latest() returns a lock-protected snapshot with source and receive timestamps.
Localization poses receive an Rx(pi) correction; twist velocities retain the
sender's LiDAR-local convention and require a matching deployment frame setting.
"""

import threading
import time
import numpy as np
from dataclasses import dataclass

from cyclonedds.core import Listener
from cyclonedds.internal import InvalidSample
from cyclonedds.domain import DomainParticipant
from cyclonedds.sub import Subscriber, DataReader
from cyclonedds.topic import Topic
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float64, int32, uint32
from cyclonedds.qos import Qos, Policy


# =================================================================
#  ROS2 nav_msgs/Odometry DDS IDL structure definitions
# =================================================================


@dataclass
class Time(IdlStruct, typename="builtin_interfaces::msg::dds_::Time_"):
    sec: int32
    nanosec: uint32


@dataclass
class Header(IdlStruct, typename="std_msgs::msg::dds_::Header_"):
    stamp: Time
    frame_id: str


@dataclass
class Vector3(IdlStruct, typename="geometry_msgs::msg::dds_::Vector3_"):
    x: float64
    y: float64
    z: float64


@dataclass
class Point(IdlStruct, typename="geometry_msgs::msg::dds_::Point_"):
    x: float64
    y: float64
    z: float64


@dataclass
class Quaternion(IdlStruct, typename="geometry_msgs::msg::dds_::Quaternion_"):
    x: float64
    y: float64
    z: float64
    w: float64


@dataclass
class Pose(IdlStruct, typename="geometry_msgs::msg::dds_::Pose_"):
    position: Point
    orientation: Quaternion


@dataclass
class PoseWithCovariance(IdlStruct, typename="geometry_msgs::msg::dds_::PoseWithCovariance_"):
    pose: Pose
    covariance: array[float64, 36]


@dataclass
class Twist(IdlStruct, typename="geometry_msgs::msg::dds_::Twist_"):
    linear: Vector3
    angular: Vector3


@dataclass
class TwistWithCovariance(IdlStruct, typename="geometry_msgs::msg::dds_::TwistWithCovariance_"):
    twist: Twist
    covariance: array[float64, 36]


@dataclass
class Odometry(IdlStruct, typename="nav_msgs::msg::dds_::Odometry_"):
    header: Header
    child_frame_id: str
    pose: PoseWithCovariance
    twist: TwistWithCovariance


def _localization_rx180(
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Rx(pi) to the received localization pose, leaving twist untouched."""
    x, y, z = np.asarray(position, dtype=np.float64)
    qx, qy, qz, qw = np.asarray(quaternion_xyzw, dtype=np.float64)
    return (
        np.array([x, -y, -z], dtype=np.float64),
        np.array([qw, -qz, qy, -qx], dtype=np.float64),
    )


# =================================================================
#  FastLIO Receiver
# =================================================================


class FastLIOReceiver:
    """
    FastLIO odometry DDS receiver.

    Subscribes to FastLIO nav_msgs/Odometry topics through CycloneDDS.
    Uses a DataReader listener to receive data asynchronously in the background.

    Attributes:
        connected (bool): Whether at least one message has been received.
        msg_count (int): Total number of received messages.
    """

    def __init__(
        self,
        topic_name: str = "rt/Odometry_imu_local",
        domain_id: int = 0,
        callback=None,
    ):
        """
        Args:
            topic_name: DDS topic name (ROS2 topic /xxx maps to DDS topic rt/xxx).
            domain_id: DDS Domain ID
            callback: Optional callback(v_local, omega_local), invoked on the
                listener thread when new data is received.
        """
        self._callback = callback

        self._lock = threading.Lock()
        self._latest = None
        self._latest_parsed = None
        self.connected = False
        self.msg_count = 0
        self.topic_name = topic_name

        self._dp = DomainParticipant(domain_id)
        self._sub = Subscriber(self._dp)
        topic = Topic(
            self._dp,
            topic_name,
            Odometry,
            qos=Qos(Policy.Reliability.Reliable(max_blocking_time=1), Policy.History.KeepLast(100)),
        )

        # Create a listener and bind the on_data_available callback.
        self._listener = Listener(on_data_available=self._on_data_available)
        self._reader = DataReader(self._sub, topic, listener=self._listener)

        print(f"[FastLIOReceiver] Subscribed to topic: {topic_name} (domain={domain_id}, async)")

    # ------------------------------------------------------------------
    #  Asynchronous callback (listener thread)
    # ------------------------------------------------------------------

    def _on_data_available(self, reader: DataReader):
        """CycloneDDS listener callback invoked on a background thread."""
        samples = [sample for sample in reader.take(N=32) if not isinstance(sample, InvalidSample)]
        if not samples:
            return

        latest_sample = samples[-1]
        parsed = self._parse_sample(latest_sample)

        with self._lock:
            if self._latest_parsed and parsed["timestamp"] <= self._latest_parsed["timestamp"]:
                return
            parsed["received_monotonic"] = time.monotonic()
            self._latest = latest_sample
            self._latest_parsed = parsed

        self.msg_count += len(samples)
        if not self.connected:
            self.connected = True
            print(f"[FastLIOReceiver] ✅ Data received ({self.topic_name})")

        if self._callback is not None:
            self._callback(parsed["v_local"], parsed["omega_local"])

    def get_latest(self) -> dict | None:
        """
        Return the latest odometry data, including all fields.

        Returns:
            dict with keys:
                'timestamp'    : float   — ROS timestamp (seconds)
                'received_monotonic': float — local monotonic receive time (seconds)
                    for timeout detection
                'position'     : (3,) np — position [x, y, z] (world frame)
                'quaternion'   : (4,) np — orientation [x, y, z, w] (world frame)
                'v_local'      : (3,) np — linear velocity (LiDAR local frame)
                'omega_local'  : (3,) np — angular velocity (LiDAR local frame)
            Returns None if no data is available.
        """
        with self._lock:
            return (
                {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in self._latest_parsed.items()
                }
                if self._latest_parsed
                else None
            )

    # ------------------------------------------------------------------
    #  Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sample(sample) -> dict:
        """Parse a DDS Odometry sample into a dictionary."""
        t = sample.header.stamp.sec + sample.header.stamp.nanosec * 1e-9
        p = sample.pose.pose.position
        q = sample.pose.pose.orientation
        v = sample.twist.twist.linear
        w = sample.twist.twist.angular
        position, quaternion = _localization_rx180(
            np.array([p.x, p.y, p.z], dtype=np.float64),
            np.array([q.x, q.y, q.z, q.w], dtype=np.float64),
        )

        return {
            "timestamp": t,
            "position": position,
            "quaternion": quaternion,
            "v_local": np.array([v.x, v.y, v.z], dtype=np.float64),
            "omega_local": np.array([w.x, w.y, w.z], dtype=np.float64),
        }
