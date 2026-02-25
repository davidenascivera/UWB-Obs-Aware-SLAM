from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np


class RVIZ_Visualizer:
    def __init__(
        self,
        node,
        marker_topic: str = "/visualization_marker",
        frame_id: str = "uwb_ekf",
        traj_max_len: int = 2000,
    ):
        """
        node: istanza di rclpy.node.Node (es. self dentro Position_EKF)
        anchors_fix: array (N x 3) o (N x 4) con [x, y, z, ...]
        """

        self.node = node
        self.frame_id = frame_id
        self.traj_max_len = traj_max_len

        # publisher per i marker (QoS compatibile con RViz)
        marker_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.marker_pub = node.create_publisher(Marker, marker_topic, marker_qos)

        # buffer per le traiettorie
        self.traj_gt_points: list[Point] = []
        self.traj_ekf_points: list[Point] = []
        
        

    def publish_anchors(self, anchors: np.ndarray, kind: str = "fix"):
        """
        anchors: array Nx3 oppure Nx4 (x,y,z[,id])
        kind: stringa per scegliere namespace e colore ("fix" o "slam")
        """
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()

        marker.ns = f"anchors_{kind}"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        # dimensione delle sfere
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # colore diverso a seconda del tipo
        if kind == "fix":
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif kind == "slam":
            marker.id = 0
            marker.color.r = 0.6
            marker.color.g = 0.6
            marker.color.b = 0.6
        elif kind == "estimated_slam":
            marker.id = 1
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:  # default neutro
            marker.color.r = 0.2
            marker.color.g = 0.2
            marker.color.b = 1.0
        marker.color.a = 1.0

        # riempi i punti
        marker.points = []
        for anc in anchors:
            p = Point()
            p.x = float(anc[0])
            p.y = float(anc[1])
            p.z = float(anc[2])
            marker.points.append(p)

        self.marker_pub.publish(marker)

       
    # ------------------------------------------------------------------
    # Tracce GT vs EKF
    # ------------------------------------------------------------------
    def update_traj(self, gt_pos, ekf_pos):
        """
        gt_pos: iterabile (x,y,z) ground truth
        ekf_pos: iterabile (x,y,z) stima EKF
        frame_id: frame in cui vuoi visualizzare le traiettorie (es. msg.header.frame_id)
        """

        # GT
        pt_gt = Point()
        pt_gt.x = float(gt_pos[0])
        pt_gt.y = float(gt_pos[1])
        pt_gt.z = float(gt_pos[2])
        self.traj_gt_points.append(pt_gt)

        # EKF
        pt_ekf = Point()
        pt_ekf.x = float(ekf_pos[0])
        pt_ekf.y = float(ekf_pos[1])
        pt_ekf.z = float(ekf_pos[2])
        self.traj_ekf_points.append(pt_ekf)

        # limita la lunghezza per non esplodere
        if len(self.traj_gt_points) > self.traj_max_len:
            self.traj_gt_points.pop(0)
        if len(self.traj_ekf_points) > self.traj_max_len:
            self.traj_ekf_points.pop(0)

        # pubblica i marker delle due traiettorie
        self._publish_traj_markers()

    def _publish_traj_markers(self):
        now = self.node.get_clock().now().to_msg()

        # linea GT (verde)
        m_gt = Marker()
        m_gt.header.frame_id = self.frame_id
        m_gt.header.stamp = now
        m_gt.ns = "traj"
        m_gt.id = 0
        m_gt.type = Marker.LINE_STRIP
        m_gt.action = Marker.ADD

        m_gt.pose.orientation.w = 1.0
        m_gt.scale.x = 0.03

        m_gt.color.r = 0.0
        m_gt.color.g = 1.0
        m_gt.color.b = 0.0
        m_gt.color.a = 1.0

        m_gt.points = self.traj_gt_points

        # linea EKF (blu)
        m_ekf = Marker()
        m_ekf.header.frame_id = self.frame_id
        m_ekf.header.stamp = now
        m_ekf.ns = "traj"
        m_ekf.id = 1
        m_ekf.type = Marker.LINE_STRIP
        m_ekf.action = Marker.ADD

        m_ekf.pose.orientation.w = 1.0
        m_ekf.scale.x = 0.03

        m_ekf.color.r = 1.0
        m_ekf.color.g = 0.5
        m_ekf.color.b = 0.0
        m_ekf.color.a = 1.0


        m_ekf.points = self.traj_ekf_points

        self.marker_pub.publish(m_gt)
        self.marker_pub.publish(m_ekf)

    def _clear_timer_cb(self):
        self.clear_markers()
        self._clear_count += 1
        # dopo N volte smetti
        if self._clear_count >= 1:
            self._clear_timer.cancel()

    def clear_markers(self):
        msg = Marker()
        msg.action = Marker.DELETEALL
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self.marker_pub.publish(msg)
        
        
    def publish_direction_arrow(
        self,
        pos_xyz,
        dir_xy,
        length: float = 1.0,
        ns: str = "apf_dir",
        marker_id: int = 0,
    ):
        """
        Disegna una freccia in 2D usando:
          pos_xyz = (x, y, z) posizione attuale del robot
          dir_xy  = (dx, dy) direzione APF in 2D
          length  = lunghezza visuale della freccia
        La quota z viene presa da pos_xyz.
        """

        x0 = float(pos_xyz[0])
        y0 = float(pos_xyz[1])
        z0 = float(pos_xyz[2])  

        dx = float(dir_xy[0])
        dy = float(dir_xy[1])

        # normalizzazione
        n = (dx * dx + dy * dy) ** 0.5
        if n < 1e-6:
            return

        dx /= n
        dy /= n

        # punto di arrivo (head)
        x1 = x0 + length * dx
        y1 = y0 + length * dy
        z1 = z0

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()

        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        # shaft diameter, head diameter, head length
        marker.scale.x = 0.05
        marker.scale.y = 0.10
        marker.scale.z = 0.15

        # colore (magenta)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        p0 = Point()
        p0.x = x0
        p0.y = y0
        p0.z = z0

        p1 = Point()
        p1.x = x1
        p1.y = y1
        p1.z = z1

        marker.points = [p0, p1]
        self.marker_pub.publish(marker)
