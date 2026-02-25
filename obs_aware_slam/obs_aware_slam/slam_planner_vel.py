#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry  # Necessario per leggere Gazebo
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand

import math 
import time

# --- COSTANTI CONTROLLORE ---
DISTANCE_TH = 0.20      # Soglia arrivo waypoint [m]
CTRL_DT     = 0.05      # 20 Hz loop
Kp_xy       = 0.8       # Guadagno P Orizzontale
Kp_z        = 1.0       # Guadagno P Verticale
V_MAX       = 1.5       # Max velocità orizzontale [m/s]
VZ_MAX      = 1.0       # Max velocità verticale [m/s]

class ArmAndTakeoff(Node):
    def __init__(self):
        super().__init__('arm_and_takeoff')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        
        # Subscriber Odometria Gazebo (ENU) -> Sostituisce VehicleLocalPosition
        self.sub_est_ekf = self.create_subscription(
            Odometry,
            '/uwb_ekf/odometry',
            self.callback_ekf,
            qos_profile
        )


        # Subscriber RP (Exploration)
        self.rp_waypoint_sub = self.create_subscription(
            TrajectorySetpoint,
            '/rp/waypoint',
            self.rp_waypoint_callback,
            qos_profile
        )

        # Waypoints missione (Calibration) - Frame NED
        self.waypoints = [
            {"y":  0.0000, "x":  0.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y": -0.4330, "x":  1.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  0.4330, "x":  2.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  0.0000, "x":  3.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y": -0.4330, "x":  4.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  0.4330, "x":  5.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  0.0000, "x":  6.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y": -0.4330, "x":  7.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  0.4330, "x":  8.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  0.0000, "x":  9.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  1.0000, "x":  9.5000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  2.0000, "x":  9.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  3.0000, "x":  8.5000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  4.0000, "x":  9.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  5.0000, "x":  9.5000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  6.0000, "x":  9.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  7.0000, "x":  8.5000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  8.0000, "x":  9.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  8.4330, "x":  8.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  7.5670, "x":  7.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  8.0000, "x":  6.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  8.4330, "x":  5.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  7.5670, "x":  4.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  8.0000, "x":  3.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  8.4330, "x":  2.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  7.5670, "x":  1.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
            {"y":  8.0000, "x":  0.0000, "z": -3.0, "yaw": 0.0, "useDegrees": True},
        ]


        # Stato interno
        self.current_wp_idx = 0
        self.controller_mode = 'Calibration'  # 'Calibration' -> 'Exploration'
        self.have_estimate = False
        self.p_est = [0.0, 0.0, 0.0]  # [North, East, Down]
        
        # Target corrente per la modalità Exploration
        self.exploration_target = {"x": 0.0, "y": 0.0, "z": -3.0, "yaw": 0.0, "useDegrees": False}
        
        # Timers
        self.create_timer(0.1, self.publish_offboard_mode)        # 10 Hz Keep Alive
        self.create_timer(CTRL_DT, self.control_loop)             # 20 Hz Control Loop
        self.arm_timer = self.create_timer(2.0, self.arm)         # Arm una tantum
        self.offboard_timer = self.create_timer(3.0, self.set_offboard_mode) # Offboard una tantum

    def now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    # --- CALLBACK ODOMETRIA (ENU -> NED) ---
    def callback_ekf(self, msg: Odometry):
        # Gazebo pubblica ENU. PX4 vuole NED.
        # X_ned = Y_enu
        # Y_ned = X_enu
        # Z_ned = -Z_enu
        
        x_enu = msg.pose.pose.position.x
        y_enu = msg.pose.pose.position.y
        z_enu = msg.pose.pose.position.z

        self.p_est[0] = y_enu
        self.p_est[1] = x_enu
        self.p_est[2] = -z_enu
        
        if not self.have_estimate:
            self.get_logger().info(f"First position estimate received: ENU=({x_enu:.3f}, {y_enu:.3f}, {z_enu:.3f}) -> NED=({self.p_est[0]:.3f}, {self.p_est[1]:.3f}, {self.p_est[2]:.3f})")
        
        self.have_estimate = True

    # --- CALLBACK RP (EXPLORATION) ---
    def rp_waypoint_callback(self, msg: TrajectorySetpoint):
        if self.controller_mode == 'Exploration':
            # Aggiorniamo il target. Il control_loop si occuperà di inseguirlo.
            self.exploration_target = {
                "x": msg.position[0],
                "y": msg.position[1],
                "z": msg.position[2],
                "yaw": msg.yaw,
                "useDegrees": False # RP invia verosimilmente in radianti
            }
            # Logging ridotto per non intasare la console
            # self.get_logger().info(f"RP Update: {self.exploration_target['x']:.2f}, {self.exploration_target['y']:.2f}")

    # --- CICLO DI CONTROLLO PRINCIPALE ---
    def control_loop(self):
        if not self.have_estimate:
            print(f"Waiting for position estimate...")
            return

        # 1. Determina il target attuale
        if self.controller_mode == 'Calibration':
            target_wp = self.waypoints[self.current_wp_idx]
            # Debug: stampa ogni 50 cicli (~2.5s)
            if not hasattr(self, '_debug_counter'):
                self._debug_counter = 0
            self._debug_counter += 1
            if self._debug_counter % 50 == 0:
                self.get_logger().info(f"WP{self.current_wp_idx}: target=({target_wp['x']:.2f},{target_wp['y']:.2f},{target_wp['z']:.2f}) p_est=({self.p_est[0]:.2f},{self.p_est[1]:.2f},{self.p_est[2]:.2f}) dist={math.sqrt((target_wp['x']-self.p_est[0])**2+(target_wp['y']-self.p_est[1])**2+(target_wp['z']-self.p_est[2])**2):.2f}m")
        else:
            target_wp = self.exploration_target

        # 2. Calcola Errore e Distanza
        ex = target_wp["x"] - self.p_est[0]
        ey = target_wp["y"] - self.p_est[1]
        ez = target_wp["z"] - self.p_est[2]
        distance = math.sqrt(ex**2 + ey**2 + ez**2)

        # 3. Gestione stati (Solo in modalità Calibration)
        if self.controller_mode == 'Calibration' and distance < DISTANCE_TH:
            if self.current_wp_idx < len(self.waypoints) - 1:
                self.get_logger().info(f"--- WP {self.current_wp_idx} raggiunto. Attesa... ---")
                time.sleep(0.5) # Piccolo stop
                self.current_wp_idx += 1
                self.get_logger().info(f"Going to WP {self.current_wp_idx}")
                # Reset errore per evitare scatti immediati nel prossimo loop
                return 
            else:
                self.get_logger().info("Calibration finita. Passo a EXPLORATION.")
                self.controller_mode = 'Exploration'
                # Imposta il target iniziale di esplorazione all'ultimo punto conosciuto per sicurezza
                self.exploration_target = target_wp

        # 4. Calcolo Velocità (P-Controller)
        vx_sp = Kp_xy * ex
        vy_sp = Kp_xy * ey
        vz_sp = Kp_z  * ez

        # 5. Saturazione Velocità
        vxy_norm = math.sqrt(vx_sp**2 + vy_sp**2)
        if vxy_norm > V_MAX:
            scale = V_MAX / vxy_norm
            vx_sp *= scale
            vy_sp *= scale
        
        if vz_sp > VZ_MAX: vz_sp = VZ_MAX
        if vz_sp < -VZ_MAX: vz_sp = -VZ_MAX

        # 6. Gestione Yaw
        target_yaw = target_wp["yaw"]
        if target_wp.get("useDegrees", False):
            target_yaw = target_yaw * (math.pi / 180.0)
        
        # 7. Pubblica Setpoint
        self.publish_velocity_setpoint(vx_sp, vy_sp, vz_sp, target_yaw)

    def publish_velocity_setpoint(self, vx, vy, vz, yaw):
        sp = TrajectorySetpoint()
        sp.timestamp = self.now_us()
        
        # NaN su Position -> PX4 usa Velocity
        sp.position = [float('nan'), float('nan'), float('nan')]
        sp.velocity = [float(vx), float(vy), float(vz)]
        sp.yaw      = float(yaw)
        
        self.setpoint_pub.publish(sp)

    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.now_us()
        msg.position = False     # DISABILITA controllo posizione
        msg.velocity = True      # ABILITA controllo velocità
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def arm(self):
        msg = VehicleCommand()
        msg.timestamp = self.now_us()  
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.get_logger().info("Arm command sent")
        self.arm_timer.cancel()

    def set_offboard_mode(self):
        msg = VehicleCommand()
        msg.timestamp = self.now_us()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0
        msg.param2 = 6.0
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.get_logger().info("Set OFFBOARD mode command sent")
        self.offboard_timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = ArmAndTakeoff()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()