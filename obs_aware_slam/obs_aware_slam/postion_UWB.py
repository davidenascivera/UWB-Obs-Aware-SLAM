import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from px4_msgs.msg import SensorCombined
from px4_msgs.msg import TrajectorySetpoint
from std_msgs.msg import Float32MultiArray

from obs_aware_slam.utils import Rbody_to_ENU, is_far_enough
from obs_aware_slam.ekf_uwb import EKF_UWB
from obs_aware_slam.wls_est import WLS_Est
from obs_aware_slam.apf_class import APF_Planner
from obs_aware_slam.RVIZ_visualizer import RVIZ_Visualizer


class Position_EKF(Node):
    def __init__(self):
        super().__init__('position_ekf')
        # Configure QoS profile for PX4 compatibility
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        qos_setpoint = QoSProfile(depth=1)
        qos_setpoint.reliability = ReliabilityPolicy.RELIABLE
        qos_setpoint.durability  = DurabilityPolicy.VOLATILE

        self.gt_sub = self.create_subscription(
            Odometry,
            '/model/x500_0/odometry',
            self.odometry_callback,
            qos_profile
        )
        self.imu_sub = self.create_subscription(
            SensorCombined,
            '/fmu/out/sensor_combined',
            self.imu_callback,
            qos_profile=qos_profile
        )

        self.uwb_pub = self.create_publisher(Float32MultiArray, '/uwb/ranges', qos_profile=qos_profile)
        self.pos_ekf_pub = self.create_publisher(Odometry, '/uwb_ekf/odometry', qos_profile=qos_profile)
        self.apf_waypoint_pub = self.create_publisher(TrajectorySetpoint, '/apf/waypoint', qos_profile=qos_setpoint)
        self.uwb_noise_std = 0.15  # Standard deviation of UWB measurement noise
        
        # TODO Le ancore nella quarta colonna hanno il loro ID. Successivamente verrà automatizzato, ma per visualizzarlo meglio così
        self.Anchors_fix = np.array([
            [-1.0, -1.0, 1.0, 100],
            [-1.0,  1.0, 3.0, 101],
            [ 1.0,  1.0, 1.0, 102],
            [ 1.0, -1.0, 2.0, 103],
            # [ 5.0,  10.0, 1.0, 104]
        ])
        
        # Get Monte Carlo seed parameter
        self.declare_parameter('mc_seed', 0)
        self.declare_parameter('x0', 0.0)
        self.declare_parameter('y0', 0.0)

        seed = self.get_parameter('mc_seed').value
        self.x0 = self.get_parameter('x0').value
        self.y0 = self.get_parameter('y0').value
        self.mission_completed = False
        self.start_time = self.get_clock().now()

        np.random.seed(seed)

        self.get_logger().info(
            f"MC seed={seed}, x0={self.x0:.2f}, y0={self.y0:.2f}"
        )
        
        
        # Create directory for Monte Carlo run (use absolute path)
        base_dir = os.path.join(os.path.expanduser("~"), "UV_proj", "mc_results")
        os.makedirs(base_dir, exist_ok=True)
        self.csv_path = os.path.join(base_dir, f"data_run_{seed:03d}.csv")
        self.get_logger().info(f"Saving results to: {base_dir}")

        

        
        self.sensor_z_height_m = 1.0  # Altezza del sensore UWB dal suolo 
        n_anchors = 11
        x_min, x_max = -2.0, 9.0
        y_min, y_max = -2.0, 11.0 
        min_dist = 2.0
        anchors = []
        
        while len(anchors) < n_anchors:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            candidate = np.array([x, y])
            if is_far_enough(candidate, anchors, min_dist):
                anchors.append(candidate)

        anchors = np.array(anchors)
        ids = np.arange(n_anchors)

        self.Anchors_slam = np.column_stack([
            anchors[:, 0],
            anchors[:, 1],
            np.ones(n_anchors) * self.sensor_z_height_m,
            ids
        ])
        
        # self.Anchors_slam = np.array([
        #     [ 2.0, 11.0, self.sensor_z_height_m, 0],
        #     [ 8.0, 11.0, self.sensor_z_height_m, 1],
        #     [ 8.0,  4.0, self.sensor_z_height_m, 2],
        #     [ 1.0,  4.0, self.sensor_z_height_m, 3],
        #     [-1.0,  7.0, self.sensor_z_height_m, 4],
        #     [-2.0,  2.0, self.sensor_z_height_m, 5],
        #     [-4.0,  6.0, self.sensor_z_height_m, 6],
        #     [ 6.0,  8.0, self.sensor_z_height_m, 7],
        #     [ -2.0,  10.0, self.sensor_z_height_m, 8]
        # ])
        
        #----------------------------------------------------
        # 1. Prepariamo i metadati (seed, x0, y0)
        metadata = [seed, self.x0, self.y0]

        # 2. Estraiamo le coordinate x, y delle ancore e le "appiattiamo" (flatten)
        # self.Anchors_slam ha colonne: [x, y, z, id]
        # Prendiamo solo le prime due colonne e le trasformiamo in una lista singola
        anchor_coords = self.Anchors_slam[:, :2].flatten().tolist()

        # 3. Uniamo tutto in un'unica riga
        config_row = metadata + anchor_coords

        # 4. Creiamo un header dinamico per chiarezza (es. seed, x0, y0, ancx0, ancy0, ...)
        n_anchors = self.Anchors_slam.shape[0]
        header_parts = ["seed", "posx", "posy"]
        for i in range(n_anchors):
            header_parts.append(f"ancx{i+1}")
            header_parts.append(f"ancy{i+1}")
        header_string = ",".join(header_parts)

        # 5. Salvataggio su file
        config_path = os.path.join(base_dir, f"conf_{seed:03d}.csv")
        np.savetxt(config_path, [config_row], delimiter=",", header=header_string, comments='')

        self.get_logger().info(f"Configurazione scenario salvata in: {config_path}")
        # ----------------------------------------------------
        
        
        x0 = np.zeros((6, 1))  # Initial state [x, y, z, vx, vy, vz]
        P0 = np.eye(6) * 10.0
        sig_a_proc = 0.2  # Process noise standard deviation for acceleration
        sig_a_meas = 0.2  # Measurement noise standard deviation for acceleration
        self.EKF_uwb = EKF_UWB(x0 = x0, P= P0,  
                               sig_a_proc=sig_a_proc, sig_a_meas=sig_a_meas,
                               CA_dynamic=True,
                               use_acc_pseudomeas=True,
                               Anchors_fix=self.Anchors_fix,
                               uwb_noise_std = self.uwb_noise_std,
                               z_anc=self.sensor_z_height_m)  
        
        self.a_body = None
        self.a_stamp_us = None
        
        
        self.Est_dict = {}
        self.initialized_anc = set()
        self.n_anc_fix_seen = 0 #TODO can be deleted
        self.update_filters_counter = 0  # Counter for periodic reordering

        self.ApfNav = APF_Planner(smooth=0.5, max_turn_deg=45, Jratio_th=1.0) #smooth basso = più smooth
        self.x_goal_list = [np.array([0.0, 7.0]),
                            np.array([2.0, 9.0]),
                            np.array([7.0, 9.0]),
                            np.array([8.0, 7.0]),
                            np.array([8.0, 0.0])]
        
        self.goal_threshold = 1.0  # Distanza per considerare goal raggiunto
        self.current_goal_idx = 0  
        
        self.displacement_apf = 0.6
        self.last_dir = np.array([0.0, 0.0])
        self.timer_planner = self.create_timer(0.6, self.publish_planner_setpoint)
        self.timer_mission = self.create_timer(0.2, self.check_finished_mission)
        
        self.get_logger().info(f"est EKF CA_dyn = {self.EKF_uwb.use_CA}, acc_pseudomeas = {self.EKF_uwb.use_acc_pseudomeas} initialized.")
        
        # Enable/disable for publishing
        self.Rviz_viz = RVIZ_Visualizer(self, frame_id="map", traj_max_len=1000)
        self.last_est_pos = np.zeros(3) # è un (3,)
        self.last_gt_pos = np.zeros(3)
        self.estimated_anc_positions = []
        self.Rviz_viz.clear_markers()
        self.Rviz_viz.publish_anchors(self.Anchors_fix, kind="fix") 
        self.Rviz_viz.publish_anchors(self.Anchors_slam, kind="slam")
        self.get_logger().info("Nodes fixed published to RViz.")
        
        self.timer_rviz = self.create_timer(0.1, self.update_drones_rviz)   # 10 Hz
        
        self.DATA = []
        self._shutdown_requested = False
        

    def check_finished_mission(self):
        if self._shutdown_requested:
            return
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if self.mission_completed or elapsed > 250:
            self.get_logger().info("Run finished -> shutdown.")
            self._shutdown_requested = True
            # Salva i dati rimanenti prima di terminare
            self.save_remaining_data()
            # Cancel all timers before shutdown
            self.timer_planner.cancel()
            self.timer_mission.cancel()
            self.timer_rviz.cancel()
            # Trigger shutdown
            raise SystemExit(0)
    
    def save_remaining_data(self):
        """Salva tutti i dati rimanenti in self.DATA che non sono ancora stati scritti su file"""
        if len(self.DATA) > 0:
            # Calcola quante righe mancano da salvare (quelle non ancora scritte)
            # Dato che salviamo ogni 10, le righe non salvate sono quelle dopo l'ultimo multiplo di 10
            last_saved = (len(self.DATA) // 10) * 10
            remaining = self.DATA[last_saved:]
            
            if remaining:
                try:
                    with open(self.csv_path, "a") as f:
                        np.savetxt(f, remaining, delimiter=",")
                    self.get_logger().info(f"Salvate {len(remaining)} righe finali. Totale: {len(self.DATA)} righe.")
                except Exception as e:
                    self.get_logger().error(f"Errore nel salvataggio dati finali: {e}")
        
    def publish_planner_setpoint(self):
        pos_robot = self.EKF_uwb.x[:2, :].flatten()
        
        # Compute the next goal and distance to it
        current_goal = self.x_goal_list[self.current_goal_idx]
        dist_to_goal = np.linalg.norm(pos_robot - current_goal)
                
        if dist_to_goal < self.goal_threshold:
            if self.current_goal_idx < (len(self.x_goal_list) - 1):
                self.get_logger().info(f"Goal {self.current_goal_idx + 1} raggiunto! Switching to next goal.")
                self.current_goal_idx += 1
            else:
                self.get_logger().info(f"Missione Completata: {self.current_goal_idx + 1} m")
                self.mission_completed = True
                return  # Exit without publishing further waypoints
        
        current_goal = self.x_goal_list[self.current_goal_idx]
        
        self.last_dir = self.ApfNav.update_direction(self.Est_dict, goal_pos=current_goal, x_robot=pos_robot)
        if np.linalg.norm(self.last_dir) < 1e-3:
            return

        sp = TrajectorySetpoint()
        sp.timestamp = self.get_clock().now().nanoseconds // 1000

        # qui devi stare attento ai frame:
        # EKF lavora in ENU (x_est, y_est)
        # TrajectorySetpoint di PX4 è in NED (x_n, y_e, z_down)
        x_enu = self.last_est_pos[0]
        y_enu = self.last_est_pos[1]
        z_enu = self.last_est_pos[2]

        dir_xy = self.last_dir / np.linalg.norm(self.last_dir)
        wp_x_enu = x_enu + self.displacement_apf * dir_xy[0]
        wp_y_enu = y_enu + self.displacement_apf * dir_xy[1]

        # ENU -> NED
        x_ned = wp_y_enu
        y_ned = wp_x_enu
        z_ned = -z_enu

        sp.position[0] = float(x_ned)
        sp.position[1] = float(y_ned)
        sp.position[2] = -3

        # yaw dalla direzione (sempre in NED)
        dx_ned = dir_xy[1]
        dy_ned = dir_xy[0]
        # sp.yaw = float(np.arctan2(dy_ned, dx_ned))
        sp.yaw = 0.0
        # print(f"Publishing APF waypoint at NED ({sp.position[0]:.2f}, {sp.position[1]:.2f}, {sp.position[2]:.2f}) with yaw {sp.yaw:.2f} rad")
    
        self.apf_waypoint_pub.publish(sp)

        
    def update_drones_rviz(self):
        data_row = np.zeros(6 + self.Anchors_slam.shape[0]*2)  # 3+3 per gt_xyz e ekf_xyz
        # 1. Updating the traj plot
        self.Rviz_viz.update_traj(gt_pos=self.last_gt_pos, ekf_pos=self.last_est_pos)
        data_row[0:3] = self.last_gt_pos[0:3]  # gt_x, gt_y, gt_z
        data_row[3:6] = self.last_est_pos[0:3]  # ekf_x, ekf_y, ekf_z
        
        # # # 2. Updating the arrow for direction
        # pos_robot = self.EKF_uwb.x[:3, :].flatten()
        # pos_xy = pos_robot[:2]
        # direzione = self.ApfNav.update_direction(self.Est_dict, goal_pos=self.x_goal, x_robot=pos_xy)
        # # self.get_logger().info(f" APF direction (non-smooth): {direzione}")
        # self.Rviz_viz.publish_direction_arrow(pos_xyz=pos_robot, dir_xy=direzione, length=2.0)
        
        # 3. Updating estimated anchor positions
        Array_pos = []
        for anc_id in self.EKF_uwb.id_to_idx.keys():
            idx_start = self.EKF_uwb.id_to_idx[anc_id]
            pos_est = self.EKF_uwb.x[
                idx_start : idx_start + 2, 0]
            Array_pos.append(pos_est)
            inizio_stampa = 6 + int(anc_id)*2  # Ora inizia da 6 (dopo gt_xyz + ekf_xyz)
            data_row[inizio_stampa : inizio_stampa +2] = pos_est.flatten()

        self.DATA.append(data_row)
        # Save every 10 entries
        # Salvataggio incrementale ogni 10 campioni
        if len(self.DATA) % 10 == 0:
            try:
                with open(self.csv_path, "a") as f:
                    # Salviamo solo le ultime 10 righe aggiunte alla lista
                    np.savetxt(f, self.DATA[-10:], delimiter=",")
                # print(f"Log aggiornato: {len(self.DATA)} righe totali.")
            except Exception as e:
                self.get_logger().error(f"Errore nel salvataggio log: {e}")
        
        if Array_pos == []:
            return
        Array_pos = np.array(Array_pos)
        const_z_column = np.ones((Array_pos.shape[0], 1))*self.sensor_z_height_m
        Array_pos_complete = np.hstack([Array_pos, const_z_column])
        
        self.Rviz_viz.publish_anchors(Array_pos_complete, kind="estimated_slam") 
            
        #save data_row: [gt_x, gt_y, ekf_x, ekf_y, anc0_x, anc0_y, anc1_x, anc1_y, ...] to csv
        
        
        
    def odometry_callback(self, msg):
        px, py, pz = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z
        p_robot = np.array([px, py, pz])
        self.last_gt_pos = p_robot
        
        if self.a_body is None or self.a_stamp_us is None:
            self.get_logger().warn("No IMU data yet, skipping EKF step")
            return
    
        msg_uwb = Float32MultiArray()
        msg_uwb.data = []
        
        # Use odometry timestamp instead of IMU to avoid race conditions
        # Convert ROS2 timestamp to microseconds
        current_time_us = int(msg.header.stamp.sec * 1e6 + msg.header.stamp.nanosec / 1e3)
        
        # Compute the elapsed time
        if not hasattr(self, 'last_time'):
            self.last_time = current_time_us
            self.get_logger().info(f"EKF initialized with timestamp: {current_time_us} us")
            return

        # Calculate delta time
        elapsed_time = current_time_us - self.last_time
        
        # Skip if timestamp hasn't changed or went backwards (bag replay loop or timing issue)
        if elapsed_time <= 0:
            self.get_logger().debug(f"Skipping: dt={elapsed_time} us (current={current_time_us}, last={self.last_time})")
            return
            
        self.last_time = current_time_us
        
        meas_vec = []
        for i, anc in enumerate(self.Anchors_fix):
            distance = np.linalg.norm(p_robot - anc[:3])
            noisy_distance = distance + np.random.normal(0, self.uwb_noise_std)
            meas_vec.append(noisy_distance)   
        meas_vec = np.array(meas_vec)
        
        _, id_dist_fix = get_meas_seen(self.Anchors_fix, p_robot, 
                                                       distance_th=6, noise_uwb=self.uwb_noise_std)        
        _, id_dist_slam = get_meas_seen(self.Anchors_slam, p_robot, 
                                         distance_th=8, noise_uwb=self.uwb_noise_std)
        id_dist_meas = np.vstack([id_dist_fix, id_dist_slam])
        self.n_anc_fix_seen = id_dist_fix.shape[0]  #TODO can be deleted
        
        # Creiamo dei filtri WLS per ogni misura SLAM vista
        for id_slam in id_dist_slam:
            # Se id della misura non è nel dizionario, creiamo un nuovo stimatore
            id_anc = int(id_slam[0])
            if id_anc not in self.Est_dict and id_anc not in self.initialized_anc:
                self.Est_dict[id_anc] = WLS_Est(th_move_delta=0.05, counter_still_th= 7,
                                                    a_z_given=self.sensor_z_height_m, min_distance_meas=0.2)
                
        # print(f"The id being tracked are: {list(self.Est_dict.keys())}")
        # print(f"Meas slam: \n {meas_id_slam}")
        
        q = np.zeros((4, 1))
        q[0] = msg.pose.pose.orientation.w
        q[1] = msg.pose.pose.orientation.x
        q[2] = msg.pose.pose.orientation.y
        q[3] = msg.pose.pose.orientation.z
        # a_enu = Rbody_to_ENU(q) @ self.a_body
        a_body_flu = np.vstack([ self.a_body[0:1],
                                -self.a_body[1:2],
                                -self.a_body[2:3] ])
        a_enu = Rbody_to_ENU(q) @ a_body_flu       # rotate specific force
        a_enu += np.array([[0.0],[0.0],[-9.81]])  # add gravity in ENU
        # print(f"Acceleration in ENU frame:\n{a_enu.flatten()}")
        
        self.update_filters(dt_sec = elapsed_time / 1e6,
                            a_enu = a_enu,
                            id_dist_meas=id_dist_meas,
                            update_wls = id_dist_slam)
        
        odom_est = Odometry()
        odom_est.header.stamp = self.get_clock().now().to_msg()
        odom_est.header.frame_id = "uwb_ekf"
        odom_est.pose.pose.position.x = float(self.EKF_uwb.x[0,0])
        odom_est.pose.pose.position.y = float(self.EKF_uwb.x[1,0])
        odom_est.pose.pose.position.z = float(self.EKF_uwb.x[2,0])
        
        odom_est.pose.covariance[0] = float(self.EKF_uwb.P[0,0])
        odom_est.pose.covariance[7] = float(self.EKF_uwb.P[1,1])
        odom_est.pose.covariance[14] = float(self.EKF_uwb.P[2,2])
        
        self.pos_ekf_pub.publish(odom_est)
        
    def update_filters(self, dt_sec, a_enu, id_dist_meas, update_wls):
        '''
        Update both EKF filters
        '''
        # EKF: 1) Prediction
        self.EKF_uwb.prediction(dt = dt_sec,a = a_enu)
        # EKF: 2) Update
        self.EKF_uwb.update(a_enu=a_enu, id_dist_meas=id_dist_meas)
        self.last_est_pos = self.EKF_uwb.x[:3, :].flatten() 
        
        # WLS: Update the estimators in order of distance
        to_pop = []
        for id_meas in update_wls:
            id = id_meas[0]
            meas_dist = id_meas[1]
            if id in self.initialized_anc:
                continue
            if id in self.Est_dict:
                pos = self.EKF_uwb.pos
                Pxx, Pyy, Pxy = self.EKF_uwb.cov_2d
                
                self.Est_dict[id].update(pos, Pxx, Pyy, Pxy, meas_dist)
                
                if self.Est_dict[id].stable_flag is True: #TODO REPUT
                    pos_est = self.Est_dict[id].last_x.flatten()
                    error = np.linalg.norm(
                        pos_est - self.Anchors_slam[int(id), :2]
                    )
                    print(f"Initialized WLS Estimator for ID {id} at position {pos_est} with error {error:.3f} m")
                    self.EKF_uwb.add_anc_slam_to_state(
                        Anc_gt=None, anc_id=id, anc_x=pos_est[0], anc_y=pos_est[1]
                    )
                    self.initialized_anc.add(id)
                    to_pop.append(id)
                    
            else:
                self.get_logger().warn(f"ID {id} not found in Est_dict during update.")
                print(f"ID {id} not found in Est_dict during update.")
        
        for id in to_pop:
            self.Est_dict.pop(id)
        
        # Riordina Est_dict per distanza ogni 10 cicli (riduce overhead CPU)
        self.update_filters_counter += 1
        if self.update_filters_counter % 10 == 0:
            def get_distance(item):
                anc_id, estimator = item
                if estimator.last_x is None:
                    return float('inf')
                return estimator.last_x[0]**2 + estimator.last_x[1]**2
            
            sorted_est_dict_items = sorted(self.Est_dict.items(), key=get_distance)
            self.Est_dict = dict(sorted_est_dict_items)
        
        
        
    def imu_callback(self, msg):
        self.a_body = np.array([msg.accelerometer_m_s2[0],
                                msg.accelerometer_m_s2[1],
                                msg.accelerometer_m_s2[2]]).reshape(-1,1)
        self.a_stamp_us = msg.timestamp

def get_meas_seen(Anchors, pos_robot, distance_th=15, noise_uwb = 0.15):
    '''
    Given the robot position, return the anchors seen within distance_th
    Returns: positions (Nx3), [id, distance_noisy] (Nx2)
    '''
    Anc_seen = []
    for row in Anchors:
        ancx, ancy, ancz, id = row[0], row[1], row[2], row[3]
        px, py, pz = pos_robot[0], pos_robot[1], pos_robot[2]
        # Compute distance
        distance = np.sqrt( (px - ancx)**2 + (py - ancy)**2 + (pz - ancz)**2 )
        if distance <= distance_th:
            distance_noisy = distance + np.random.normal(0, noise_uwb)
            new_row = [ancx, ancy, ancz, id, distance_noisy]
            Anc_seen.append(new_row)
            
    Anc_seen_vec = np.array(Anc_seen)
    
    if not Anc_seen:
        return np.empty((0, 3)), np.empty((0, 2))
    
    if Anc_seen_vec.ndim == 1:
        Anc_seen_vec = Anc_seen_vec.reshape(1, -1)
        
    id_dist = Anc_seen_vec[:, 3:]
    positions = Anc_seen_vec[:, :3]
    return positions, id_dist
            
    
def main(args=None):
    rclpy.init(args=args)
    position_ekf = Position_EKF()
    try:
        rclpy.spin(position_ekf)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        position_ekf.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    
if __name__ == '__main__':
    main()