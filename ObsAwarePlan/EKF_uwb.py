import numpy as np

class EKF_UWB:
    def __init__(self, x0, P, sig_a_proc, sig_a_meas, CA_dynamic=False, use_acc_pseudomeas = False,
                 Anchors_fix=None, uwb_noise_std=0.15, initial_P_ancSlam_std=0.2, z_anc=0.0):
        self.x = x0  # State vector
        self.P = P   # Covariance matrix
        
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()
        
        self.sig_a_proc = sig_a_proc  # Process noise standard deviation for acceleration
        self.sig_a_meas = sig_a_meas  # Measurement noise standard deviation for acceleration
        self.Anchors_fix = Anchors_fix  # Fixed anchor positions
        
        self.use_CA = CA_dynamic  # Constant Acceleration dynamic model flag
        self.use_acc_pseudomeas = use_acc_pseudomeas
        self.uwb_noise_std = uwb_noise_std
        self.initial_P_Slam_std = initial_P_ancSlam_std 
        
        self.id_to_idx = {}  # Mapping from anchor ID to state index for SLAM anchors
        self.counter = 0  # Counter for SLAM anchors added
        self.z_anc = z_anc  # Assumed fixed height for unknown UWB anchors
        
        if self.use_CA:
            # Extend state and covariance for CA model
            self.x = np.vstack((self.x, np.zeros((3, 1))))  # Add acceleration states
            n = self.P.shape[0]
            P_ext = np.zeros((n+3, n+3))
            P_ext[:n, :n] = self.P
            P_ext[n:, n:] = np.eye(3) * 1.0  # Initial uncertainty for acceleration
            self.P = P_ext
    @property
    def pos(self):
        '''Getter for position state flattened'''
        return self.x[:3].flatten()     # getter
    
    @property
    def cov_2d(self):
        Pxx = self.P[0,0]
        Pyy = self.P[1,1]
        Pxy = self.P[0,1]
        return Pxx, Pyy, Pxy
        
    def prediction(self, dt, a):
        if self.use_CA:
            n_uwb = n_uwb_in_state(self.x)
            A = compute_A_CA(dt, x=self.x)
            Q = compute_Q_CA(dt, sigma_j=0.7, n_uwb=n_uwb, q_uwb =1e-20)
            self.x_pred = A @ self.x 
            self.P_pred = A @ self.P @ A.T + Q
        else:    
            # State transition matrix
            A = compute_A(dt)
            self.x_pred = A @ self.x + np.vstack((0.5 * a * dt**2, a * dt))

            Qd = make_Q(dt, sigma_a_proc=self.sig_a_proc, sigma_a_meas=self.sig_a_meas)
            self.P_pred = A @ self.P @ A.T + Qd
        
        
    def update(self, a_enu=None, id_dist_meas=None):
        if id_dist_meas is None:
            return
        
        id_meas_fix =  id_dist_meas[ id_dist_meas[:, 0] >= 100 ]
        id_meas_slam_all = id_dist_meas[ id_dist_meas[:, 0] < 100 ]
        
        # Dobbiamo tenere le misure slam che abbiamo nello stato
        ids_tracked = np.fromiter(self.id_to_idx.keys(), dtype=int)
        mask_tracked = np.isin(id_meas_slam_all[:, 0].astype(int), ids_tracked)

        #Finalmente abbiamo le misure delle ancore SLAM nel nostro stato
        id_meas_slam_tracked = id_meas_slam_all[mask_tracked]
        id_slam_tracked = id_meas_slam_tracked[:,0].astype(int)        
        
        meas_fix = id_meas_fix[:,1]
        meas_slam = id_meas_slam_tracked[:,1]
        meas_uwb = np.vstack((meas_fix.reshape(-1,1), meas_slam.reshape(-1,1)))
  
        if id_meas_slam_all is not None:
            Anc_pos_fix = self.anc_pos_gt_from_meas(id_meas_fix) 
            # print("Anchor positions from measurements:\n", Anc_pos_fix)
        

        # --- 1) UPDATE CON LE MISURE UWB (come prima) ---
        H = self.compute_H_slam(Anc_pos_fix, id_anchors_slam=id_slam_tracked, anc_z=self.z_anc)
        R_meas = np.eye(H.shape[0]) * (self.uwb_noise_std**2)
        S = H @ self.P_pred @ H.T + R_meas

        # Regolarizzazione adattiva (aiuta molto con coplanarità/condizionamento)
        eps = 1e-6 * np.trace(S) / S.shape[0]
        S = S + eps * np.eye(S.shape[0])

        # Gain per le misure UWB
        W = np.linalg.solve(S, H @ self.P_pred.T).T
        
        # Innovazione UWB
        h = self.compute_h_slam(Anc_pos_fix, id_anchors_slam=id_slam_tracked, anc_z=self.z_anc)
        innovation_uwb = meas_uwb - h

        # Stato e covarianza dopo l'update UWB
        self.x = self.x_pred + W @ innovation_uwb
        n_state = self.x.shape[0]
        self.P = (np.eye(n_state) - W @ H) @ self.P_pred

        # Optional: Update with acceleration pseudomeasurements
        if self.use_CA and self.use_acc_pseudomeas and (a_enu is not None):
            self.correction_acceleration_pseudomeas(a_enu)
            

    def correction_acceleration_pseudomeas(self, a_enu):
        # a_enu deve essere accelerazione traslazionale in ENU (quella che hai già ruotato e
        # a cui hai aggiunto la gravità in ENU nel nodo ROS).
        z_acc = np.array(a_enu, dtype=float).reshape(3, 1)  # 3x1
        n_state = self.x.shape[0]   
        # Matrice di osservazione per le componenti di accelerazione
        H_acc = np.zeros((3, n_state))
        H_acc[0, 6] = 1.0  # a_x
        H_acc[1, 7] = 1.0  # a_y
        H_acc[2, 8] = 1.0  # a_z

        # Predizione della misura: accelerazioni di stato in ENU
        h_acc = self.x[6:9, :]  # 3x1

        # Rumore di misura sull'accelerazione (usa sigma_a_meas che hai già)
        R_acc = (self.sig_a_meas**2) * np.eye(3)

        # Innovazione pseudomisura: a_enu - a_stato
        innovation_acc = z_acc - h_acc

        # Covarianza dell'innovazione
        S_acc = H_acc @ self.P @ H_acc.T + R_acc
        eps_acc = 1e-6 * np.trace(S_acc) / S_acc.shape[0]
        S_acc = S_acc + eps_acc * np.eye(S_acc.shape[0])

        # Gain per pseudomisura accelerazione
        W_acc = self.P @ H_acc.T @ np.linalg.pinv(S_acc)
        

        # Update finale con la pseudomisura
        self.x = self.x + W_acc @ innovation_acc
        self.P = (np.eye(n_state) - W_acc @ H_acc) @ self.P
    
    def add_anc_slam_to_state(self, Anc_gt, anc_id, anc_x, anc_y):
        new_anchor_idx = self.x.shape[0]
        self.id_to_idx[anc_id] = new_anchor_idx
        
        self.x = np.vstack((self.x, np.array([[anc_x], [anc_y]])))
        self.P = get_diag(self.P, np.eye(2) * self.initial_P_Slam_std**2)  # Initial uncertainty for new anchor
        
    
    def compute_h(self, Anchors):
        '''
            Distance with sqrt root
            Create a matrix with    
        '''
        h_vec = []
        for row in Anchors:
            ancx, ancy, ancz = row[0], row[1], row[2]
            # Compute distance
            distance = np.sqrt( (self.x_pred[0,0]-ancx)**2 + (self.x_pred[1,0]-ancy)**2 + (self.x_pred[2,0]-ancz)**2 )      
            h_vec.append(distance)
        h = np.array(h_vec).reshape(-1,1)
        return h

    def compute_H(self, Anchors):
        '''
        Jacobian of h with respect to state x
        Non mi serve sapere quale ancore, do solo le ancore che vedo. 
        '''
        n_state = self.x_pred.shape[0]
        H_mat = []
        for row in Anchors:
            ancx, ancy, ancz = row[0], row[1], row[2]
            dx = self.x_pred[0,0] - ancx
            dy = self.x_pred[1,0] - ancy
            dz = self.x_pred[2,0] - ancz
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            H_row = np.zeros((1, n_state))
            H_row[0,0] = dx / dist
            H_row[0,1] = dy / dist
            H_row[0,2] = dz / dist
            # H_row = np.array([[dx/dist, dy/dist, dz/dist, 0, 0, 0]])
            H_mat.append(H_row)
        H = np.vstack(H_mat)
        return H
    
    def compute_H_slam(self, anchor_known, id_anchors_slam, anc_z):
        '''
        State vector : [x, y,z, v_x, v_y,v_z, a_x, a_y, a_z, uwb1_x, uwb1_y, ...].T
        
        Return the measurement matrix H for EKF-SLAM with UWB ranges.
        H = n_measurements x n_state
        '''
        x_pred = np.asarray(self.x_pred)
        n_state = x_pred.shape[0]
        px, py, pz = x_pred[0], x_pred[1], x_pred[2]
        H_mat = []
        for anc in anchor_known:
            row = np.zeros((1, n_state))
            dx = px - anc[0]
            dy = py - anc[1]
            dz = pz - anc[2]
            r = max(np.sqrt(dx*dx + dy*dy + dz*dz), 1e-12)
            row[0, 0] = dx/r
            row[0, 1] = dy/r
            row[0, 2] = dz/r

            H_mat.append(row)
        if id_anchors_slam is not None:
            for id_anc in id_anchors_slam:
                row = np.zeros((1, n_state))

                idx_state = self.id_to_idx[id_anc]
                dx = px - x_pred[idx_state]
                dy = py - x_pred[idx_state + 1]
                dz = pz - anc_z  # assumed fixed height for unknown UWB anchors
                r = max(np.sqrt(dx*dx + dy*dy + dz*dz), 1e-12)
                row[0, 0] = dx/r
                row[0, 1] = dy/r
                row[0, 2] = dz/r
                row[0, idx_state]     = -dx/r
                row[0, idx_state + 1] = -dy/r
                H_mat.append(row)

        if not H_mat:
            return np.zeros((0, n_state))
        return np.vstack(H_mat)
    
    def compute_h_slam(self, anchors_known, id_anchors_slam, anc_z):
        '''
        State vector : [x, y,z, v_x, v_y,v_z, a_x, a_y, a_z, uwb1_x, uwb1_y, ...].T
        Return the difference between the est state of the robot and the UWB positions
        as range measurements.
        
        - anchors_known: lista di ancore note con posizione fissa
        - id_to_idx: dizionario che mappa l'id dell'ancora all'indice nello stato
        - id_anchors_slam: lista di id delle ancore stimate nello stato
        
        '''
        x = self.x_pred.flatten()
        h_vec = []
        px, py, pz = x[0], x[1], x[2]
        for anc in anchors_known:
            dx = px - anc[0]
            dy = py - anc[1]
            dz = pz - anc[2]
            h_vec.append(np.sqrt(dx*dx + dy*dy + dz*dz))
        
        if id_anchors_slam is not None:
            for id_anc in id_anchors_slam:
                idx_state = self.id_to_idx[id_anc]
                dx = px - x[idx_state]
                dy = py - x[idx_state + 1]
                dz = pz - anc_z  # assumed fixed height for unknown UWB anchors
                h_vec.append(np.sqrt(dx*dx + dy*dy + dz*dz))

        if len(h_vec) == 0:
            return np.empty((0,1))
        return np.array(h_vec, dtype=float).reshape(-1, 1)
    def anc_pos_gt_from_meas(self, id_dist_meas):
        '''
        Given anchor ID, return its position
        The id of the anchor is row number in the Anchors_fix list
        '''
        ids_in_message = id_dist_meas[:,0].astype(int)
        Positions_anc = []
        for id_anc in ids_in_message:
            # Note that Anchors_fix IDs start from 100
            id_fix = int(id_anc)-100
            if self.Anchors_fix is None:
                raise ValueError("Anchors_fix is not set in EKF_UWB instance.")
            if id_fix < 0 or id_fix >= self.Anchors_fix.shape[0]:
                print(f"Anchor ID {id_anc} out of bounds!")
                continue
            Positions_anc.append(self.Anchors_fix[id_fix][:3])
        return np.array(Positions_anc)
        



def compute_A(dt):
    '''
    x = [x, y, z, vx, vy, vz]^T 
    '''
    top_left = np.eye(3)
    top_right = np.eye(3) * dt
    bottom_left = np.zeros((3, 3))
    bottom_right = np.eye(3)
    A = np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])
    return A

def make_Q(dt, sigma_a_proc=0.2, sigma_a_meas=0.2):
    '''
    Q = Q_model + Q_imu
    Q_model: processo di rumore sul modello (accelerazione come rumore bianco)
    Q_imu: rumore dovuto alla misura dell'accelerazione
    dt: intervallo di tempo [s]
    '''
    I3 = np.eye(3)
    # blocco 2x2 per una singola asse
    q11 = (dt**4) / 4.0
    q12 = (dt**3) / 2.0
    q22 = (dt**2)
    Q1D = np.array([[q11, q12],
                    [q12, q22]]) * (sigma_a_proc**2)
    Q_model = np.kron(I3, Q1D)  # 6x6

    # B per ingresso accelerazione
    B = np.block([
        [0.5*(dt**2)*I3],
        [dt*I3]
    ])  # (6x3)

    R_a = (sigma_a_meas**2) * I3  # 3x3
    Q_imu = B @ R_a @ B.T          # 6x6

    return Q_model + Q_imu

def compute_A_CA(dt, x=None):
    """
    State transition matrix A for Constant Acceleration (CA) model in 3D.
    State vector: [x, y, z, v_x, v_y, v_z, a_x, a_y, a_z].T
    Resulting matrix A (9×9):

        | 1 0 0  dt  0  0  0.5*dt²     0        0     |
        | 0 1 0   0 dt  0      0   0.5*dt²      0     |
        | 0 0 1   0  0 dt      0       0    0.5*dt²   |
        | 0 0 0   1  0  0     dt       0        0     |
        | 0 0 0   0  1  0      0      dt        0     |
        | 0 0 0   0  0  1      0       0       dt     |
        | 0 0 0   0  0  0      1       0        0     |
        | 0 0 0   0  0  0      0       1        0     |
        | 0 0 0   0  0  0      0       0        1     |

    """
    n_state = len(x) 
    A_fin = np.eye(n_state)
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    A = np.block([
        [I3, dt * I3, 0.5 * dt**2 * I3],
        [Z3,    I3,          dt * I3],
        [Z3,    Z3,             I3   ]
    ])
    A_fin[:9, :9] = A

    return A_fin

def compute_Q_CA(dt, sigma_j=0.7, q_uwb=1e-8, n_uwb=0):
    dt2, dt3, dt4, dt5 = dt*dt, dt**3, dt**4, dt**5
    Q3 = np.array([
        [dt5/20.0, dt4/8.0, dt3/6.0],
        [dt4/8.0,  dt3/3.0, dt2/2.0],
        [dt3/6.0,  dt2/2.0, dt      ]
    ]) * (sigma_j**2)

    # Stato: [px,py,pz, vx,vy,vz, ax,ay,az]
    Q9 = np.zeros((9, 9))
    for a in range(3):                  # assi: 0->x, 1->y, 2->z
        idx = [a, 3+a, 6+a]             # (p, v, a) dell'asse a
        Q9[np.ix_(idx, idx)] = Q3

    dim = 9 + 2*n_uwb                   # UWB nello stato sono 2D nel tuo codice
    Q = np.zeros((dim, dim))
    Q[:9, :9] = Q9

    for i in range(n_uwb):              # random walk 2D per ogni ancora
        s = 9 + 2*i
        Q[s:s+2, s:s+2] = q_uwb * dt * np.eye(2)

    return Q

def get_diag(A, B):
    """
    Return the block diagonal matrix:
        [[A, 0],
         [0, B]]
    """
    z1 = np.zeros((A.shape[0], B.shape[1]))
    z2 = np.zeros((B.shape[0], A.shape[1]))
    return np.block([[A, z1],
                     [z2, B]])
    
def n_uwb_in_state(x):
    '''
    Given the state vector x of EKF-SLAM with UWB positions,
    return the number of UWB anchors in the state.
    State vector: [x, y, v_x, v_y, a_x, a_y, uwb1_x, uwb1_y, ...].T
    '''
    dim = x.shape[0]
    n_uwb = (dim - 9) // 2
    return n_uwb