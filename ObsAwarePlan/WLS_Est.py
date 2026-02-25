import numpy as np  
import logging
import time
import sys
logger = logging.getLogger(__name__)


class WLS_Est:
    def __init__(self, th_move_delta=0.05, counter_still_th=5, std_uwb=0.15, min_distance_meas = 0.05,   
                 rolling_window_size=100, max_iters=10, min_measurements=10, id=None, a_z_given=None): 
        self.th_move_delta = th_move_delta
        self.min_distance_meas = min_distance_meas
        self.counter_still_th = counter_still_th
        self.std_uwb = std_uwb
        self.last_robot_pos = None
        self.last_x = None
        self.last_P = None  
        self.max_iters = max_iters
        self.rolling_window_size = rolling_window_size
        self.window_data = []
        self.still_counter = 0
        self.stable_flag = False
        self.Jratio = 0
        self.min_measurements = min_measurements
        self.id = id
        self.last_x_still = None
        self.a_z_given = a_z_given # Altitude of the anchor
        self.perpendicular_dir = None

        self.last_cov_GN = 1e6 * np.eye(2, dtype=np.float64)
        self.hdop_geom = np.inf
        
    def update(self, robot_pos, Pxx, Pyy, Pxy, measurements):
        # Early exit if no previous position
        robot_pos = np.asarray(robot_pos, dtype=np.float64)
        if self.last_robot_pos is None:
            self.last_robot_pos = robot_pos.copy()
            return self.last_x, self.last_P, self.stable_flag
        
        # Check if robot has moved significantly
        if np.hypot(robot_pos[0] - self.last_robot_pos[0], robot_pos[1] - self.last_robot_pos[1]) < self.min_distance_meas:
            return self.last_x, self.last_P, self.stable_flag

        # Update last position and add new data
        self.last_robot_pos = robot_pos.copy()
        self.window_data.append([robot_pos[0], robot_pos[1], robot_pos[2], Pxx, Pyy, Pxy, measurements])
        if len(self.window_data) > self.rolling_window_size:
            self.window_data.pop(0) 
        
        # Not enough measurements yet
        if len(self.window_data) <  self.min_measurements:
            return self.last_x, self.last_P, self.stable_flag
                
        # Update last position and add new data
        x_est, cov_est = self.estimatePosition()
        self.last_x, self.last_P = x_est, cov_est
        
        if self.isDeviceStill(x_est):
            self.still_counter += 1
            if self.still_counter > self.counter_still_th:
                self.stable_flag = True
                return self.last_x, self.last_P, self.stable_flag
        else:
            self.stable_flag = False
            self.still_counter = 0

        return self.last_x, self.last_P, self.stable_flag
    
    def estimatePosition(self):
        data_i = np.asarray(self.window_data, dtype=np.float64)
        Pos = data_i[:, 0:3]               # [x_r, y_r, z_r]
        Pxx, Pyy, Pxy = data_i[:,3], data_i[:,4], data_i[:,5]
        measurements  = data_i[:,6]

        
        x_est = self.WLLS_2_5D(Pos, Pxx, Pyy, Pxy, measurements, self.std_uwb)
        self.Jratio = self._compute_Jratio(Pos, measurements, x_est)
        if x_est is None:
            raise ValueError("WLLS_2_5D returned None estimate.")
        x_refined, cov_gn = self.GN_WLS_2_5D(Pos, Pxx, Pyy, Pxy, measurements, self.std_uwb,
                              initial_guess=x_est.flatten(), max_iters=self.max_iters,
                              update_weights=True, damping=1e-9)
        self.last_cov_GN = cov_gn

        try:
            self.hdop_geom = self.compute_hdop_geom(Pos, x_refined.flatten())
        except Exception:
            self.hdop_geom = float("inf")

        return x_refined, cov_gn
    
    def isDeviceStill(self, x_refined):
        if self.last_x_still is None:
            self.last_x_still = x_refined.copy()
            return False
        delta_move = np.linalg.norm(x_refined - self.last_x_still)
        self.last_x_still = x_refined.copy()
        if delta_move < self.th_move_delta:
            return True
        return False
    
    
    def WLLS_2_5D(self, Pos, Pxx, Pyy, Pxy, measurements, std_uwb, lambda_reg=1e-9):
        n_rows = len(measurements)
        if n_rows < 3:
            return None  # non abbastanza misure

        Z = np.zeros((n_rows, 1), dtype=np.float64)
        H = np.zeros((n_rows, 3), dtype=np.float64)
        weights = np.zeros((n_rows,), dtype=np.float64)  # vettore pesi

        for i in range(n_rows):
            z_i, g_i, w_i = self._rows_WLLS_2_5D(Pos[i], Pxx[i], Pyy[i], Pxy[i], measurements[i], std_uwb)
            Z[i, 0]   = z_i
            H[i, :]   = g_i
            weights[i] = w_i

        # Whitening (evitiamo W nxn)
        sqrt_w = np.sqrt(np.maximum(weights, 0.0))[:, None]  # (n,1)
        H_w = H * sqrt_w                                    # (n,3)
        Z_w = Z * sqrt_w                                    # (n,1)

        # Normali regolarizzate
        A = H_w.T @ H_w + lambda_reg * np.eye(3, dtype=np.float64)  # (3,3)
        b = H_w.T @ Z_w                                             # (3,1)

        try:
            # SPD → Cholesky
            L = np.linalg.cholesky(A)
            y = np.linalg.solve(L, b)
            x_hat = np.linalg.solve(L.T, y)                         # (3,1)
        except np.linalg.LinAlgError:
            # fallback
            x_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
            x_hat = np.array(x_hat, dtype=np.float64).reshape(-1, 1)
            
        return x_hat.flatten()[:2]

    def GN_WLS_2_5D(self, Pos_robot, Pxx, Pyy, Pxy, measurements, std_uwb,
           initial_guess, max_iters=20, tol=1e-8,
           damping=1e-6, update_weights=True, backtracking=True):
        
        
        x = np.array(initial_guess, dtype=float).reshape(2)
        # initial weight
        weights = self._compute_w_vector_with_pose_uncertainty(Pos_robot, Pxx, Pyy, Pxy, x, std_uwb)
        sqrt_w = np.sqrt(np.maximum(weights, 1e-12))[:, None]  # (n,1) stessa cosa che fare reshape(-1,1)
        # initial cost
        J = self._compute_J(est_anc=x, X_robot=Pos_robot)
        r = self._compute_f_cost(est_anc=x, X_robot=Pos_robot, meas=measurements)
        
        # Whitening
        J_w = J *  sqrt_w  # n×2
        r_w = r * sqrt_w  # n×1
        
        cost_prev = float(np.dot(r_w.T, r_w))
        # cost_prev = (r.T @ W @ r).item()

        for _ in range(max_iters):
            if update_weights:
                # nello Jacobiano e residui usa i pesi aggiornati
                weights = self._compute_w_vector_with_pose_uncertainty(Pos_robot, Pxx, Pyy, Pxy, x, std_uwb)
                sqrt_w = np.sqrt(np.maximum(weights, 1e-12))[:, None]  # (n,1) stessa cosa che fare reshape(-1,1)
            
            # ri-calcolo J, r e whitening
            J = self._compute_J(est_anc=x, X_robot=Pos_robot)
            r = self._compute_f_cost(est_anc=x, X_robot=Pos_robot, meas=measurements)
            J_w = J *  sqrt_w  # n×2
            r_w = r * sqrt_w  # n×1
            
            H = np.dot(J_w.T, J_w) + damping * np.eye(2)
            g = np.dot(J_w.T, r_w)

            try:
                step = np.linalg.solve(H, g).reshape(2)
            except np.linalg.LinAlgError:
                step, *_ = np.linalg.lstsq(H, g, rcond=None)
                step = step.reshape(2)

            # backtracking line search to ensure decrease
            alpha = 1.0
            x_new = x - alpha * step
            r_new = self._compute_f_cost(est_anc=x_new, X_robot=Pos_robot, meas=measurements)
            # whitening
            r_w_new = r_new * sqrt_w  # n×1
            cost_new = float(np.dot(r_w_new.T, r_w_new))

            if backtracking:
                tries = 0
                while cost_new > cost_prev and tries < 10:
                    alpha *= 0.5
                    x_new = x - alpha * step
                    r_new = self._compute_f_cost(est_anc=x_new, X_robot=Pos_robot, meas=measurements)
                    r_w_new = r_new * sqrt_w  # n×1
                    cost_new = float(np.dot(r_w_new.T, r_w_new))
                    tries += 1

            if abs(cost_prev - cost_new) < tol:
                x = x_new
                break

            # accept step and continue
            x = x_new
            cost_prev = cost_new

        # --- covarianza finale (whitened) ---
        if update_weights:
            weights = self._compute_w_vector_with_pose_uncertainty(Pos_robot, Pxx, Pyy, Pxy, x, std_uwb)
            sqrt_w = np.sqrt(np.maximum(weights, 1e-12))[:, None]

        J = self._compute_J(est_anc=x, X_robot=Pos_robot)
        J_w = J * sqrt_w
        H_final = J_w.T @ J_w + damping * np.eye(2, dtype=np.float64)
        try:
            cov = np.linalg.inv(H_final)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H_final)

        return x.reshape(2, 1), cov
    
    def _compute_w_vector_with_pose_uncertainty(self, Pos_robot, Pxx, Pyy, Pxy, x_est, std_uwb):
        Pos_robot = np.asarray(Pos_robot, dtype=np.float64)
        x, y = x_est[0], x_est[1]

        dx = x - Pos_robot[:, 0]
        dy = y - Pos_robot[:, 1]
        dz = self.a_z_given - Pos_robot[:, 2]      # <-- usa Δz corretto (ancora - robot)

        dist3 = np.sqrt(dx*dx + dy*dy + dz*dz)
        dist3 = np.maximum(dist3, 1e-12)           # evita div/0

        # direzioni normalizzate corrette (derivate ∂r/∂px, ∂r/∂py)
        jx = dx / dist3
        jy = dy / dist3

        var_geom = jx*jx * Pxx + jy*jy * Pyy + 2.0 * jx * jy * Pxy
        var_tot  = var_geom + std_uwb**2
        w = 1.0 / np.maximum(var_tot, 1e-12)
        return w

    def _compute_J(self, est_anc, X_robot):
        '''
        ref è la posizione del robot
        x_state è l'ancora che stiamo stimando [px, py] (niente pz)
        '''
        p_refx, p_refy = est_anc[0], est_anc[1]                  # niente pz da x_state
        X_robot = np.asarray(X_robot, dtype=np.float64)
        dx = p_refx - X_robot[:, 0]
        dy = p_refy - X_robot[:, 1]
        dz = X_robot[:, 2] - self.a_z_given                  # z_robot - z_anchor_nota
        dist = np.sqrt(np.maximum(dx*dx + dy*dy + dz*dz, 1e-12))
        J = np.empty((X_robot.shape[0], 2), dtype=np.float64)
        J[:, 0] = dx / dist
        J[:, 1] = dy / dist
        return J  # Nx2

    def _compute_f_cost(self, est_anc, X_robot, meas):
        '''
        ref è la posizione del robot
        x_state è l'ancora che stiamo stimando [px, py] (niente pz)
        '''
        anc_x, anc_y = est_anc[0], est_anc[1]                  # niente pz da x_state
        X_robot  = np.asarray(X_robot,  dtype=np.float64)
        meas = np.asarray(meas, dtype=np.float64).reshape(-1)
        dx = anc_x - X_robot[:, 0]
        dy = anc_y - X_robot[:, 1]
        dz = X_robot[:, 2] - self.a_z_given                  # z_robot - z_anchor_nota
        dist = np.sqrt(np.maximum(dx*dx + dy*dy + dz*dz, 1e-12))
        res = dist - meas
        return res.reshape(-1, 1)  # Nx1

    def _rows_WLLS_2_5D(self, pos_robot, Pxx, Pyy, Pxy, meas, std_uwb):
        '''
        Appunti su ipad ma consideriamo (p_rz- a_z) nel termine noto
        '''
        if self.a_z_given is None:
            raise ValueError("Altitude estimate a_z_given is not set.")
        
        p_rx, p_ry, p_rz = pos_robot
        eps = 1e-12
        
        z = meas**2 - p_rx**2 - p_ry**2 - (p_rz - self.a_z_given)**2
        
        g = np.zeros((1,3), dtype=np.float64)
        g[0,0] = -2.0*p_rx
        g[0,1] = -2.0*p_ry
        g[0,2] = 1.0
        
        P = np.zeros((2,2), dtype=np.float64)
        P[0,0] = Pxx
        P[0,1] = Pxy
        P[1,0] = Pxy
        P[1,1] = Pyy
        
        # Compute the variance of the measurement given uncertainty in robot position
        # Var(z_i) ≈ 4 r_i^2 σ_r^2 + 4 p_i^T P p_i
        pos_xy = np.array([p_rx, p_ry], dtype=np.float64)
        tmp = np.dot(pos_xy.T, np.dot(P, pos_xy))
        r2 = np.maximum(meas**2, eps)
        var_z = 4.0 * r2 * (std_uwb**2) + 4.0 * tmp
        w = 1.0 / (var_z + eps)  # avoid division by zero
        return z, g, w
    
    
    def _compute_Jratio(self, Pos, measurements, x_est):
        '''
        Compute the ratio between the cost function of the LLS between the right and mirrored
        Pos is vector ndarray Nx3 with robot positions
        '''
        
        J_lls_rows = self._compute_f_cost(est_anc=x_est, X_robot=Pos, meas=measurements) 
        J_lls = np.sum(J_lls_rows**2)
        x_mirror = self._mirror_position(x_est, Pos)
        J_mirr_rows = self._compute_f_cost(est_anc=x_mirror, X_robot=Pos, meas=measurements)
        J_mirr = np.sum(J_mirr_rows**2)
        ratio = J_mirr / (J_lls + 1e-12) 
        self.Jratio = ratio
        return ratio
    
    def _mirror_position(self, x_est, Pos):
        x_est = x_est.reshape(-1)
        mean_pos = _compute_mean(Pos).reshape(-1)
        principal_dir, _, self.perpendicular_dir = principal_direction(Pos)
        norm_u = np.linalg.norm(principal_dir)
        if norm_u == 0 or not np.isfinite(norm_u):
            # fallback: nessuna direzione definita (sensori allineati tutti uguali, ecc.)
            # in questo caso rifletto rispetto al punto medio (cioè inversione centrale)
            x_mirror = 2*mean_pos - x_est
        else:
            u = principal_dir / norm_u
            x_mirror = mean_pos + (mean_pos - x_est) + 2.0 * u * np.dot(u, (x_est - mean_pos))
        return x_mirror
    
    def compute_hdop_geom(self, Pos, anc_xy):
        """
        HDOP geometrico adimensionale: sqrt( trace( (J^T J)^{-1} ) ), pesi unitari.
        """
        J = self._compute_J(est_anc=np.asarray(anc_xy).reshape(2), X_robot=Pos)
        H = J.T @ J
        try:
            P = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(H)
        return np.sqrt(P[0,0] + P[1,1])
            
def _compute_mean(Pos):
        if len(Pos) == 0:
            return None
        data = np.array(Pos)
        mean_pos = np.mean(data[:, 0:2], axis=0)
        return mean_pos.reshape(2,1)
    
def principal_direction(Pos):
    '''
    return: principal_dir, principal_val, perpendicular_dir
    '''
    data = np.array(Pos)
    cov = np.cov(data[:,0:2].T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    max_index = np.argmax(eigvals)
    principal_dir = eigvecs[:, max_index]
    principal_val = eigvals[max_index]
    perpendicular_dir = eigvecs[:, (max_index + 1) % 2]
    return principal_dir, principal_val, perpendicular_dir

