import numpy as np
import logging


logger = logging.getLogger(__name__)
class RP_Planner:
    
    def __init__(self, smooth = 0.3, max_turn_deg = 30.0, Jratio_th = 1.0):
        self.smooth = smooth
        self.max_turn_deg = max_turn_deg
        self.prev_direction = None
        self.Jratio_threshold = Jratio_th
    
    def update_direction(self, est_dict, goal_pos, x_robot):
        new_direction = self.compute_force(est_dict, goal_pos, x_robot)  
        if list(est_dict.keys()) == []:
            return new_direction
        
        smoothed_direction = self.smooth_direction(new_direction)        
        return smoothed_direction
    
    def compute_force(self, est_dict, goal_pos, x_robot):
        goal_direction = Attractive_force_dir(x_robot, goal_pos)

        for est_id, est in est_dict.items():
            # skip if no covariance or already initialized
            if est.last_x is None: continue
            if est.stable_flag is True: continue
            # est_anc = est.est_GN if est.est_GN is not None else est.est_lin
            est_anc = est.last_x 
            
            distanza = np.linalg.norm(est_anc - x_robot)
            # print(f"Distanza from anchor {est_id}: {distanza:.2f} m")
            if distanza > 11.0:
                continue
            
            #1. Symmetry breaking via Jratio
            if est.Jratio < self.Jratio_threshold and est.Jratio is not None:
                direction_break_symmetry = direction_max_Jratio(est, goal_direction)
                
                # Check if direction is valid-> case where principal direction is ill-defined
                if direction_break_symmetry is None: continue
                logger.info(f"Breaking symmetry {est_id} Jratio:{est.Jratio:.3f}")
                return direction_break_symmetry
            
            #2. Tangent direction correction
            else:
                result = _direction_max_info(x_robot, goal_direction, est_anc)
                if result is not None:
                    return result
                else:
                    continue
        return  goal_direction
        
    def smooth_direction(self, new_direction):
        if self.prev_direction is None:
            self.prev_direction = new_direction.copy()
            return new_direction
        else:
            limited = _limit_turn(self.prev_direction, new_direction, self.max_turn_deg)
            new_smooth_dir = self.smooth*limited + (1.0 - self.smooth)*self.prev_direction
            n = np.linalg.norm(new_smooth_dir)
            if n > 1e-12:
                new_smooth_dir = new_smooth_dir / n
            self.prev_direction = new_smooth_dir.copy()
        return new_smooth_dir                
    
def direction_max_Jratio(est, goal_dir):
    # get the direction perpendicular to the principal direction
    perp_vec = est.perpendicular_dir
    if perp_vec is None or np.linalg.norm(perp_vec) < 1e-6:
        return None
    perp_dir = _normalize(perp_vec)
    
    # align to goal half-space
    if np.dot(goal_dir, perp_dir) < 0:
        perp_dir = -perp_dir
    
    # compute the new direction
    jr_clip = np.clip(est.Jratio, 0.0, 1.0)
    w = np.clip(0.25 + 0.6 * (1.0 - jr_clip), 0.25, 0.85)
    u = goal_dir - w * perp_dir
    result = _normalize(u)
    return result

def _direction_max_info(x_robot, goal_dir, est_anc):
    tangent_dir = _direction_tan_robot(x_robot, np.asarray(est_anc).flatten())
    nt = np.linalg.norm(tangent_dir)
    if nt < 1e-12: return None
    
    # allinea al semispazio del goal e calcolo angolo
    if np.dot(goal_dir, tangent_dir) < 0:
        tangent_dir = -tangent_dir
    cosang = np.clip(np.dot(goal_dir, tangent_dir), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosang))
    TH_ANGLE_1 = 55.0
    TH_ANGLE_2 = 90.0
    if angle_deg < TH_ANGLE_1:
        result = tangent_dir
    elif angle_deg >= TH_ANGLE_2:
        result = goal_dir
        # niente break: puoi lasciare spazio ad altre ancore solo se vuoi
    else:
        # blend lineare semplice e stabile
        a = (angle_deg - TH_ANGLE_1) / (TH_ANGLE_2 - TH_ANGLE_1)
        u = (1.0 - a) * goal_dir + a * tangent_dir
        u = _normalize(u)
        if np.linalg.norm(u) < 1e-12:
            result = goal_dir
        else:
            result = u
    return result


def _normalize(v):
    norm = np.linalg.norm(v)
    # if norm < 1e-8:
    #     return v
    return v / (norm + 1e-12)



def _direction_tan_robot(x_robot, x_anc):
    distance = x_anc - x_robot
    radial_dir = _normalize(distance)
    if np.linalg.norm(distance) == 0:
        return np.zeros(2)
    tangent = np.array([-radial_dir[1], radial_dir[0]])
    return tangent

def _limit_turn(prev_vec, new_vec, max_deg):
    p = _normalize(prev_vec)
    n = _normalize(new_vec)
    c = float(np.clip(np.dot(p, n), -1.0, 1.0))
    ang = np.degrees(np.arccos(c))
    if ang <= max_deg:
        return n
    frac = max_deg / (ang + 1e-12)
    theta = np.arccos(c)
    if theta < 1e-6:
        return n
    s = np.sin(theta)
    return (np.sin((1-frac)*theta)/s)*p + (np.sin(frac*theta)/s)*n

def Attractive_force_dir(x_robot, x_goal):
    """
    Calculate the attractive force towards the goal.
    
    Parameters:
    x_robot : np.ndarray
        Current position of the robot (2D).
    x_goal : np.ndarray
        Goal position (2D).
    zeta : float
        Attractive potential gain.
        
    Returns:
    np.ndarray
        Attractive force vector (2D).
    """
    direction = x_goal - x_robot
    distance = np.linalg.norm(direction)
    if distance == 0:
        return np.zeros(2)
    force = direction / distance
    return force