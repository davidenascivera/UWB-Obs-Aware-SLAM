import numpy as np

def is_far_enough(pt, others, min_dist):
    for o in others:
        if np.linalg.norm(pt - o) < min_dist:
            return False
    return True

def S(w):
    w = np.array(w).reshape(3,) 
    wx, wy, wz = w
    return np.array([[0, -wx, -wy, -wz],
                     [wx, 0, wz, -wy],
                     [wy, -wz, 0, wx],
                     [wz, wy, -wx, 0]])
def Jnorm(x):
    r = np.linalg.norm(x)
    return (r**2*np.eye(3) - x @ x.T) / (r**3)

def compute_alpha(a, m):
    '''
    Funzione che calcola la misura del magnetometro usando il vettore di gravità (misura senza l'ultimo elemento alpha)
    a, m: shape (3,1) o (3,)
    '''
    a = np.asarray(a).reshape(3,1)
    m = np.asarray(m).reshape(3,1)  
    m_D = a.T @ m
    # print(m_D)
    m_N = np.sqrt(1 - m_D**2)
    alpha = (a[1,0]*m[2,0] - a[2,0]*m[1,0]) / m_N
    return alpha

def h(q):
    q = q.reshape(-4,)
    q0, q1, q2, q3 = q
    Za = Za_from_q(q)
    return np.vstack((Za, 2*q3*q0 + 2*q1*q2))

def Za_from_q(q):
    '''
    Funzione che calcola la misura dell'accelerometro usando il quaternione (misura senza l'ultimo elemento alpha)
    '''
    q = q.reshape(-4,)
    q0, q1, q2, q3 = q
    return np.array([
        [-2*q2*q0 + 2*q3*q1],
        [ 2*q1*q0 + 2*q3*q2],
        [ q0**2 - q1**2 - q2**2 + q3**2]
    ])

    
def Rb(q):
    '''
    Rotation matrix from body to inertial frame
    
    v_w = R(q) v_b
    '''
    q0, q1, q2, q3 = q[0,0], q[1,0], q[2,0], q[3,0]
    
    return np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3,  2*q2*q1 - 2*q3*q0,            2*q2*q0 + 2*q3*q1],
        [2*q3*q0 + 2*q2*q1,              q0*q0 - q1*q1 + q2*q2 - q3*q3, -2*q1*q0 + 2*q3*q2],
        [-2*q2*q0 + 2*q3*q1,             2*q1*q0 + 2*q3*q2,             q0*q0 - q1*q1 - q2*q2 + q3*q3]
    ], dtype=float)

def G(q):
    q = q.reshape(-4,)
    q0, q1, q2, q3 = q
    return np.array([[- q1, -q2, -q3],
                     [ q0, -q3,  q2],
                     [ q3,  q0, -q1],
                     [-q2,  q1,  q0]])


def H(q):
    '''
    Per calcolare H dobbiamo fare lo jacobiano di Z rispetto a q
     h(q) = -2q2q0 + 2q3q1
            2q1q0 + 2q3q2
            q0^2 - q1^2 - q2^2 + q3^2
            2q3q0 + 2q1q2
    '''
    q = q.reshape(-4,)
    q0, q1, q2, q3 = q
    return np.array([
        [-2*q2,  2*q3, -2*q0,  2*q1],
        [ 2*q1,  2*q0, 2*q3,  2*q2],
        [ 2*q0, -2*q1, -2*q2,  2*q3],
        [ 2*q3,  2*q2,  2*q1,  2*q0]
    ])
    


def Ja(a, m, eps=1e-9):
    """
    J_a = d alpha / d a  (1x3)
    alpha(a,m) = (a_y m_z - a_z m_y) / sqrt(1 - (a^T m)^2)
    a, m: shape (3,) oppure (3,1), nominali (unitari)
    
    NB: Nella funzione e implementazione dobbiamo passarli a e m. a possiamo calcolarlo come a(q), in quanto
    è facile ottenerlo usando la funzione Z_a, invece per Z_m è più complesso e usiamo direttamente m
    """
    a = np.asarray(a, float).reshape(3,1)
    m = np.asarray(m, float).reshape(3,1)
    s  = float(a.T @ m)                              # a^T m
    s  = np.clip(s, -1.0, 1.0)
    mN = max(np.sqrt(max(1.0 - s*s, 0.0)), eps)      # sqrt(1 - s^2)
    vx = float(a[1,0]*m[2,0] - a[2,0]*m[1,0])        # (a × m)_x
    c  = vx * s / (mN**3)
    return np.array([[ c*m[0,0],
                       m[2,0]/mN + c*m[1,0],
                      -m[1,0]/mN + c*m[2,0] ]])

def Jm(a, m, eps=1e-9):
    """
    J_m = d alpha / d m  (1x3)
    alpha(a,m) = (a_y m_z - a_z m_y) / sqrt(1 - (a^T m)^2)
    a, m: shape (3,) oppure (3,1), nominali (unitari)
    """
    a = np.asarray(a, float).reshape(3,1)
    m = np.asarray(m, float).reshape(3,1)
    s  = float(a.T @ m)
    s  = np.clip(s, -1.0, 1.0)
    mN = max(np.sqrt(max(1.0 - s*s, 0.0)), eps)
    vx = float(a[1,0]*m[2,0] - a[2,0]*m[1,0])
    c  = vx * s / (mN**3)
    return np.array([[ c*a[0,0],
                      -a[2,0]/mN + c*a[1,0],
                       a[1,0]/mN + c*a[2,0] ]])


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.
    Convention: ZYX (yaw → pitch → roll).
    
    Parameters
    ----------
    roll : float
        Rotation about the x-axis (rad)
    pitch : float
        Rotation about the y-axis (rad)
    yaw : float
        Rotation about the z-axis (rad)

    Returns
    -------
    q : np.ndarray, shape (4,)
        Quaternion [qw, qx, qy, qz]
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])

def quaternion_to_euler(q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    Convention: ZYX (yaw -> pitch -> roll).
    """
    qw, qx, qy, qz = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.pi/2 * np.sign(sinp)  # use 90° if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def bootstrap_euler(a,m):
    '''
    Stimatore statico per trovare gli angoli di eulero a partire dalle misure di accelerometro e magnetometro.
    Utile per inizializzare il filtro
    '''
    ax = a[0]
    ay = a[1]
    az = a[2]
    mx = m[0]
    my = m[1]
    mz = m[2]
    theta = -np.arcsin(ax)
    phi = np.arctan2(ay, az)
    m_D_unclip = a[:].T@m[:]
    m_D = np.clip(m_D_unclip, -1.0+1e-12, 1.0-1e-12)
    psi = np.arcsin((ay*mz - az*my)/np.sqrt(1-m_D**2))
    
    return phi, theta, psi

def enu_to_ned_quaternion(q_enu):
    """
    Converte quaternion da ENU a NED
    Input: q_enu = [q0, q1, q2, q3] (w, x, y, z) in ENU
    Output: q_ned = [q0, q1, q2, q3] (w, x, y, z) in NED
    """
    # Inverso della trasformazione NED->ENU
    q_ned = np.array([q_enu[0], q_enu[2], q_enu[1], -q_enu[3]])
    return q_ned / np.linalg.norm(q_ned)

def quat_from_axis_angle(axis, angle_rad):
    """Crea un quaternione da un asse (unitario) e un angolo [rad]."""
    axis = np.array(axis) / np.linalg.norm(axis)
    s = np.sin(angle_rad / 2.0)
    return np.array([np.cos(angle_rad / 2.0), axis[0]*s, axis[1]*s, axis[2]*s])

def quat_multiply(q1, q2):
    """
    Moltiplicazione tra due quaternioni (ordine [w, x, y, z]).

    Restituisce q = q1 * q2, cioè applica prima q2 e poi q1.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)  # normalizza per sicurezza
    return q


def Rbody_to_NED(q):
        '''
        q = [q0, q1, q2, q3]^T
        '''
        q0, q1, q2, q3 = q.reshape(-1)
        R = np.array([
            [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
            [2*(q1*q2 - q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 + q0*q1)],
            [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
        ])
        return R
    
def msg_to_array(msg):
    '''
    Converti un messaggio di tipo Float32MultiArray in una numpy array
    Assumiamo che il messaggio sia strutturato come:
        data = [id_ancora_1, distanza_1, id_ancora_2, distanza_2, ..., id_ancora_n, distanza_n]
    Restituisce un array con solo le distanze misurate
    '''
    list_anc = []
    for i in range(0, len(msg.data), 2):
        list_anc.append(msg.data[i + 1])
    return np.array(list_anc)   
    

def Rbody_to_ENU(q):
    w, x, y, z = q.reshape(-1)
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)]
    ])
    return R