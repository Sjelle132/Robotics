import numpy as np

# Robot dimensions [mm] from problems 1-9
L1, L2, L3, L4, L5, L6 = 50., 93., 93., 50., 45., 35.


def DH(q):
    q = np.asarray(q, dtype=float)
    return np.array([
        [q[0],           L1, 0., np.pi/2],
        [q[1] + np.pi/2, 0., L2, 0.],
        [q[2],           0., L3, 0.],
        [q[3],           0., L4, 0.],
    ])


def FK(q):
    """Returns T0, T1, ..., T5. T4: stylus; T5: camera mount."""
    T = [np.eye(4)]

    for theta, d, a, alpha in DH(q):
        c, s = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        A = np.array([
            [c, -s*ca,  s*sa, a*c],
            [s,  c*ca, -c*sa, a*s],
            [0.,   sa,    ca,   d],
            [0.,   0.,    0.,  1.],
        ])
        T.append(T[-1] @ A)

    A5 = np.eye(4)
    A5[:3, 3] = [L6-L4, L5, 0.]
    T.append(T[-1] @ A5)
    return np.array(T)


def IK(x, y, z, Theta=0., elbow=-1):
    """Stylus target in mm; returns 4 joint angles in radians."""
    if elbow not in (-1, 1):
        raise ValueError('elbow skal vaere -1 eller +1')
    r = np.hypot(x, y)
    if r < 1e-12:
        raise ValueError('Basevinklen er ubestemt ved x=y=0')
    rw = r - L4*np.cos(Theta)
    hw = z - L1 - L4*np.sin(Theta)
    C3 = (rw**2 + hw**2 - L2**2 - L3**2)/(2*L2*L3)
    if abs(C3) > 1 + 1e-12:
        raise ValueError('Punktet kan ikke naas')

    C3 = np.clip(C3, -1., 1.)
    S3 = elbow*np.sqrt(max(0., 1-C3**2))
    theta1 = np.arctan2(y, x)
    theta3 = np.arctan2(S3, C3)
    theta2 = np.arctan2(hw, rw) - np.arctan2(L3*S3, L2+L3*C3)
    theta4 = Theta - theta2 - theta3
    return np.array([theta1, theta2-np.pi/2, theta3, theta4])


def J(q, endpoint=4):
    """6x4 geometric Jacobian for stylus (4) or camera mount (5)."""
    if endpoint not in (4, 5):
        raise ValueError('endpoint skal være 4 eller 5')
    T = FK(q)
    end_position = T[endpoint, :3, 3]
    columns = []
    for i in range(4):
        z_previous = T[i, :3, 2]
        o_previous = T[i, :3, 3]
        linear = np.cross(z_previous, end_position-o_previous)
        angular = z_previous
        columns.append(np.r_[linear, angular])
    return np.column_stack(columns)


def task_jacobian(q):
    return np.vstack((J(q)[:3, :], [0., 1., 1., 1.]))


if __name__ == '__main__':
    q = np.zeros(4)
    T = FK(q)
    print('Stylus O4:', T[4, :3, 3], 'mm')
    print('Camera O5:', T[5, :3, 3], 'mm')

    target = np.array([150., 0., 120.])
    q = IK(*target)
    result = FK(q)[4, :3, 3]
    print('Target:', target)
    print('Joint angles [deg]:', np.degrees(q))
    print('FK check:', result)
    print('Error [mm]:', np.linalg.norm(result-target))
