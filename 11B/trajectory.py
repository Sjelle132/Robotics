import numpy as np

# Fifth-order polynomial from problem 6

def basis(t):
    return np.array([1., t, t**2, t**3, t**4, t**5])


def d1(t):
    return np.array([0., 1., 2*t, 3*t**2, 4*t**3, 5*t**4])


def d2(t):
    return np.array([0., 0., 2., 6*t, 12*t**2, 20*t**3])


def make_traj(q0, q1, duration, v0=None, v1=None, a0=None, a1=None):
    """Fit a quintic between two joint configurations (radians)."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if q0.shape != (4,) or q1.shape != (4,):
        raise ValueError('q0 and q1 must have 4 joint angles')
    if duration <= 0:
        raise ValueError('duration must be positive')

    def value(x):
        if x is None:
            return np.zeros(4)
        x = np.asarray(x, dtype=float)
        if x.shape != (4,):
            raise ValueError('Velocity and acceleration need 4 values')
        return x

    P = np.column_stack((
        basis(0.), d1(0.), d2(0.),
        basis(duration), d1(duration), d2(duration)
    ))
    bc = np.column_stack((q0, value(v0), value(a0), q1, value(v1), value(a1)))
    return np.linalg.solve(P.T, bc.T).T


def eval_traj(coeff, t):
    """Returns joint position, velocity and acceleration."""
    return coeff @ basis(t), coeff @ d1(t), coeff @ d2(t)


def sample_traj(coeff, duration, dt=0.05):
    """Sample trajectory for plotting or offline checks, not motor control."""
    if duration <= 0 or dt <= 0:
        raise ValueError('duration and dt must be positive')
    times = np.linspace(0., duration, max(2, int(np.ceil(duration / dt)) + 1))
    qs = np.array([eval_traj(coeff, t)[0] for t in times])
    return times, qs


if __name__ == '__main__':
    from kinematics import IK, FK

    q0 = IK(150., 0., 120.)
    q1 = IK(150., 20., 120.)
    coeff = make_traj(q0, q1, duration=2.)
    t, qs = sample_traj(coeff, duration=2.)
    print('Samples:', len(t))
    print('Start [mm]:', FK(qs[0])[4, :3, 3])
    print('End [mm]:', FK(qs[-1])[4, :3, 3])
