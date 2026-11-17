import numpy as np

# Robot communication is deliberately disabled until servo settings
# and the hardware setup have been checked in the lab.

dry_run = True


def move(q):
    """Show a joint target. Does not send motor commands yet."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError('Expected 4 finite joint angles in radians')

    if dry_run:
        print('TEST only - angles [deg]:', np.round(np.degrees(q), 2))
        return

    raise RuntimeError(
        'Motor control not configured: verify servo IDs, offsets, '
        'directions, limits and connection before enabling movement.'
    )


if __name__ == '__main__':
    move(np.zeros(4))
