# 3D-simulering af CourseBot: fire røde Smarties, helt uden motorkommandoer.

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from kinematics import FK, IK
from trajectory import make_traj, sample_traj

# Testdata fra 11b.py: Smarties i robotbasens koordinater [mm]
# De kommer fra det FIKTIVE kamera i testen, ikke en virkelig måling.
smarties = np.array([
    [43.45, -3.79, 100.00],
    [95.58, -38.45, 100.00],
    [147.20, -70.84, 100.00],
    [194.26, -7.13, 100.00],
])

z_plane = 100.0            # Selvvalgt testhøjde [mm]
frihoejde = 20.0          # Højde over Smartien før og efter berøring [mm]
tid_pr_bevaegelse = 1.2  # Varighed af hvert ledsegment [s]
dt = 0.08                 # Tidsinterval for animationen [s]

# Kun simulerede ledvinkler. Servoer kan aldrig aktiveres af denne fil.
q_start = IK(150., 0., 120.)


def lav_bane(smarties):

    frames = [q_start.copy()]
    labels = ['Startposition']

    # Stylusspidsens startposition fra FK
    p_forrige = FK(q_start)[4, :3, 3]

    for nr, p in enumerate(smarties, start=1):

        p_over = p + np.array([0., 0., frihoejde])

        for beskrivelse, maal in [
            ('over', p_over),
            ('berør', p),
            ('løft', p_over)
        ]:

            n = round(tid_pr_bevaegelse / dt)

            for i in range(1, n + 1):

                t = i / n

                # Glat bevægelse med 5.-gradspolynomium
                s = 10*t**3 - 15*t**4 + 6*t**5

                # Interpolation i x, y og z
                pos = p_forrige + s * (maal - p_forrige)

                # Beregn ledvinkler til hvert punkt
                q = IK(*pos)

                frames.append(q)
                labels.append(f'Smartie {nr}: {beskrivelse}')

            # Kontroller den beregnede slutposition
            p_check = FK(frames[-1])[4, :3, 3]
            fejl = np.linalg.norm(p_check - maal)

            print(f'Smartie {nr}, {beskrivelse}: fejl = {fejl:.6f} mm')

            p_forrige = maal.copy()

    return np.array(frames), labels


def simuler(smarties, z_plane, gem_gif=False, vis=True):
    qs, labels = lav_bane(smarties)
    pos = np.array([FK(q)[4, :3, 3] for q in qs])

    print('Fire Smarties - simulationsdata [mm]:')
    for nr, p in enumerate(smarties, start=1):
        print(f'  Smartie {nr}: {np.round(p, 2)}')
    print(f'Antal animationstrin: {len(qs)}')
    print('OBS: Ledinterpolation. Stylus bevæger sig ikke nødvendigvis lodret.')
    print('OBS: Ingen kontrol af kollisioner, motorgrænser eller reel kalibrering.')

    fig = plt.figure(figsize=(10.2, 7.1))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle('CourseBot – simulering af Smarties (testkoordinater)', fontsize=13)

    # Vis det vandrette målplan og Smarties
    grid_x, grid_y = np.meshgrid([0, 230], [-125, 100])
    ax.plot_surface(grid_x, grid_y, np.full_like(grid_x, z_plane),
                    alpha=0.10, color='gray', shade=False)
    ax.scatter(smarties[:, 0], smarties[:, 1], smarties[:, 2],
               color='crimson', s=100, depthshade=False, label='Smarties')
    for i, p in enumerate(smarties, 1):
        ax.text(p[0]+3, p[1]+3, p[2]+3, str(i), fontsize=10)

    # Robotlinjer fra FK: base, led og stylus
    (arm,) = ax.plot([], [], [], '-o', linewidth=4.5, markersize=6,
                     color='#2463a1', markerfacecolor='white', label='Robotarm')
    (camera,) = ax.plot([], [], [], 's', markersize=8, color='#00a6a6', label='Kamera (frame 5)')
    (trail,) = ax.plot([], [], [], '-', linewidth=1.4, color='darkorange', alpha=.75,
                       label='Stylusens bane')
    (tip,) = ax.plot([], [], [], 'o', color='gold', markersize=8)
    message = ax.text2D(0.02, 0.96, '', transform=ax.transAxes, fontsize=11,
                        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': .85})

    ax.set_xlim(-80, 260)
    ax.set_ylim(-150, 150)
    ax.set_zlim(0, 310)
    ax.set_box_aspect((340, 300, 310))
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.view_init(elev=25, azim=-57)
    ax.legend(loc='upper right', fontsize=9)

    def update(k):
        T = FK(qs[k])
        # Ledpositioner fra basen til stylus (frame 4)
        joints = T[:5, :3, 3]
        arm.set_data(joints[:, 0], joints[:, 1])
        arm.set_3d_properties(joints[:, 2])

        cam_pos = T[5, :3, 3]
        camera.set_data([cam_pos[0]], [cam_pos[1]])
        camera.set_3d_properties([cam_pos[2]])

        trail.set_data(pos[:k+1, 0], pos[:k+1, 1])
        trail.set_3d_properties(pos[:k+1, 2])
        tip.set_data([pos[k, 0]], [pos[k, 1]])
        tip.set_3d_properties([pos[k, 2]])
        message.set_text(f'{labels[k]}\nTid: {k*dt:.1f} s')
        return arm, camera, trail, tip, message

    anim = FuncAnimation(fig, update, frames=len(qs), interval=dt*1000,
                         blit=False, repeat=True)
    if gem_gif:
        navn = 'smarties_simulering.gif'
        # Reducér antal GIF-frames uden at ændre planlagte bevægelser
        hop = max(1, int(np.ceil(len(qs) / 130)))
        gif_anim = FuncAnimation(fig, update, frames=range(0, len(qs), hop),
                                 interval=dt*1000*hop, blit=False, repeat=True)
        gif_anim.save(navn, writer=PillowWriter(fps=round(1/(dt*hop))), dpi=85)
        print(f'GIF gemt som: {navn}')
        plt.close(fig)
    else:
        plt.show()
    return anim


if __name__ == '__main__':
    simuler(gem_gif='--gif' in sys.argv)
