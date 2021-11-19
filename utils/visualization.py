import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def find_bounds(worldpos):
    max_x, max_y, max_z = np.amax(worldpos, axis=(0, 1))
    min_x, min_y, min_z = np.amin(worldpos, axis=(0, 1))
    return (min_x, max_x), (min_y, max_y), (min_z, max_z)


def render_animation(sample, output, fps=60, contact_check=False, contact_state=None):
    worldpos = sample['trajectory']
    edges = sample['edges']

    if contact_check:
        contact_l, contact_r, lfoot_idx, rfoot_idx = contact_state
    plt.ioff()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20., azim=30)
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = find_bounds(worldpos)
    # We do seemingly weird assignments of bounds here, but it's due to the conventions
    ax.set_xlim(left=x_max, right=x_min)
    ax.set_ylim(bottom=z_min, top=z_max)
    ax.set_zlim(bottom=-y_min, top=-y_max)
    x = []
    y = []
    z = []
    facecolor = ['g']*worldpos.shape[1]
    sc = ax.scatter(x, z, y, color='g', zdir='y', marker="o", alpha=1.0)
    ax.set_aspect('auto')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5
    lines = []
    initialized = False
    fig.tight_layout()

    def animate(frame):
        nonlocal initialized
        x_coord = worldpos[frame, :, 0]
        y_coord = worldpos[frame, :, 1]
        z_coord = worldpos[frame, :, 2]
        sc._offsets3d = (x_coord, z_coord, -y_coord)
        if contact_check:
            if contact_l[frame]:
                facecolor[lfoot_idx] = 'b'
            else:
                facecolor[lfoot_idx] = 'r'
            if contact_r[frame]:
                facecolor[rfoot_idx] = 'b'
            else:
                facecolor[rfoot_idx] = 'r'
            sc._facecolor3d = facecolor

        if not initialized:
            for edge in edges[1:]:
                lines.append(ax.plot([worldpos[frame][edge[0]][0], worldpos[frame][edge[1]][0]],
                                     [-worldpos[frame][edge[0]][1], -worldpos[frame][edge[1]][1]],
                                     [worldpos[frame][edge[0]][2], worldpos[frame][edge[1]][2]], color='r', zdir='y'))
        else:
            for i, edge in enumerate(edges[1:]):
                lines[i][0].set_xdata([worldpos[frame][edge[0]][0], worldpos[frame][edge[1]][0]])
                lines[i][0].set_ydata([-worldpos[frame][edge[0]][1], -worldpos[frame][edge[1]][1]])
                lines[i][0].set_3d_properties([worldpos[frame][edge[0]][2], worldpos[frame][edge[1]][2]], zdir='y')
        initialized = True

    anim = FuncAnimation(fig, animate, frames=worldpos.shape[0], interval=1000 / fps, repeat=False)
    Writer = writers['ffmpeg']
    writer = Writer(fps=fps, metadata={}, bitrate=1000)
    anim.save(output, writer=writer)
    plt.close()
