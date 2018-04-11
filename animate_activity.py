import sys
import numpy as np
import pylab as pl
from matplotlib import animation

def animate_image(ax, h, *args, **kwargs):

    # First set up the figure, the axis, and the plot element we want to animate
    im = ax.imshow(h[0], *args, **kwargs)

    # initialization function: plot the background of each frame
    def init():
        im.set_array([])
        return im,

    def animate(ii):
        im.set_array(h[ii])
        ax.set_title('%s'%ii)
        pl.draw()
        return im,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(ax.get_figure(), animate, frames=len(h), interval=50, blit=True)

    return anim

def animate_activity(filenames):

    data = []
    for filename in filenames:
        d = np.loadtxt(filename)
        data.append(d)
    gids, ts = np.concatenate(data).T

    nrow = ncol = 120
    npop = nrow * ncol
    offset = 1

    idx = gids - offset < npop
    gids, ts = gids[idx], ts[idx]

    ts_bins = np.arange(500., 1500., 10.)
    h = np.histogram2d(ts, gids - offset, bins=[ts_bins, range(npop + 1)])[0]
    hh = h.reshape(-1, nrow, ncol)

    fig, ax = pl.subplots(1)
    a = animate_image(ax, hh, vmin=0, vmax=np.max(hh))
    a.save('%s.mp4' %filenames[0].split('.')[0], fps=10, extra_args=['-vcodec', 'libx264'])
    pl.show()


if __name__ == '__main__':
    argv = sys.argv
    animate_activity(argv[1:])
