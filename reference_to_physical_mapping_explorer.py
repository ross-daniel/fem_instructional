#!/usr/bin/env python3
"""Interactive curved-quadrilateral geometry and H(curl) Piola-map explorer.

Requirements: numpy, matplotlib. Run with an interactive GUI backend.

Click anywhere in the LEFT (reference) plot to select the interpolation-grid
cell containing that point. The same curved patch is highlighted on the right.
Reference and physical vectors are shown side by side.

Geometry uses a degree <= 3 polynomial mapping, exactly representable by the
4th-order tensor-product Lagrange interpolation nodes shown in the plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons, Button
from matplotlib.patches import Polygon

GEOM_ORDER = 4                 # Five nodes in each parametric direction
NODES = np.linspace(-1.0, 1.0, GEOM_ORDER + 1)
COLORS = dict(grid='#8494a7', patch='#f3b23c', point='#d54445',
              au='#2264a7', av='#9c49ad', bu='#0f8a67', bv='#db713b', field='#222222')


def F(u, v):
    """Smooth curvilinear physical map (u,v) -> (x,y)."""
    u, v = np.asarray(u), np.asarray(v)
    x = u + 0.22 * (1 - v*v) + 0.12*u*v + 0.08*u*u*v
    y = v + 0.18 * (1 - u*u) + 0.10*u*v + 0.07*u*v*v
    return np.array([x, y])


def jacobian(u, v):
    """J = d(x,y) / d(u,v); columns are physical covariant tangents."""
    return np.array([[1 + 0.12*v + 0.16*u*v, -0.44*v + 0.12*u + 0.08*u*u],
                     [-0.36*u + 0.10*v + 0.07*v*v, 1 + 0.10*u + 0.14*u*v]])


def ref_basis(u, v, which):
    """Illustrative reference H(curl) vector modes on [-1,1]^2."""
    if which == 'Bottom edge (u)':
        return np.array([0.5*(1-v), np.zeros_like(u)])
    if which == 'Left edge (v)':
        return np.array([np.zeros_like(u), 0.5*(1-u)])
    # Bubble has zero tangential trace on every side.
    return np.array([u*(1-v*v), np.zeros_like(u)])


def mapped_basis(u, v, which):
    """Covariant Piola transform N_phys = J^{-T} N_ref."""
    b = ref_basis(u, v, which)
    j = jacobian(u, v)
    det = j[0,0]*j[1,1] - j[0,1]*j[1,0]
    return np.array([(j[1,1]*b[0]-j[1,0]*b[1])/det,
                     (-j[0,1]*b[0]+j[0,0]*b[1])/det])


def cell_polygon(i, j, samples=23):
    """Reference cell boundary, densely sampled to show curved physical edges."""
    u0, u1 = NODES[i], NODES[i+1]
    v0, v1 = NODES[j], NODES[j+1]
    s = np.linspace(0, 1, samples)
    u = np.r_[u0+(u1-u0)*s, np.full(samples, u1),
              u1-(u1-u0)*s, np.full(samples, u0)]
    v = np.r_[np.full(samples, v0), v0+(v1-v0)*s,
              np.full(samples, v1), v1-(v1-v0)*s]
    return np.column_stack([u, v])


def vec_arrow(ax, anchor, vec, color, text, scale=0.25):
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return
    delta = vec * scale  # true coordinate direction, common scale for all vectors
    ax.annotate('', xy=anchor + delta, xytext=anchor,
                arrowprops={'arrowstyle': '-|>', 'lw': 2.6, 'color': color,
                            'mutation_scale': 15})
    ax.text(*(anchor + 1.14*delta), text, color=color, fontsize=11,
            weight='bold', ha='center', va='center',
            bbox={'facecolor': 'white', 'alpha': .76, 'edgecolor': 'none', 'pad': 1.1})


class Explorer:
    def __init__(self):
        self.uv = np.array([0.18, -0.26])
        self.which = 'Bottom edge (u)'
        self.draw_basis = True
        self.draw_tangents = True
        self.show_linearized = True
        self.fig = plt.figure(figsize=(14.8, 8.2))
        self.fig.suptitle('Curved-element mapping: reference to physical coordinates',
                          fontsize=17, weight='bold', y=.976)
        self.ax_ref = self.fig.add_axes([.055, .205, .39, .67])
        self.ax_phys = self.fig.add_axes([.545, .205, .39, .67])
        self.ax_basis = self.fig.add_axes([.07, .045, .225, .115])
        self.ax_checks = self.fig.add_axes([.34, .035, .31, .135])
        self.ax_reset = self.fig.add_axes([.77, .065, .14, .065])
        self.radio = RadioButtons(self.ax_basis,
                                  ('Bottom edge (u)', 'Left edge (v)', 'Interior bubble'),
                                  active=0)
        self.radio.on_clicked(self.change_basis)
        self.check = CheckButtons(self.ax_checks,
                                  ('Show mapped H(curl) basis', 'Show covariant tangents',
                                   'Show Jacobian linearization'),
                                  (True, True, True))
        self.check.on_clicked(self.change_flags)
        self.reset = Button(self.ax_reset, 'Reset point')
        self.reset.on_clicked(self.reset_point)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.stats = self.fig.text(.5, .012, '', ha='center', va='bottom', fontsize=10)
        self.render()

    def change_basis(self, name):
        self.which = name
        self.render()

    def change_flags(self, name):
        if name == 'Show mapped H(curl) basis':
            self.draw_basis = not self.draw_basis
        elif name == 'Show covariant tangents':
            self.draw_tangents = not self.draw_tangents
        else:
            self.show_linearized = not self.show_linearized
        self.render()

    def reset_point(self, _):
        self.uv[:] = [0.18, -0.26]
        self.render()

    def on_click(self, event):
        if event.inaxes != self.ax_ref or event.xdata is None or event.ydata is None:
            return
        if not (-1 <= event.xdata <= 1 and -1 <= event.ydata <= 1):
            return
        self.uv[:] = [event.xdata, event.ydata]
        self.render()

    def draw_grid(self, ax, mapping):
        t = np.linspace(-1, 1, 150)
        for k in NODES:
            for u, v in [(t, np.full_like(t, k)), (np.full_like(t, k), t)]:
                xy = mapping(u, v)
                ax.plot(*xy, color=COLORS['grid'], lw=1.05, alpha=.9, zorder=2)
        for u in NODES:
            for v in NODES:
                x, y = mapping(u, v)
                ax.plot(x, y, marker='o', ms=4.5, mfc='white', mec='#596b7c', zorder=5)

    def draw_basis_arrows(self, ax, physical):
        uu, vv = np.meshgrid(np.linspace(-.84, .84, 9), np.linspace(-.84, .84, 9))
        coords = F(uu, vv) if physical else np.array([uu, vv])
        if physical:
            vectors = mapped_basis(uu, vv, self.which)
        else:
            vectors = ref_basis(uu, vv, self.which)
        mag = np.hypot(vectors[0], vectors[1])
        # Keep a shared, fixed magnitude convention so distortion is meaningful.
        keep = mag > 0.045
        ax.quiver(coords[0][keep], coords[1][keep], vectors[0][keep], vectors[1][keep],
                  color=COLORS['field'], angles='xy', scale_units='xy', scale=9.0,
                  width=.0035, alpha=.68, zorder=4, pivot='mid')

    def render(self):
        for ax in (self.ax_ref, self.ax_phys):
            ax.clear()
            ax.set_aspect('equal', adjustable='box')
            ax.grid(False)
            ax.set_facecolor('#fafbfc')
            ax.tick_params(labelsize=9)
        self.ax_ref.set_xlim(-1.27, 1.27)
        self.ax_ref.set_ylim(-1.29, 1.34)
        self.ax_phys.set_xlim(-1.35, 1.67)
        self.ax_phys.set_ylim(-1.33, 1.51)
        self.ax_ref.set_title(r'Reference element $\hat K$: $(u,v)$', fontsize=13)
        self.ax_phys.set_title(r'Physical element $K$: $(x,y)=F_K(u,v)$', fontsize=13)
        self.ax_ref.set_xlabel('$u$')
        self.ax_ref.set_ylabel('$v$')
        self.ax_phys.set_xlabel('$x$')
        self.ax_phys.set_ylabel('$y$')
        self.draw_grid(self.ax_ref, lambda u,v: np.array([u,v]))
        self.draw_grid(self.ax_phys, F)
        u, v = self.uv
        i = np.clip(np.searchsorted(NODES, u, side='right') - 1, 0, GEOM_ORDER-1)
        j = np.clip(np.searchsorted(NODES, v, side='right') - 1, 0, GEOM_ORDER-1)
        poly = cell_polygon(i,j)
        self.ax_ref.add_patch(Polygon(poly, closed=True, color=COLORS['patch'],
                                       alpha=.32, ec='#ac730b', lw=2, zorder=3))
        ppoly = F(poly[:,0], poly[:,1]).T
        self.ax_phys.add_patch(Polygon(ppoly, closed=True, color=COLORS['patch'],
                                        alpha=.32, ec='#ac730b', lw=2, zorder=3))
        if self.draw_basis:
            self.draw_basis_arrows(self.ax_ref, False)
            self.draw_basis_arrows(self.ax_phys, True)
        origin_ref = np.array([u,v])
        origin_phys = F(u,v)
        self.ax_ref.scatter([u], [v], c=COLORS['point'], edgecolors='white', s=105,
                            linewidth=1.5, zorder=9)
        self.ax_phys.scatter([origin_phys[0]], [origin_phys[1]], c=COLORS['point'],
                             edgecolors='white', s=105, linewidth=1.5, zorder=9)
        J = jacobian(u,v)
        au, av = J[:,0], J[:,1]  # covariant vectors
        contravariant = np.linalg.inv(J).T
        b_u, b_v = contravariant[:,0], contravariant[:,1]
        if self.draw_tangents:
            vec_arrow(self.ax_ref, origin_ref, np.array([1.,0.]), COLORS['au'], r'$\hat e_u$')
            vec_arrow(self.ax_ref, origin_ref, np.array([0.,1.]), COLORS['av'], r'$\hat e_v$')
            vec_arrow(self.ax_phys, origin_phys, au, COLORS['au'], r'$a_u$')
            vec_arrow(self.ax_phys, origin_phys, av, COLORS['av'], r'$a_v$')
        # Reciprocal/contravariant vectors in physical coordinates (not unit normalized).
        vec_arrow(self.ax_phys, origin_phys, b_u, COLORS['bu'], r"$a'_u$", scale=.38)
        vec_arrow(self.ax_phys, origin_phys, b_v, COLORS['bv'], r"$a'_v$", scale=.38)
        if self.show_linearized:
            # Locally approximate the selected grid cell by F(u,v)+J*[delta_u,delta_v].
            corners = np.array([[NODES[i],NODES[j]], [NODES[i+1],NODES[j]],
                                [NODES[i+1],NODES[j+1]], [NODES[i],NODES[j+1]]])
            linear = origin_phys[:,None] + J @ (corners - origin_ref).T
            line = np.column_stack([linear,linear[:,0]])
            self.ax_phys.plot(*line, ls='--', lw=1.75, color=COLORS['point'], zorder=8)
        det = np.linalg.det(J)
        self.stats.set_text(
            f'Click reference plot to select a point / interpolation cell   •   '
            f'(u,v)=({u:+.3f},{v:+.3f})   →   (x,y)=({origin_phys[0]:+.3f},{origin_phys[1]:+.3f})'
            f'   •   det(J)={det:.4f}   •   cell=({i+1},{j+1})')
        self.fig.canvas.draw_idle()


def main():
    print('Click on the reference element to select an interpolation-grid cell.')
    print("Blue/purple: physical tangent vectors a_u, a_v = J e_u, J e_v")
    print("Green/orange: reciprocal (contravariant) vectors a'_u, a'_v = J^{-T} e_u, J^{-T} e_v")
    print('The reciprocal vectors are NOT generally unit length.')
    print('Dashed red cell: first-order Jacobian approximation at clicked point.')
    Explorer()
    plt.show()


if __name__ == '__main__':
    main()
