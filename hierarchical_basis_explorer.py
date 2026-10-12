#!/usr/bin/env python3
"""Interactive hierarchical H1 / H(curl) basis explorer (quadrilateral).

Run: python hierarchical_basis_explorer.py
Dependencies: numpy, matplotlib. Use a desktop Matplotlib backend (QtAgg/TkAgg).

Illustrative tensor-product hierarchy, NOT a direct import of py-fem2d.
H1 scalar:       phi_ij(u,v) = L_i(u) L_j(v), 0<=i,j<=p.
H(curl) vector:  U_ij = [L'_{i+1}(u)L_j(v), 0], 0<=i<p, 0<=j<=p;
                 V_ij = [0,L_i(u)L'_{j+1}(v)], 0<=i<=p, 0<=j<p.
This spans Q_{p-1,p} x Q_{p,p-1} for the curl-conforming space.

L0=(1-t)/2, L1=(1+t)/2, Lk=t**k-1 if even, t**k-t if odd (k>=2).
Each function of k>=2 vanishes at both endpoints. These are simple
monomial hierarchical polynomials, not normalized/orthogonal Legendre modes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider, CheckButtons
from matplotlib.colors import TwoSlopeNorm


def L(k, t):
    t = np.asarray(t)
    if k == 0: return (1 - t) / 2
    if k == 1: return (1 + t) / 2
    return t**k - (1 if k % 2 == 0 else t)


def dL(k, t):
    t = np.asarray(t)
    if k == 0: return -np.ones_like(t) / 2
    if k == 1: return np.ones_like(t) / 2
    return k * t**(k - 1) - (0 if k % 2 == 0 else 1)


def category(space, direction, i, j):
    if space == 'H1':
        if i < 2 and j < 2: return 'VERTEX mode'
        if i >= 2 and j >= 2: return 'INTERIOR bubble'
        return 'EDGE mode'
    if direction == 'U':
        return 'EDGE mode (bottom/top)' if j < 2 else 'INTERIOR tangential bubble'
    return 'EDGE mode (left/right)' if i < 2 else 'INTERIOR tangential bubble'


def main():
    fig = plt.figure(figsize=(14, 8.2), constrained_layout=False)
    fig.patch.set_facecolor('#f7f9fc')
    # An axes is replaced when the function space changes.
    plot_rect = [0.075, 0.26, 0.53, 0.67]
    ax_main = None
    ax_tr = fig.add_axes([0.67, 0.43, 0.29, 0.40])
    ax_space = fig.add_axes([0.67, 0.87, 0.12, 0.095])
    ax_dir = fig.add_axes([0.83, 0.87, 0.12, 0.095])
    ax_p = fig.add_axes([0.13, 0.17, 0.45, 0.026])
    ax_i = fig.add_axes([0.13, 0.12, 0.45, 0.026])
    ax_j = fig.add_axes([0.13, 0.07, 0.45, 0.026])
    ax_a = fig.add_axes([0.13, 0.02, 0.45, 0.026])
    ax_check = fig.add_axes([0.71, 0.295, 0.23, 0.075])
    ax_space.set_facecolor('#f7f9fc'); ax_dir.set_facecolor('#f7f9fc')
    ax_check.set_facecolor('#f7f9fc')
    space_radio = RadioButtons(ax_space, ['H1', 'Hcurl'], active=0)
    dir_radio = RadioButtons(ax_dir, ['U', 'V'], active=0)
    p_slider = Slider(ax_p, 'Order p', 1, 6, valinit=3, valstep=1)
    i_slider = Slider(ax_i, 'Index i', 0, 6, valinit=2, valstep=1)
    j_slider = Slider(ax_j, 'Index j', 0, 6, valinit=0, valstep=1)
    a_slider = Slider(ax_a, 'Amplitude', -2, 2, valinit=1.0, valstep=0.05)
    trace_toggle = CheckButtons(ax_check, ['Show traces'], [True])
    status = fig.text(0.67, 0.21, '', fontsize=11, va='top', linespacing=1.5)
    fig.suptitle('Hierarchical Finite Element Basis Explorer', fontsize=17, fontweight='bold', y=0.995)
    state = {'space': 'H1', 'direction': 'U', 'traces': True}
    u = np.linspace(-1, 1, 45); v = np.linspace(-1, 1, 45)
    U, V = np.meshgrid(u, v)

    def redraw(_=None):
        nonlocal ax_main
        p = int(p_slider.val)
        space = state['space']; direction = state['direction']
        max_i = p - 1 if space == 'Hcurl' and direction == 'U' else p
        max_j = p - 1 if space == 'Hcurl' and direction == 'V' else p
        i = min(int(i_slider.val), max_i)
        j = min(int(j_slider.val), max_j)
        # Set sliders back into range when order/direction changes. Values
        # beyond active range are clamped during computation to avoid crashes.
        a = float(a_slider.val)
        if ax_main is not None:
            ax_main.remove()
        ax_main = fig.add_axes(plot_rect, projection='3d' if space == 'H1' else None)
        ax_tr.clear()
        if space == 'H1':
            Z = a * L(i, U) * L(j, V)
            bound = max(0.5, float(np.max(np.abs(Z))))
            ax_main.plot_surface(U, V, Z, cmap='coolwarm', norm=TwoSlopeNorm(vcenter=0, vmin=-bound, vmax=bound),
                                 edgecolor='none', antialiased=True, alpha=.97)
            ax_main.set(xlabel='u', ylabel='v', zlabel='Scalar value', xlim=(-1,1), ylim=(-1,1), zlim=(-1.2*bound,1.2*bound))
            ax_main.view_init(elev=28, azim=-60)
            ax_main.set_title(f'H¹:  a · L{i}(u) L{j}(v)', fontsize=13, pad=12)
            field = lambda uu,vv: a * L(i,uu)*L(j,vv)
            trace_label = 'Scalar trace  φ|edge'
        else:
            if direction == 'U':
                Fx = a * dL(i+1,U) * L(j,V)
                Fy = np.zeros_like(Fx)
                field = lambda uu,vv: (a*dL(i+1,uu)*L(j,vv), np.zeros_like(np.asarray(uu)+np.asarray(vv),dtype=float))
                formula = f'[L\u2032{i+1}(u) L{j}(v), 0]'
            else:
                Fx = np.zeros_like(U)
                Fy = a * L(i,U) * dL(j+1,V)
                field = lambda uu,vv: (np.zeros_like(np.asarray(uu)+np.asarray(vv),dtype=float), a*L(i,uu)*dL(j+1,vv))
                formula = f'[0, L{i}(u) L\u2032{j+1}(v)]'
            xx = np.linspace(-.92,.92,17); yy = np.linspace(-.92,.92,17)
            XX, YY = np.meshgrid(xx, yy)
            VX,VY = field(XX,YY)
            strength = np.hypot(VX,VY)
            m = np.max(strength)
            ax_main.quiver(XX, YY, VX, VY, strength, cmap='viridis', angles='xy',
                           scale_units='xy', scale=max(m*9, 1e-10), pivot='mid', width=.005)
            ax_main.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1], 'k-', lw=1.5)
            ax_main.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15), xlabel='u', ylabel='v')
            ax_main.set_aspect('equal')
            ax_main.set_title(f'H(curl):  a · {formula}', fontsize=13)
            trace_label = 'Tangential trace  E·t'
        cat = category(space, direction, i, j)
        s = np.linspace(-1,1,201)
        edges = [('bottom',s,-np.ones_like(s)),('right',np.ones_like(s),s),
                 ('top',s,np.ones_like(s)),('left',-np.ones_like(s),s)]
        for name, uu, vv in edges:
            if space == 'H1':
                values = field(uu,vv)
            else:
                fx, fy = field(uu,vv)
                values = fx if name in ('bottom','top') else fy
            if state['traces']:
                ax_tr.plot(s, values, lw=2, label=name)
        ax_tr.axhline(0, lw=.7, color='gray')
        ax_tr.set(xlim=(-1,1),xlabel='Edge parameter s',ylabel=trace_label)
        ax_tr.set_title('Boundary traces (local +u / +v tangent)')
        ax_tr.grid(alpha=.22)
        if state['traces']:
            ax_tr.legend(loc='upper right', fontsize=9, ncol=2)
        ax_tr.set_ylim(-max(.2,abs(a)*3), max(.2,abs(a)*3))
        status.set_text(f'Function: {cat}\nPolynomial order: p={p}\nSelected indices: ({i}, {j})\n'
                        'Edge modes need shared global DoFs.\nInterior modes have zero relevant trace.')
        if int(i_slider.val) != i or int(j_slider.val) != j:
            status.set_text(status.get_text()+'\n(indices clamped to active p)')
        fig.canvas.draw_idle()

    def change_space(label):
        state['space'] = label
        redraw()
    def change_dir(label):
        state['direction'] = label
        redraw()
    def change_trace(_):
        state['traces'] = not state['traces']
        redraw()
    space_radio.on_clicked(change_space)
    dir_radio.on_clicked(change_dir)
    trace_toggle.on_clicked(change_trace)
    for slider in (p_slider, i_slider, j_slider, a_slider): slider.on_changed(redraw)
    redraw()
    plt.show()


if __name__ == '__main__':
    main()
