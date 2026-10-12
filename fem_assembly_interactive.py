"""Interactive finite-element DoF connectivity and global matrix sparsity.

Run: python fem_assembly_interactive.py
Requires: numpy, matplotlib and a GUI-capable Matplotlib backend.
Click a mesh node or a matrix row/column to explore its connections.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Global nodes, numbered from 1, arranged in a 3x3 grid.
coords = {1:(0,2), 2:(1,2), 3:(2,2),
          4:(0,1), 5:(1,1), 6:(2,1),
          7:(0,0), 8:(1,0), 9:(2,0)}
# Each element lists its four global DoFs in an arbitrary consistent order.
elements = {1:(1,2,4,5), 2:(2,3,5,6),
            3:(4,5,7,8), 4:(5,6,8,9)}

# Symbolic (potential) nonzero pattern: element-wise full matrices.
adjacency = np.zeros((9,9), dtype=bool)
for dofs in elements.values():
    indices = np.array(dofs)-1
    adjacency[np.ix_(indices, indices)] = True

fig, (ax_mesh, ax_mat) = plt.subplots(1, 2, figsize=(11, 5.5))
fig.suptitle('FEM global assembly: mesh connectivity → matrix sparsity', fontsize=15)
fig.subplots_adjust(bottom=.23, wspace=.36, top=.86)
status = fig.text(.5, .105, '', ha='center', va='center', fontsize=11)
help_text = fig.text(.5, .055, 'Click a node in the mesh or a row/column in the matrix to select a DoF.',
                     ha='center', color='dimgray', fontsize=10)
selected = 5


def redraw():
    ax_mesh.clear()
    ax_mat.clear()
    memberships = [e for e, dofs in elements.items() if selected in dofs]
    coupled = np.flatnonzero(adjacency[selected-1]).tolist()
    for eid, dofs in elements.items():
        xy = np.array([coords[d] for d in dofs], dtype=float)
        xmin, ymin = xy.min(axis=0)
        rect = Rectangle((xmin,ymin), 1, 1,
                         facecolor='mediumseagreen' if eid in memberships else 'steelblue',
                         alpha=.23 if eid in memberships else .11,
                         edgecolor='seagreen' if eid in memberships else 'slategray',
                         linewidth=2.5 if eid in memberships else 1.5)
        ax_mesh.add_patch(rect)
        ax_mesh.text(xmin+.5,ymin+.5, f'e{eid}', ha='center',va='center', fontsize=13)
    for node,(x,y) in coords.items():
        chosen = node==selected
        ax_mesh.scatter(x,y,s=210 if chosen else 125, zorder=3,
                        color='seagreen' if chosen else ('cornflowerblue' if node-1 in coupled else 'gray'),
                        edgecolor='black', linewidth=.8)
        ax_mesh.annotate(str(node), (x,y), xytext=(0,13),textcoords='offset points',
                         ha='center',fontsize=12,fontweight='bold' if chosen else 'normal')
    ax_mesh.set(xlim=(-.27,2.27),ylim=(-.25,2.35),aspect='equal',title='2×2 quadrilateral mesh')
    ax_mesh.axis('off')

    # 0: no common element, 1: possible coupling, 2: selected row/column.
    cells = adjacency.astype(int)
    cells[selected-1, adjacency[selected-1]] = 2
    cells[adjacency[:,selected-1], selected-1] = 2
    from matplotlib.colors import ListedColormap
    ax_mat.imshow(cells, cmap=ListedColormap(['#f0f0f0','#779fc6','#2aa476']),
                  vmin=0,vmax=2, interpolation='nearest')
    ax_mat.set_xticks(range(9), labels=range(1,10))
    ax_mat.set_yticks(range(9), labels=range(1,10))
    ax_mat.set_xlabel('Global DoF j')
    ax_mat.set_ylabel('Global DoF i')
    ax_mat.set_title('Potential nonzero pattern of K')
    ax_mat.set_xticks(np.arange(-.5,9,1),minor=True)
    ax_mat.set_yticks(np.arange(-.5,9,1),minor=True)
    ax_mat.grid(which='minor', color='white', linewidth=1.4)
    ax_mat.tick_params(which='minor',bottom=False,left=False)
    status.set_text(f'DoF {selected} belongs to elements {memberships}; '
                    f'potentially couples to DoFs {[i+1 for i in coupled]}')
    fig.canvas.draw_idle()


def on_click(event):
    global selected
    if event.inaxes is ax_mesh and event.xdata is not None:
        nearest = min(coords, key=lambda n: (coords[n][0]-event.xdata)**2 +
                      (coords[n][1]-event.ydata)**2)
        x,y=coords[nearest]
        if (x-event.xdata)**2+(y-event.ydata)**2 < .14:
            selected = nearest
            redraw()
    elif event.inaxes is ax_mat and event.xdata is not None and event.ydata is not None:
        # Click a row number or any cell: pick its row (global DoF i).
        idx=int(np.floor(event.ydata+.5))
        if 0 <= idx < 9:
            selected=idx+1
            redraw()

fig.canvas.mpl_connect('button_press_event', on_click)
redraw()
plt.show()
