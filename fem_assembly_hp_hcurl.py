"""Interactive FEM assembly explorer: H1 versus H(curl), p=2,3,4.

pip install numpy matplotlib
python fem_assembly_hp_hcurl.py

Two-by-two conforming quadrilateral mesh. H1 uses tensor-product Q_p Lagrange
nodal DoFs. H(curl) uses first-family Nedelec tensor-product spaces
Q_(p-1,p) x Q_(p,p-1). Edge DoFs are shared, while interior DoFs are local.
We show *structural* coupling (two DoFs belonging to the same element), not
numerically evaluated Maxwell stiffness/mass entries. Coordinates/arrows for
H(curl) DoFs are representative glyphs, NOT nodal field values.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RadioButtons
from matplotlib.colors import ListedColormap

COLORS = {'edge':'#3182bd', 'interior_x':'#b07d27', 'interior_y':'#a24d8e',
          'node':'#4c78a8', 'active':'#16876a'}


def build(p, space):
    pos, kinds, directions, element_dofs = [], [], [], []
    def add(x,y,kind,vec=None):
        pos.append((x,y)); kinds.append(kind); directions.append(vec)
        return len(pos)-1
    if space=='H1':
        # Each node is a true shared global Q_p nodal DoF.
        ids={}
        for j in range(2*p+1):
            for i in range(2*p+1):
                ids[i,j] = add(i/p,j/p,'node')
        for ey in range(2):
            for ex in range(2):
                element_dofs.append([ids[ex*p+i,ey*p+j]
                                    for j in range(p+1) for i in range(p+1)])
    else:
        # H(curl) first-family on quads: tangential edge moments (p/edge),
        # interior x-directed moments p(p-1) and y-directed p(p-1).
        h, v = {}, {}
        for row in range(3):
            for col in range(2):
                for k in range(p):
                    h[col,row,k] = add(col+(k+.5)/p,row,'edge',(1,0))
        for row in range(2):
            for col in range(3):
                for k in range(p):
                    v[col,row,k] = add(col,row+(k+.5)/p,'edge',(0,1))
        for ey in range(2):
            for ex in range(2):
                dofs=[]
                for k in range(p):
                    dofs.extend((h[ex,ey,k], h[ex,ey+1,k],
                                 v[ex,ey,k], v[ex+1,ey,k]))
                # Representative positions for element-interior DOF functionals.
                # x component: p along x, p-1 along y; vice versa for y.
                for j in range(1,p):
                    for i in range(p):
                        dofs.append(add(ex+(i+.5)/p,ey+j/p,
                                        'interior_x',(1,0)))
                for j in range(p):
                    for i in range(1,p):
                        dofs.append(add(ex+i/p,ey+(j+.5)/p,
                                        'interior_y',(0,1)))
                element_dofs.append(dofs)
    n=len(pos)
    pattern=np.zeros((n,n),dtype=bool)
    for dofs in element_dofs:
        ix=np.asarray(dofs)
        pattern[np.ix_(ix,ix)] = True
    return np.asarray(pos),kinds,directions,element_dofs,pattern


fig=plt.figure(figsize=(13.3,7.5))
fig.suptitle('Global FEM matrix assembly: $H^1$ vs. $H(\\mathrm{curl})$',fontsize=16)
ax_mesh=fig.add_axes([.05,.22,.41,.63])
ax_mat=fig.add_axes([.52,.22,.41,.63])
ax_space=fig.add_axes([.08,.075,.17,.10])
ax_p=fig.add_axes([.30,.075,.17,.10])
space_radio=RadioButtons(ax_space,['H1','Hcurl'],active=0)
p_radio=RadioButtons(ax_p,['2','3','4'],active=0)
ax_space.set_title('Function space',fontsize=10,loc='left')
ax_p.set_title('Polynomial order p',fontsize=10,loc='left')
info=fig.text(.53,.115,'',fontsize=10,va='center')
info2=fig.text(.53,.076,'',fontsize=9,color='dimgray',va='center')
fig.text(.5,.022,'Click a DoF glyph (left) or a matrix row (right) to highlight its element support and potential couplings.',
         ha='center',fontsize=10)
state={'space':'H1','p':2,'selected':None}
data={}


def redraw():
    p=state['p']; space=state['space']; sel=state['selected']
    pos,kinds,dirs,elems,pat=build(p,space)
    data.update(pos=pos,kinds=kinds,dirs=dirs,elems=elems,pat=pat)
    n=len(pos)
    if sel is None or sel>=n:
        sel=next((i for i,(x,y) in enumerate(pos) if .7<=x<=1.3 and .7<=y<=1.3),0)
        state['selected']=sel
    memberships=[i+1 for i,dofs in enumerate(elems) if sel in dofs]
    coupled=pat[sel]
    ax_mesh.clear(); ax_mat.clear()
    for e,(ex,ey) in enumerate([(0,0),(1,0),(0,1),(1,1)]):
        active=(e+1) in memberships
        ax_mesh.add_patch(Rectangle((ex,ey),1,1,
                          facecolor='#16876a' if active else '#6688aa',
                          alpha=.16 if active else .055,
                          edgecolor='#16876a' if active else '#555555',
                          linewidth=2.5 if active else 1.1))
        ax_mesh.text(ex+.5,ey+.5,f'e{e+1}',color='#555555',alpha=.75,
                     fontsize=11,ha='center',va='center',zorder=0)
    # plot small DoF glyphs in global number order; draw directional arrows
    for k,((x,y),kind,vec) in enumerate(zip(pos,kinds,dirs)):
        is_sel=k==sel; clr=COLORS['active'] if is_sel else COLORS[kind]
        alpha=1 if is_sel or coupled[k] else .30
        ax_mesh.scatter([x],[y],s=78 if is_sel else (29 if space=='H1' else 19),
                        color=clr,alpha=alpha,zorder=3,
                        edgecolors='black' if is_sel else 'none',linewidths=.65)
        if vec is not None:
            dx,dy=np.asarray(vec)*(.11 if is_sel else .075)
            ax_mesh.arrow(x-dx/2,y-dy/2,dx,dy,head_width=.026,
                          head_length=.027,length_includes_head=True,
                          color=clr,alpha=alpha,zorder=4)
    x,y=pos[sel]
    ax_mesh.annotate(f'#{sel+1}',(x,y),xytext=(8,9),textcoords='offset points',
                     fontsize=10,fontweight='bold',color=COLORS['active'])
    ax_mesh.set_xlim(-.20,2.21);ax_mesh.set_ylim(-.20,2.20)
    ax_mesh.set_aspect('equal');ax_mesh.axis('off')
    ax_mesh.set_title(f'2×2 quadrilateral mesh — {n} global DoFs',fontsize=12)
    # 0 structural zero, 1 possible coupling, 2 highlighted selected row/col
    pic=pat.astype(np.uint8)
    pic[sel,coupled]=2
    pic[coupled,sel]=2
    ax_mat.imshow(pic,interpolation='nearest',origin='upper',aspect='equal',
                  cmap=ListedColormap(['#f1f3f5','#799fc7','#16876a']),vmin=0,vmax=2)
    ticks=np.unique(np.linspace(0,n-1,min(n,9),dtype=int))
    ax_mat.set_xticks(ticks,labels=[str(t+1) for t in ticks])
    ax_mat.set_yticks(ticks,labels=[str(t+1) for t in ticks])
    ax_mat.set_xlabel('Global DoF j');ax_mat.set_ylabel('Global DoF i')
    ax_mat.set_title(f'Potential matrix sparsity: {n} × {n}',fontsize=12)
    local=(p+1)**2 if space=='H1' else 2*p*(p+1)
    nnz=int(pat.sum())
    info.set_text(f'Element DoFs: {local}  |  Global DoFs: {n}  |  Structural entries: {nnz:,} ({100*nnz/n**2:.1f}%)')
    info2.set_text(f'Selected DoF #{sel+1} ({kinds[sel]})  |  Elements: {memberships}  |  Coupled DoFs: {int(coupled.sum())}')
    fig.canvas.draw_idle()


def change_space(label):
    state['space']='H1' if label=='H1' else 'Hcurl'
    state['selected']=None
    redraw()

def change_p(label):
    state['p']=int(label)
    state['selected']=None
    redraw()

def on_click(event):
    pos=data['pos'];n=len(pos)
    if event.inaxes is ax_mesh and event.xdata is not None:
        d2=np.sum((pos-np.array([event.xdata,event.ydata]))**2,axis=1)
        j=int(np.argmin(d2))
        if d2[j] <= .012:
            state['selected']=j;redraw()
    elif event.inaxes is ax_mat and event.ydata is not None:
        j=int(np.floor(event.ydata+.5))
        if 0<=j<n:
            state['selected']=j;redraw()

space_radio.on_clicked(change_space)
p_radio.on_clicked(change_p)
fig.canvas.mpl_connect('button_press_event',on_click)
redraw()
plt.show()
