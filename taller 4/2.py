# 2-field reaction–diffusion (Turing) with Neumann BC
# saves: 2_turing_u_minus_u3_minus_v_minus005.png
import numpy as np, matplotlib.pyplot as plt
from numba import njit

# ----- domain & numerics -----
L = 3.0
Nx = Ny = 256
dx = L/(Nx-1)
alpha, beta = 2.8e-4, 5e-2
Dmax = max(alpha, beta)
dt = 0.2*dx*dx/(4*Dmax)    # CFL-safe for 2D 5-point Laplacian
Tfinal = 15.0
nsteps = int(np.ceil(Tfinal/dt))

# ----- model: F, G -----
def F(u, v): return u - u**3 - v - 0.05
def G(u, v): return 10.0*(u - v)

# ----- init -----
rng = np.random.default_rng(0)
u = 0.05*rng.standard_normal((Ny, Nx))
v = 0.05*rng.standard_normal((Ny, Nx))

@njit
def laplace_5pt(a, dx):
    out = np.empty_like(a)
    out[1:-1,1:-1] = (a[1:-1,2:] + a[1:-1,:-2] + a[2:,1:-1] + a[:-2,1:-1] - 4*a[1:-1,1:-1])/(dx*dx)
    # Neumann BC: copy neighbors to edges (zero normal derivative)
    out[0, :]  = out[1, :]
    out[-1,:]  = out[-2,:]
    out[:, 0]  = out[:, 1]
    out[:,-1]  = out[:,-2]
    return out

@njit
def neumann_inplace(a):
    a[0, :] = a[1, :]; a[-1, :] = a[-2, :]
    a[:, 0] = a[:, 1]; a[:, -1] = a[:, -2]

@njit
def step(u, v, dx, dt, alpha, beta):
    Lu = laplace_5pt(u, dx)
    Lv = laplace_5pt(v, dx)
    u_new = u + dt*(alpha*Lu + (u - u**3 - v - 0.05))
    v_new = v + dt*(beta*Lv + (10.0*(u - v)))
    neumann_inplace(u_new); neumann_inplace(v_new)
    return u_new, v_new

for _ in range(nsteps):
    u, v = step(u, v, dx, dt, alpha, beta)

# ----- figure -----
fig, ax = plt.subplots(figsize=(6,5), dpi=160)
im = ax.imshow(u, origin="lower", extent=(0,L,0,L))
ax.set_xticks([]); ax.set_yticks([])
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("u(x,y,T)")
txt = (r"$\partial_t u=\alpha\nabla^2 u+u-u^3-v-0.05,$ "
       r"$\partial_t v=\beta\nabla^2 v+10(u-v)$"+"\n"
       fr"$\alpha={alpha}$, $\beta={beta}$, $L={L}$, $N={Nx}\times{Ny}$, $t\approx{Tfinal}$")
ax.text(0.01,0.99,txt,transform=ax.transAxes,va="top",ha="left",fontsize=8,
        color="w",bbox=dict(facecolor="k",alpha=0.45,pad=4,edgecolor="none"))
fig.tight_layout()
fig.savefig("2_turing_u_minus_u3_minus_v_minus005.pdf", bbox_inches="tight")
