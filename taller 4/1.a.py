import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numba as nb
from IPython.display import HTML

dx=0.05; dx2=dx*dx; k0=-30.; dt=dx2/20.0; xmax=20.0
xs=np.arange(-xmax, xmax+dx/2, dx)

psr=np.exp(-2*(xs-0)**2)*np.cos(k0*xs)
psi=np.exp(-2*(xs-0)**2)*np.sin(k0*xs)
v=(50/50.)*xs**2
a=0.1
Tfinal=20
steps_per_frame = 200
nsteps = int(round(Tfinal/dt))
n_frames = nsteps // steps_per_frame

@nb.njit(fastmath=True)
def step(psr0, psi0, dt, dx2, v, a):
    psr1=psr0.copy(); psi1=psi0.copy()
    psr1[1:-1]=psr0[1:-1]-a*(dt/dx2)*(psi0[2:]+psi0[:-2]-2*psi0[1:-1])+dt*v[1:-1]*psi0[1:-1]
    psi1[1:-1]=psi0[1:-1]+a*(dt/dx2)*(psr1[2:]+psr1[:-2]-2*psr1[1:-1])-dt*v[1:-1]*psr1[1:-1]
    psr1[0]=psr1[1]; psr1[-1]=psr1[-2]
    psi1[0]=psi1[1]; psi1[-1]=psi1[-2]
    return psr1, psi1

fig,ax=plt.subplots(figsize=(7,3))
(line,)=ax.plot(xs,4*(psr**2+psi**2))
ax2 = ax.twinx()
ax2.plot(xs, v, 'r--', alpha=0.7, label="V(x)")
ax2.set_ylabel("V(x)")
time_txt=ax.text(0.01,0.92,"t=0.000", transform=ax.transAxes)
ax.set_xlim(xs[0],xs[-1]); ax.set_ylim(0,1.2*np.max(4*(psr**2+psi**2)))
ax.set_xlabel("x"); ax.set_ylabel("|ψ|² ×4"); ax.set_title("Wave packet")

sim_time=0.0
def update(_):
    global psr,psi,sim_time
    for _ in range(steps_per_frame):
        psr,psi=step(psr,psi,dt,dx2,v,a)
        sim_time+=dt
    y=4*(psr**2+psi**2)
    line.set_ydata(y)
    time_txt.set_text(f"t={sim_time:.3f}")
    #ax.set_ylim(0, (psr**2+psi**2))  # in case range changes
    return line,time_txt

ani=FuncAnimation(fig, update, frames=n_frames, interval=10, blit=False)
plt.close(fig)
ani.save("1.a.mp4", fps=30)

# --- métricas: mu(t) y sigma(t) ---
Tfinal = 20                     # cambia solo esto si quieres otro tiempo
nsteps = int(round(Tfinal/dt))
sample_every = 200                 # guarda cada 200 pasos para no ralentizar
t = 0.0

times = []
mus = []
sigmas = []

def moments(psr, psi, xs, dx):
    rho = psr*psr + psi*psi                  # |ψ|^2
    Z = np.sum(rho)*dx                       # norma (por si deriva)
    mu = np.sum(xs*rho)*dx / Z
    var = np.sum((xs-mu)**2 * rho)*dx / Z
    return mu, np.sqrt(var)

for k in range(nsteps):
    psr, psi = step(psr, psi, dt, dx2, v, a)  # tu paso Numba
    t += dt
    if (k+1) % sample_every == 0 or k+1 == nsteps:
        mu, sigma = moments(psr, psi, xs, dx)
        times.append(t); mus.append(mu); sigmas.append(sigma)

# --- gráfico y PDF ---
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,3))
ax.plot(times, mus, lw=1.6, label=r'$\mu(t)$')
ax.fill_between(times,
                np.array(mus)-np.array(sigmas),
                np.array(mus)+np.array(sigmas),
                alpha=0.25, label=r'$\mu \pm \sigma$')
ax.set_xlabel('t'); ax.set_ylabel('posición')
ax.legend(loc='best', frameon=False)
fig.tight_layout()
fig.savefig('1.a.pdf')   # <- salida pedida
plt.show()
