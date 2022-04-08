import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.mplot3d.art3d as art3d

def integrate(ic, ti, p):
	m, l, xp, yp, zp, gc = p
	theta, omegat, phi, omegap = ic

	sub = {g:gc, M:m, L:l, Xp:xp, Yp:yp, Zp:zp, THETA:theta, THETAdot:omegat, PHI:phi, PHIdot:omegap}

	diff_eq = [omegat,ALPHA_T.subs(sub),omegap,ALPHA_P.subs(sub)]
	
	print(ti)

	return diff_eq


g, t = sp.symbols('g t')
M, L, Xp, Yp, Zp = sp.symbols('M L Xp Yp Zp')
THETA, PHI = dynamicsymbols('THETA PHI')

X = L * sp.cos(THETA) * sp.cos(PHI)
Y = L * sp.sin(THETA) * sp.cos(PHI)
Z = L * sp.sin(PHI)

THETAdot = THETA.diff(t, 1)
THETAddot = THETA.diff(t, 2)
PHIdot = PHI.diff(t, 1)
PHIddot = PHI.diff(t, 2)

Xdot = X.diff(t, 1)
Ydot = Y.diff(t, 1)
Zdot = Z.diff(t, 1)

T = sp.Rational(1, 2) * M * (Xdot**2 + Ydot**2 + Zdot**2)
V = M * g * Z

Lg = T - V

dLdTHETA = Lg.diff(THETA, 1)
dLdTHETAdot = Lg.diff(THETAdot, 1)
ddtdLdTHETAdot = dLdTHETAdot.diff(t, 1)
dLTHETA = ddtdLdTHETAdot - dLdTHETA

dLdPHI = Lg.diff(PHI, 1)
dLdPHIdot = Lg.diff(PHIdot, 1)
ddtdLdPHIdot = dLdPHIdot.diff(t, 1)
dLPHI = ddtdLdPHIdot - dLdPHI

sol = sp.solve([dLTHETA,dLPHI], [THETAddot,PHIddot])

ALPHA_T = sp.simplify(sol[THETAddot])
ALPHA_P = sp.simplify(sol[PHIddot])

#---------------------------------------------------------


gc = 9.8
m = 1
l = 1
xp,yp,zp = [0, 0, 1]
thetao = 45
omegato = 30
phio = -45
omegapo = 0
tf = 30 

cnvrt = np.pi/180
thetao *= cnvrt
omegato *= cnvrt
phio *= cnvrt
omegapo *= cnvrt

p = m, l, xp, yp, zp, gc
ic = thetao, omegato, phio, omegapo

nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

topo = odeint(integrate, ic, ta, args=(p,))

x = np.asarray([X.subs({L:l, THETA:topo[i,0], PHI:topo[i,2]}) for i in range(nframes)])
y = np.asarray([Y.subs({L:l, THETA:topo[i,0], PHI:topo[i,2]}) for i in range(nframes)])
z = np.asarray([Z.subs({L:l, THETA:topo[i,0], PHI:topo[i,2]}) for i in range(nframes)])

ke = np.asarray([T.subs({M:m, L:l, THETA:topo[i,0], THETAdot:topo[i,1], PHI:topo[i,2], PHIdot:topo[i,3]}) for i in range(nframes)])
pe = np.asarray([V.subs({g:gc, M:m, L:l, PHI:topo[i,2]}) for i in range(nframes)])
E = ke + pe

#------------------------------------------------------------

xmax = max(x) if max(x) > xp else xp
xmin = min(x) if min(x) < xp else xp
ymax = max(y) if max(y) > yp else yp
ymin = min(y) if min(y) < yp else yp
zmax = max(z) if max(z) > zp else zp
zmin = min(z) if min(z) < zp else zp
dth=360/nframes

zmax += .5
zmin -= .5

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def run(frame):
	plt.clf()
	fig.suptitle('A 3D Simple Pendulum')
	ax = fig.add_subplot(211,projection='3d')
	ax.plot([xp,x[frame]],[yp,y[frame]],[zp,z[frame]],color="xkcd:cerulean")
	ax.scatter(x[frame],y[frame],z[frame], color="xkcd:red", s=100)
	#circle=Circle((x[frame],y[frame]),radius=0.05,fc='xkcd:black')
	#ax.add_patch(circle)
	#art3d.pathpatch_2d_to_3d(circle, z=zmin)
	ax.plot(x[0:frame],y[0:frame],zs=zmin,color='xkcd:black')
	ax.set_xlim3d(float(xmin),float(xmax))
	ax.set_ylim3d(float(ymin),float(ymax))
	ax.set_zlim(float(zmin),float(zmax))
	ax.set_facecolor('xkcd:black')
	ax.xaxis.pane.fill = False
	ax.yaxis.pane.fill = False
	#ax.zaxis.pane.fill = False
	ax.xaxis.pane.set_edgecolor('black')
	ax.yaxis.pane.set_edgecolor('black')
	#ax.zaxis.pane.set_edgecolor('black')
	ax.grid(False)
	ax.set_xticks([])                               
	ax.set_yticks([])                               
	ax.set_zticks([])
	#ax.view_init(elev=30, azim=frame*dth)
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('pendulum3D_v2.mp4', writer=writervideo)
plt.show()


