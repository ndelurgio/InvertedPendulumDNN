import numpy as np

def dynamics(x,u,M,m,g,l):
    x0,x1,x2,x3 = x
    x0dot = x1
    x1dot = ((M+m)*g*np.sin(x0) - m*l*x1**2*np.sin(x0)*np.cos(x0) - u*np.cos(x0)) / (M*l + m*l*np.sin(x0)**2)
    x2dot = x3
    x3dot = (m*x1**2*l*np.sin(x0) - m*g*np.sin(x0)*np.cos(x0) + u) / (m*np.sin(x0)**2 + M)
    return np.array([x0dot,x1dot,x2dot,x3dot])

def integrate(x0,xdot0,xdotf,dt):
    xf = np.zeros(len(x0))
    for i in range(0,len(xf)):
        xf[i] = x0[i] + 0.5*dt*(xdot0[i] + xdotf[i])
    return xf