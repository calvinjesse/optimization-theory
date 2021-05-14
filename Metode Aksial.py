import numpy as np
import sympy as sym
import pandas as pd
import matplotlib.pyplot as plt

def newton(f,x0,tol):
    g=sym.diff(f)
    h=sym.diff(g)
    
    f=sym.lambdify(lamb,f)
    g=sym.lambdify(lamb,g)
    h=sym.lambdify(lamb,h)
    
    x1=x0-g(x0)/h(x0)
    
    i=1
    while abs(x1-x0)>tol:
        x0=x1
        x1=x0-g(x0)/h(x0)
        i=i+1
        
    return x1

def f(x):
    return 10*x[0]**2 + 10*(x[1]-3)**2 + 5*x[0]*x[1]

x0=[10,15]
n=len(x0)
e=np.diag(np.ones(n))
tol=5*10**(-9)
xk=[x0]
fxk=[f(x0)]

i=0
norm=tol+10
lamb=sym.Symbol('lamb')
while norm > tol:
    y=[xk[i]]
    for j in range (n):
        lamb_obj=newton(f(np.add(y[j],lamb*e[j,:])),y[j][j],tol)
        y.append(np.add(y[j],lamb_obj*e[j,:]))
    i=i+1
    xk.append(y[n])
    fxk.append(f(xk[i]))
    norm=np.linalg.norm(xk[i]-xk[i-1])
    
xp=xk[i]
fxp=fxk[i]
print('Peminimum dari f adalah ',xp)
print('Nilai minimum dari f adalah ',fxp)

data={'x':xk,'f(x)':fxk}

tabel = pd.DataFrame.from_dict(data, orient='index')
print(tabel.transpose())

#Plot Grafik

def g(x,y):
    return 10*x**2+10*(y-3)**2+5*x*y
x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
X,Y = np.meshgrid(x,y)
Z = g(X,Y)
fig=plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X,Y,Z,50,cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(10,130)
ax.plot_surface(X,Y,Z,cmap='viridis',edgecolor='none')
