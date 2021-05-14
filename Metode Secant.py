import sympy as sym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x=sym.Symbol('x')
f=x**4-4*x**3+60*x**2-70 
g=sym.diff(f)

f=sym.lambdify(x,f)
g=sym.lambdify(x,g)

x0=2
x1=3
tol=0.05
xk=[x0,x1]
fxk=[f(x0),f(x1)]
e=[None,abs(x1-x0)]

x2=xk[1]-(xk[1]-xk[0])/(g(xk[1])-g(xk[0]))*g(xk[1])
xk.append(x2)
fxk.append(f(x2))
e.append(abs(x2-x1))

i=2
while e[i]>tol:
    x2=xk[i]-(xk[i]-xk[i-1])/(g(xk[i])-g(xk[i-1]))*g(xk[i])
    xk.append(x2)
    fxk.append(f(x2))
    e.append(abs(xk[i]-xk[i-1]))
    i=i+1

xp=xk[i]
fxp=fxk[i]
print('titik minimum adalah (%f,%f)'%(xp,fxp))

data={'x':xk,'f(x)':fxk,'e':e} 
print(pd.DataFrame(data))

x = np.linspace(-3,3,100)
plt.plot(x,f(x))
plt.grid()
plt.show()
