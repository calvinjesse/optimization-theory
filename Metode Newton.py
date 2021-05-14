import matplotlib.pyplot as plt
import sympy as sym
import pandas as pd
import numpy as np

x = sym.Symbol('x')
f = x*sym.cos(x) - x**2 
g = sym.diff(f)                         
h = sym.diff(g)                         

f = sym.lambdify(x,f)
g = sym.lambdify(x,g)
h = sym.lambdify(x,h)


tol = 0.05
x0 = 1
e = [None]
xk = [x0]
fxk = [f(x0)]

x1 = xk[0]-g(xk[0])/h(xk[0])
xk.append(x1)
fxk.append(f(x1))
e.append(abs(x1-x0))


i=1
while e[i] > tol:
  x1 = xk[i]-g(xk[i])/h(xk[i])
  xk.append(x1)
  fxk.append(f(x1))
  e.append(abs(xk[i+1]-xk[i]))
  i = i+1

xp = xk[i]
fxp = fxk[i]
print('Titik minimum adalah (%f,%f) ' %(xp,fxp))

data = {
    'x': xk,
    'f(x)': fxk,
    'e': e
}

print(pd.DataFrame(data))

x = np.linspace(-2,2,100)
plt.plot(x,f(x))
plt.grid()
plt.show()
