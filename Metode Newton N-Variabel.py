import sympy as sym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def grad(f,nilai,n):
    G=[]
    
    for i in range (n):
        G.append(sym.diff(f,x[i]))
        G[i]=sym.lambdify(x,G[i])
        calc='G[%d](%f' %(i,nilai[0,0])
        for j in range (1,n):
            calc=calc+',%f'%nilai[0,j]
        calc=calc+')'
        G[i]=eval(calc)
    G=np.matrix(G).transpose()
    return(G)

def hessian(f,nilai,n):
    H=[]
    for i in range (n):
        row=[]
        for j in range (n):
            row.append(sym.diff(f,x[i],x[j]))
            row[j]=sym.lambdify(x,row[j])
            
            calc='row[%d](%f'%(j,nilai[0,0])
            for k in range (1,n):
                calc=calc+',%f' %nilai[0,k]
            calc=calc+')'
            row[j]=eval(calc)
        H.append(row)
        
    H=np.matrix(H)
    return H

x_awal=np.matrix([4,4])
n=x_awal.shape[1]
tol=0.0001

x=[]
for i in range (n):
    x.append(sym.Symbol('x%d'%i))

f=(x[0]**2-2*x[0]+4*x[1]**2-8*x[1])**2
f_func=sym.lambdify(x,f)
e=[None]
xk=[x_awal]

calc='f_func(%f'%x_awal[0,0]
for i in range (1,n):
    calc=calc+',%f'%x_awal[0,i]
calc=calc+')'
fxk=[eval(calc)]

G=grad(f,x_awal,n)
H=hessian(f,x_awal,n)
x_next=x_awal-(np.linalg.inv(H)*G).transpose()
xk.append(x_next)
e.append(np.linalg.norm(x_next-x_awal))

calc='f_func(%f'%x_next[0,0]
for i in range (1,n):
    calc=calc+',%f'%x_next[0,i]
calc=calc+')'
fxk.append(eval(calc))

i=1
while e[i]>tol:
    G=grad(f,xk[i],n)
    H=hessian(f,xk[i],n)
    x_next=xk[i]-(np.linalg.inv(H)*G).transpose()
    xk.append(x_next)
    e.append(np.linalg.norm(xk[i+1]-xk[i]))
    
    calc='f_func(%f'%x_next[0,0]
    for j in range (1,n):
        calc=calc+',%f' %x_next[0,j]
    calc=calc+')'
    fxk.append(eval(calc))
    i=i+1
    
xp=xk[i]
fxp=fxk[i]
print('Peminimum dari f adalah ',xp)
print('Nilai minimum dari f adalah ',fxp)

data={'x':xk,'f(x)':fxk,'e':e}
print(pd.DataFrame(data))

#Plot Grafik

def g(x,y):
    return (x**2-2*x+4*y**2-8*y)**2
x = np.linspace(0,3.5,100)
y = np.linspace(0,2,100)
X,Y = np.meshgrid(x,y)
Z = g(X,Y)
fig=plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X,Y,Z,50,cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(10,190)
ax.plot_surface(X,Y,Z,cmap='inferno',edgecolor='none')
        
    

    
   
    