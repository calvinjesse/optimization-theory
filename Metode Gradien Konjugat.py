import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


A = [[6,4],[4,4]]
b = [6,8]
x0 = [1,1]
xk = [x0]
eps=0.01

def cg(A,b,x):
    r=b-np.dot(A,x)
    p=r
    rs_old=np.dot(np.transpose(r),r)
    
    for i in range (len(b)):
        Ap = np.dot(A,p)
        alpha = rs_old/np.dot(np.transpose(p),Ap)
        x=x+np.dot(alpha,p)
        xk.append(x)
        r=r-np.dot(alpha,Ap)
        rs_new=np.dot(np.transpose(r),r)
        if np.sqrt(rs_new)<eps:
            break
        p=r+(rs_new/rs_old)*p
        rs_old=rs_new
    return x

xp=cg(A,b,x0)
    
print("Nilai x adalah : ",xp)

for i in range (len(xp)):
    print('x_'+str(i+1),' = ',xp[i])
    x = xp[0]
    y = xp[1]

print("Nilai minimum fungsi f(x1,x2) =",3*x**2+2*y**2+4*x*y-6*x-8*y+6)
    
data={'x':xk}
print(pd.DataFrame(data))

def g(x,y):
    return 3*x**2+2*y**2+4*x*y-6*x-8*y+6
x = np.linspace(-10,10,100)
y = np.linspace(-30,30,100)
X,Y = np.meshgrid(x,y)
Z = g(X,Y)
fig=plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X,Y,Z,50,cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(10,30)
ax.plot_surface(X,Y,Z,cmap='inferno',edgecolor='none')

