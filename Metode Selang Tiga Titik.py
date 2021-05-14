from numpy import linspace
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return ((np.sin(2*x))**3 + np.cos(x) )

k=0
a=[-4]
b=[3]
L=b[0]-a[0]
alpha=[]
beta=[]
gamma=[]
falpha=[]
fbeta=[]
fgamma=[]

while L>10**(-7):
    s=linspace(a[k],b[k],5)
    alpha.append(s[1])
    beta.append(s[2])
    gamma.append(s[3])
    falpha.append(f(alpha[k]))
    fbeta.append(f(beta[k]))
    fgamma.append(f(gamma[k]))
    
    if falpha[k]<fbeta[k]<fgamma[k] :
        a.append(a[k])
        b.append(beta[k])
    elif falpha[k]>fbeta[k]>fgamma[k]:
        a.append(beta[k])
        b.append(b[k])
    else :
        a.append(alpha[k])
        b.append(gamma[k])
        
    k=k+1
    L=(b[k]-a[k])
    
alpha.append(float('nan'))
beta.append(float('nan'))
gamma.append(float('nan'))
falpha.append(float('nan'))
fbeta.append(float('nan'))
fgamma.append(float('nan'))

xp=(b[k]+a[k])/2

fxp = f(xp)

print("Titik optimum f adalah (%f,%f)"%(xp,fxp))
data={'a':a,'alpha':alpha,'beta':beta,'gamma':gamma,'b':b,'falpha':falpha,
      'fbeta':fbeta,'fgamma':fgamma}

print(DataFrame(data))

x=linspace(-4,3,100)
plt.plot(x,f(x))
plt.grid()
plt.show()
