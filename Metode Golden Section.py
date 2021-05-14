import numpy as np
from pandas import DataFrame
from math import log
import matplotlib.pyplot as plt

#fungsi yang mau dicari minimumnya
def f(x):
    return x**5 + 10*x**2 - np.tan(x)

#inisialisasi variabel
delta=10**(-10)
a = [-7.86]
b = [-7.85]
L = [b[0]-a[0]]
alpha = []
beta = []
falpha = []
fbeta = []
rho = (3-np.sqrt(5))/2

#mencari nilai N
N = int(np.ceil(log((delta/L[0]), 0.61803)))

#mulai iterasi metode Golden Section
for k in range(N):
    #tentukan alpha dan beta
    alpha.append(a[k] + rho * L[k])
    beta.append(a[k] + (1-rho) * L[k])
    
    #hitung f di alpha dan beta
    falpha.append(f(alpha[k]))
    fbeta.append(f(beta[k]))
    
    #cek mana yang lebih kecil antara falpha dengan fbeta
    if falpha[k] > fbeta[k]:
        a.append(a[k])
        b.append(beta[k])
    else:
        a.append(alpha[k])
        b.append(b[k])
    
    L.append(b[k+1]-a[k+1])
    
    
#alpha, beta, falpha, fbeta terakhir diisi NaN
alpha.append(float('nan'))
beta.append(float('nan'))
falpha.append(float('nan'))
fbeta.append(float('nan'))

xp = (b[N-1] + a[N-1])/2
fxp = f(xp)

print('Titik optimum f adalah (%f, %f)' %(xp, fxp))
data = {'a': a, 'alpha': alpha, 'beta': beta, 'b': b, 'falpha': falpha, 
        'fbeta': fbeta, 'rho': rho, 'L': L}
print(DataFrame(data))

x = np.linspace(-7.86,-7.85,100)
plt.plot(x,f(x))
plt.grid()
plt.show()

