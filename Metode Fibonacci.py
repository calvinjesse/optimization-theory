#Metode Fibonacci
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from numpy import linspace


#fungsi menghitung barisan fibonacci
def fibonacci(n):
    if n == 0:
        return([1])
    elif n == 1:
        return([1,1])
    else:
        F = [1,1]
        for i in range(1,n):
            F.append(F[i] + F[i-1])
        return(F)

#fungsi yang mau dicari minimumnya
def f(x):
    return np.cos(x)-np.sin(2*x)

#inisialisasi variabel
eps = 0.05
delta = 0.01
a = [-4]
b = [3]
L = [b[0] - a[0]]
alpha = []
beta = []
falpha = []
fbeta = []
rho = []

#mencari nilai N
z = (L[0]/delta)*(1+2*eps)
N = 0
F = fibonacci(N)
while F[N] < z:
    N = N+1
    F = fibonacci(N)
N = N-1

#mulai iterasi metode Fibonacci
for k in range(N-1):
    #hitung rhoo
    rho.append(1-(F[N-(k+1)]/F[N-k]))
    
    #cek iterasi
    #pada iterasi terakhir, rho dikurangi eps
    if k > N-2:
        alpha.append(a[k]+rho[k]*L[k])
        beta.append(a[k]+(1-rho[k])*L[k])
    else:
        alpha.append(a[k]+(rho[k]-eps)*L[k])
        beta.append(a[k]+(1-(rho[k]-eps))*L[k])
     
    falpha.append(f(alpha[k]))
    fbeta.append(f(beta[k]))
    
    #cek mana yang lebih kecil antara falpha dengan fbeta
    #jika falpha lebih kecil, artinya minimum di antara a dan beta
    #jika fbeta lebih kecil, artinya minumumnya di antara alpha dan b
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
rho.append(float('nan'))

xp = (b[N-1] + a[N-1])/2
fxp = f(xp)

print('Titik optimum f adalah (%f, %f)' %(xp, fxp))
data = {'a': a, 'alpha': alpha, 'beta': beta, 'b': b, 'falpha': falpha, 'fbeta': fbeta} #'rho': rho, 'L': L}
print(DataFrame(data))



x = linspace(-4, 3, 100)
#plt.figure(figsize=(8,5))
plt.plot(x,f(x))
#plt.xticks(x, rotation=20)
plt.grid()
plt.show()
