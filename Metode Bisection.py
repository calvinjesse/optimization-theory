import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

#fungsi yang mau dicari minimumnya
def f(x):
    return x**4-4*x**3+60*x**2-70


#turunan fungsi f
def f_derivative(x):
    return 4*x**3-12*x**2+120*x


#inisialisasi variabel
delta = 0.000001
a = [-1]
b = [3]
c = []
L = [b[0]-a[0]]
f_derivative_a = []
f_derivative_b = []
f_derivative_c = []
N = int(np.ceil(math.log(delta/L[0],0.5)))

#cek apakah titik peminimumnya di batas atau bukan
#jika bukan, jalankan metode Bisection
if f_derivative(a[0]) == 0:
    print('Titik optimum f adalah (%f, %f)' %(a[0], f(a[0])))
elif f_derivative(b[0]) == 0:
    print('Titik optimum f adalah (%f, %f)' %(b[0], f(b[0])))
else:
    for k in range(N):
        #hitung titik tengah dari [a,b]
        c.append((a[k]+b[k])/2)
        
        #hitung f' di a, b, dan c
        f_derivative_a.append(f_derivative(a[k]))
        f_derivative_b.append(f_derivative(b[k]))
        f_derivative_c.append(f_derivative(c[k]))
        
        if f_derivative_a[k] * f_derivative_c[k] < 0:
            a.append(a[k])
            b.append(c[k])
            L.append(b[k+1] - a[k+1])
        elif f_derivative_c[k] * f_derivative_b[k] < 0:
            a.append(c[k])
            b.append(b[k])
            L.append(b[k+1] - a[k+1])
        elif f_derivative_c[k] == 0:
            print('Titik optimum f adalah (%f, %f)' %(c[k], f(c[k])))
            break
            
        if k == N-1:
            c.append((a[k-1] + b[k-1])/2)
            print('Titik optimum f adalah (%f, %f)' %(c[k], f(c[k])))
            
#f_derivative_a, f_derivative_b, f_derivative_c
f_derivative_a.append(float('nan'))
f_derivative_b.append(float('nan'))
f_derivative_c.append(float('nan'))

data = {'a': a, 'c': c, 'b': b, "f'(a)": f_derivative_a, "f'(c)": 
        f_derivative_c, "f'(b)": f_derivative_b, 'L': L}

tabel = pd.DataFrame.from_dict(data, orient='index')
print(tabel.transpose())

x = np.linspace(0.5,np.pi,100)
plt.plot(x,f(x))
plt.grid()
plt.show()

