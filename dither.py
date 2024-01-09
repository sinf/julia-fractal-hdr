
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

steps=8
step=1/steps

def f(y):
    return y

x=np.arange(0, 1.01, 0.01)
y=np.array([f(t) for t in x])

n=step
#n*=2
y1 = y + np.random.triangular(left=-n, mode=0, right=n, size=y.shape)
#y1 = y + np.random.uniform(-n, n, size=y.shape)

s=1

fig,ax = plt.subplots()
#ax.scatter(x, y, s=s, label='y')
#ax.scatter(x, (y*steps).round()/steps, s=s, label='quantized')
ax.scatter(x, (y1*steps).round()/steps, s=s, label='dithered')
ax.axline((x[0], y[0]), (x[-1], y[-1]), marker='o', color='red')

ax.legend()

plt.show()

