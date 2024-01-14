import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from math import *

def f(t):
  c=2*pi*t*2
  b=2*pi*t*3
  a=2*pi*t*4
  return sin(a)*sin(a)+cos(a)*cos(b) , (sin(c)+cos(b)+sin(a) )*0.5

param = [i/999 for i in range(2000)]
points = np.array([f(p) for p in param])
x,y = points.T
fig,ax = plt.subplots()
ax.scatter(x, y, s=2)

def update(i):
  ax.clear()
  ax.scatter(x, y, s=2)
  p=f(i/100.0)
  ax.scatter(p[0], p[1], s=10, color='red')

plt.show()

sys.exit(0)
anim = FuncAnimation(fig, update, frames=np.arange(200), interval=50)
with tqdm(total=200) as progbar:
  anim.save('plot.mp4', fps=20, progress_callback=lambda i,imax: progbar.update(1) )

