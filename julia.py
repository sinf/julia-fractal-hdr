import time
import numpy as np
import cv2
from scipy import ndimage
import multiprocessing
from argparse import ArgumentParser
from math import log, sqrt
from tqdm import tqdm

NPROC=12

class Julia:
    def __init__(self, c, viewbox, max_iter=300):
        """ viewbox: (min real, min imag, max real, max imag) """
        self.c = c
        self.max_iter = max_iter
        self.view_org = complex(viewbox[0], viewbox[1])
        self.view_scale = (viewbox[2]-viewbox[0], viewbox[3]-viewbox[1])

    def sample(self, point):
        """ https://linas.org/art-gallery/escape/escape.html """
        z = self.view_org + complex(self.view_scale[0]*point[0], self.view_scale[1]*point[1])
        c = self.c
        for n in range(self.max_iter):
            z = z*z + c
            if z.real*z.real + z.imag*z.imag > 2*2:
                break
        for _ in range(3):
            z = z*z + c
        za = max(abs(z), 1.0000001)
        return n - log(log(za)) / log(2)

def render(dim, sample):
    tr = lambda x,y: (x/(dim[0]-1), 1-y/(dim[1]-1))
    coords = (tr(x,y) for y in range(dim[1]) for x in range(dim[0]))
    with multiprocessing.Pool(NPROC) as pool:
        pixels = []
        for p in tqdm(pool.imap(sample, coords, chunksize=256), total=dim[0]*dim[1]):
            pixels += [p]
        pixels = tuple(pixels)
    buf = np.array(pixels, dtype='float64').reshape((dim[1],dim[0],-1))
    return buf

def render_ss(dim, sample, ss):
    dim2=(dim[0]*ss, dim[1]*ss)
    buf=render(dim2, sample)
    if ss>1:
        buf=ndimage.zoom(buf, (1/ss,1/ss,1), order=3, prefilter=True)
    return buf

def map_gradient(buf, c0, c1):
    c0 = np.array(c0, dtype=float) * 255.0
    c1 = np.array(c1, dtype=float) * 255.0
    lo = buf.min()
    hi = buf.max()
    #lo,hi = 0,200
    n = (buf - lo) / (hi - lo)
    n = np.power(n, 0.7) # brighten up
    n = np.clip(n, 0, 1)
    return c0 + (c1-c0)*n.reshape((*buf.shape[:2],1))

def cv2ify(buf):
    buf = map_gradient(buf, (0,0,0), (1,1,0))

    # noise fixes gradients by dithering when shit software downsamples image to 8bit
    n = 4 # resist downscaling by half
    buf += np.random.triangular(left=-0.5*n, mode=0, right=0.5*n, size=buf.shape)

    buf = buf.astype('float16')
    return buf

def viewbox_at(center, scale, dim):
    d = sqrt( dim[0]**2 + dim[1]**2 )*2
    x = dim[0] / d * scale
    y = dim[1] / d * scale
    l = center[0] - x
    r = center[0] + x
    b = center[1] - y
    t = center[1] + y
    return (l, b, r, t)

def main():
    ap = ArgumentParser()
    ap.add_argument('-s', '--supersample', type=int, default=1)
    ap.add_argument('-r', '--resolution', nargs=2, type=int, default=(720,480))
    ap.add_argument('-o', '--output', default='image.png')
    args=ap.parse_args()

    ss=args.supersample
    dim=args.resolution
    n_samples=ss*ss*dim[0]*dim[1]

    print('Samples', n_samples)
    #print('ETA', n_samples/(720*480*3*3)*3.9521849155426025)

    t0=time.time()

    fractal=Julia(c=(0.37+0.1j), viewbox=viewbox_at((0.165,0.12),0.15,dim))
    buf=render_ss(dim, fractal.sample, ss)

    t1=time.time()
    elapsed=t1-t0
    time_per_pixel=1e6*elapsed/n_samples

    print('elapsed:', elapsed)
    print('us/pixel:', time_per_pixel)
    print(buf.dtype, buf.shape)

    cvbuf = cv2ify(buf)
    save(args.output, cvbuf)

def save(fn,buf):
    cv2.imwrite(fn, buf)

if __name__=='__main__':
    main()

