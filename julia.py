import time
import numpy as np
import cv2
from scipy import ndimage
import multiprocessing
from argparse import ArgumentParser
from math import log

NPROC=12

class Julia:
    def __init__(self, c, viewbox, max_iter=1000):
        self.viewbox = viewbox
        self.c = c
        self.max_iter = max_iter

    def sample(self, point):
        """ https://linas.org/art-gallery/escape/escape.html """
        z = complex(*point)
        c = self.c
        for n in range(self.max_iter):
            z = z*z + c
            if z.real*z.real + z.imag*z.imag > 2*2:
                break
        z = z*z + c
        z = z*z + c
        return n - log(log(abs(z))) / log(2)

def render(dim, sample):
    tr = lambda x,y: (x/(dim[0]-1), 1-y/(dim[1]-1))
    coords = (tr(x,y) for y in range(dim[1]) for x in range(dim[0]))
    with multiprocessing.Pool(NPROC) as pool:
        pixels = pool.imap(sample, coords, chunksize=256)
        pixels = tuple(pixels)
    buf = np.array(pixels, dtype=float).reshape((dim[1],dim[0],1))
    return buf

def render_ms(dim, sample, ms):
    dim2=(dim[0]*ms, dim[1]*ms)
    buf=render(dim2, sample)
    if ms>1:
        buf=ndimage.zoom(buf, (1/ms,1/ms,1), order=3, prefilter=True)
    return buf

def map_gradient(buf, c0, c1):
    c0 = np.array(c0, dtype=float) * 255.0
    c1 = np.array(c1, dtype=float) * 255.0
    lo = buf.min()
    hi = buf.max()
    n = (buf - lo) / (hi - lo)
    return c0 + (c1-c0)*n;

def cv2ify(buf):
    buf = map_gradient(buf, (0,0,0), (1,1,0))
    # noise fixes gradients by dithering when shit software downsamples image to 8bit
    buf += np.random.triangular(left=-0.5, mode=0, right=0.5, size=buf.shape)
    buf = buf.astype('float16')
    return buf

def save(buf):
    filename='image.png'
    cv2.imwrite(filename, buf)

def main():
    #dim=(3840,3840)
    ms=8
    dim=(720,480)
    n_pixels=ms*ms*dim[0]*dim[1]
    print('ETA', n_pixels/(3840*3480)*7.194469451904297)
    t0=time.time()
    fractal=Julia(c=(0.37+0.1j), viewbox=(-2,-2,2,2))
    buf=render_ms(dim, fractal.sample, ms=ms)
    t1=time.time()
    elapsed=t1-t0
    time_per_pixel=1e6*elapsed/n_pixels
    print('elapsed:', elapsed)
    print('us/pixel:', time_per_pixel)
    print(buf.dtype, buf.shape)

    cvbuf = cv2ify(buf)
    #cv2.imshow('julia', cvbuf)
    save(cvbuf)

if __name__=='__main__':
    main()

