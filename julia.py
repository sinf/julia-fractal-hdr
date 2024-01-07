import time
import numpy as np
import cv2
from scipy import ndimage
import multiprocessing
from argparse import ArgumentParser
from math import log, sqrt
from tqdm import tqdm

NPROC=multiprocessing.cpu_count()

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
        for p in tqdm(pool.imap(sample, coords, chunksize=8192), total=dim[0]*dim[1]):
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
    lo = buf.min()
    hi = buf.max()
    n = (buf - lo) / (hi - lo)
    #n = np.power(n, 0.7) # brighten up
    n = np.clip(n, 0, 1)
    return c0 + (c1-c0) * n.reshape((*buf.shape[:2],1))

def cv2ify(buf):
    hi=0xffff
    buf = map_gradient(buf,
       hi * np.array((0,0,0)),
       hi * np.array((0.393,0.87,0.041)),
    )

    # noise fixes gradients by dithering when shit software downsamples image to 8bit
    n = 128
    n *= 4 # resist downscaling by half
    buf += np.random.triangular(left=-n, mode=0, right=n, size=buf.shape)

    # cv2.imwrite only writes 16bit png if type is uint16, not float16
    buf = np.clip(buf, 0, hi).astype('uint16')

    return buf

def save(fn,buf):
    cv2.imwrite(fn, buf)

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
    global NPROC

    ap = ArgumentParser()
    ap.add_argument('-s', '--supersample', type=int, default=1,
      help='take N^2 samples and then downsample [1]', metavar='N')
    ap.add_argument('-r', '--resolution', nargs=2, type=int, default=(720,480),
      help='output resolution [720 480]', metavar=('WIDTH','HEIGHT'))
    ap.add_argument('-o', '--output', nargs='+', default=['image.png'],
      help='Output filename(s) [image.png]', metavar='PATH')
    ap.add_argument('-p', '--nproc', default=NPROC,
      help=f'Parallel processes to use [{NPROC}]', metavar='COUNT')
    args=ap.parse_args()

    ss=args.supersample
    dim=args.resolution
    n_samples=ss*ss*dim[0]*dim[1]
    NPROC=args.nproc

    print('Using processes:', NPROC)
    print('Samples/pixel', ss**2)
    print('Total samples to calculate:', n_samples)
    print('Outputs', args.output)

    t0=time.time()

    fractal=Julia(c=(0.37+0.1j), viewbox=viewbox_at((0.165,0.12),0.15,dim))
    buf=render_ss(dim, fractal.sample, ss)

    t1=time.time()
    elapsed=t1-t0
    time_per_sample=1e6*elapsed/n_samples

    print(f'Elapsed: {elapsed:.3f} s')
    print(f'µs/sample: {time_per_sample:.6f} s')
    #print(buf.dtype, buf.shape)

    cvbuf = cv2ify(buf)

    for filename in args.output:
      save(filename, cvbuf)

if __name__=='__main__':
    main()

