import time
import numpy as np
import cv2
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.stats.qmc import Halton
import multiprocessing
from argparse import ArgumentParser
from math import log, sqrt
from tqdm import tqdm

try:
    from itertools import batched
except ImportError:
    # <3.12
    from itertools import takewhile, zip_longest
    def batched(iterable, n):
        fillvalue = object()
        args = (iter(iterable),) * n
        for x in zip_longest(*args, fillvalue=fillvalue):
            if x[-1] is fillvalue:
                yield tuple(takewhile(lambda v: v is not fillvalue, x))
            else:
                yield x

NPROC=multiprocessing.cpu_count()
MAX_SAMPLE_BATCH=64

class Julia:
    def __init__(self, c, viewbox, max_iter=1000):
        """ viewbox: (min real, min imag, max real, max imag) """
        self.c = np.complex128(c)
        self.max_iter = max_iter
        self.view_org = np.array((viewbox[0], viewbox[1]))
        self.view_scale = np.array((viewbox[2]-viewbox[0], viewbox[3]-viewbox[1]))

    def sample(self, points):
        """ https://linas.org/art-gallery/escape/escape.html
        points: 2d numpy array of points (float)
        output: 1d numpy array of intensity
        """
        z = self.view_org + self.view_scale*points
        z = z.astype('float64').view(dtype=np.complex128).flatten()
        c = self.c
        n = np.zeros(len(points), dtype=int)
        mask = np.full(len(points), False)
        with np.errstate(divide='ignore', under='ignore', over='ignore', invalid='ignore'):
            for _ in range(self.max_iter):
                z1 = z*z + c
                mask = ( z1.real*z1.real + z1.imag*z1.imag > 2*2 ).flatten()
                masknot = ~mask
                z = z*mask + z1*masknot
                if mask.all():
                    break
                n += masknot.astype(int)
            n = np.asarray(n)
            for _ in range(3):
                z = z*z + c
        za = np.maximum(np.abs(z), 1.0000001)
        return n - np.log(np.log(za)).flatten() / np.log(2)

def make_subsample_offsets(k, h, w):
    return np.zeros(k*h*w*2).reshape((h*w*k,2))
    if k>1:
        return ( Halton(d=2).random(k) - 0.5 ) / (np.array((w,h)) - 1)
    return np.array([(0,0)], dtype='float64')

class Tile:
    def __init__(self, src_x, src_y, tile_w, tile_h, image_w, image_h, f, sub):
        self.org = np.array((src_x, src_y), dtype=int)
        self.shape = (tile_h, tile_w)
        self.size = np.array((tile_w, tile_h), dtype=int)
        self.img_size = np.array((image_w, image_h), dtype=int)
        self.sample=f
        self.subsample_offsets = sub

    def get_points(self, x, y, w, h):
        maxx, maxy = self.img_size - 1
        ox, oy = self.org
        return np.array([ off+np.array(((ox+x)/maxx, 1-(oy+y)/maxy)) for y in range(h) for x in range(w) for off in self.subsample_offsets ])

    def render(self):
        points = self.get_points( *self.org, *self.size )
        k = len(self.subsample_offsets)
        if len(points) <= MAX_SAMPLE_BATCH:
            values = self.sample(points)
        else:
            values = np.array([], dtype='float64')
            for points_batch in batched(points, MAX_SAMPLE_BATCH): 
                values = np.concatenate((values, self.sample(points_batch)))
        values = values.reshape((*self.shape, k))
        if k > 1:
            values = values.mean(axis=2, keepdims=True)
        self.buf = values
        return self

def blit(dst, src, dst_y, dst_x):
    rows,cols = src.shape[:2]
    dst[dst_y:dst_y+rows , dst_x:dst_x+cols, :] = src

def render_main(dim, sample, ss):
    tiles_per_proc = 256  # each process gets this many tiles
    tile = 32
    #dim=((dim1[0]+tile-1)//tile*tile,(dim1[1]+tile-1)//tile*tile)

    dimy,dimx = dim
    ntx=(dimx+tile-1)//tile
    nty=(dimy+tile-1)//tile
    nt=ntx*nty

    tiles=[]
    sub=make_subsample_offsets(k=ss, h=dimy, w=dimx)
    for ty in range(nty):
        y=ty*tile
        h=min(y+tile, dimy) - y
        for tx in range(ntx):
            x=tx*tile
            w=min(x+tile, dimx) - x
            tiles.append(Tile(x, y, w, h, dimx, dimy, sample, sub))

    print('Buffer size:', dim)
    print(f'Tile size: {tile}x{tile} = {tile*tile}')
    print('Tiles:', nt)
    print('Tiles/process:', nt/NPROC)

    with multiprocessing.Pool(NPROC) as pool:
        buf = np.zeros((*dim,1), dtype='float64')
        with tqdm(total=dim[0]*dim[1]*ss) as progress:
            for t in pool.map(Tile.render, tiles, chunksize=tiles_per_proc):
                blit(buf, t.buf, dst_y=t.org[1], dst_x=t.org[0])
                progress.update(tile*tile*ss)

    buf = format_output(buf)
    return buf

def colorize(buf):
    def hexcolor(x):
        return np.array([int(x[i:i+2],16)/255.0 for i in (4,2,0)])
    colormap = [
        (0.00, (0,0,0)),
        (0.02, (0,0,0)),
        (0.07, hexcolor('64dd0a')*0.30),
        (0.60, hexcolor('64dd0a')),
        #(0.90, hexcolor('ffc900')),
        (1.00, (1.0,1.0,1.0)),
    ]
    cs = CubicSpline(*zip(*colormap))
    m = cs(buf)
    m = np.clip(m, 0, 1)
    return m.reshape((*buf.shape[:2], 3))

def normalize(buf):
    lo,hi = buf.min(), buf.max()
    return (buf - lo) / (hi - lo)

def format_output(buf):
    buf = normalize(buf)
    buf = colorize(buf)

    hi=0xffff
    buf *= hi

    # noise fixes gradients by dithering when shit software downsamples image to 8bit
    n = 128
    n *= 4 # resist downscaling by half
    buf += np.random.triangular(left=-n, mode=0, right=n, size=buf.shape)

    # cv2.imwrite only writes 16bit png if type is uint16, not float16
    buf = np.clip(buf, 0, hi).astype('uint16')

    return buf

def save(fn, buf):
    if fn.endswith('.bin'):
      with open(fn+'.txt', 'w') as f:
        print(fn, buf.dtype, buf.shape, file=f)
      buf.tofile(fn)
    else:
      cv2.imwrite(fn, buf)

def viewbox_at(center, scale, dimx, dimy):
    d = sqrt( dimx**2 + dimy**2 )*2
    x = dimx / d * scale
    y = dimy / d * scale
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
    ap.add_argument('-p', '--nproc', default=NPROC, type=int,
      help=f'Parallel processes to use [{NPROC}]', metavar='COUNT')
    args=ap.parse_args()

    ss=args.supersample
    ss**=2
    dim=(args.resolution[1], args.resolution[0])
    n_samples=ss*dim[0]*dim[1]
    NPROC=args.nproc

    print('Using processes:', NPROC)
    print('Samples/pixel', ss)
    print('Total samples to calculate:', n_samples)
    print('Outputs', args.output)

    t0=time.time()

    fractal=Julia(c=(0.37+0.1j), viewbox=viewbox_at((0.165,0.12),0.15,dim[1],dim[0]))
    buf=render_main(dim, fractal.sample, ss=ss)

    t1=time.time()
    elapsed=t1-t0
    time_per_sample=1e6*elapsed/n_samples

    print(f'Elapsed: {elapsed:.3f} s')
    print(f'µs/sample: {time_per_sample:.6f} s')

    for filename in args.output:
      print('Saving', filename)
      save(filename, buf)

if __name__=='__main__':
    main()

