import time
import numpy as np
import cv2
from scipy import ndimage
from scipy.interpolate import CubicSpline
import multiprocessing
from argparse import ArgumentParser
from math import log, sqrt
from tqdm import tqdm

NPROC=multiprocessing.cpu_count()

class Julia:
    def __init__(self, c, viewbox, max_iter=1000):
        """ viewbox: (min real, min imag, max real, max imag) """
        self.c = np.complex128(c)
        self.max_iter = max_iter
        self.view_org = np.array((viewbox[0], viewbox[1]))
        self.view_scale = np.array((viewbox[2]-viewbox[0], viewbox[3]-viewbox[1]))

    def sample(self, tile):
        """ https://linas.org/art-gallery/escape/escape.html
        points: 2d numpy array of points (float)
        output: 1d numpy array of intensity
        """
        whatever, points = tile
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
        return whatever , n - np.log(np.log(za)).flatten() / np.log(2)

def coords_gen(dim, tile, mode):
    dimy,dimx = dim
    ntx=(dimx+tile-1)//tile
    nty=(dimy+tile-1)//tile
    if mode=='count':
        yield (ntx, nty, ntx*nty)
        return
    for ty in range(nty):
        for tx in range(ntx):
            points=[]
            for yy in range(tile):
                y=ty*tile + yy
                if y>=dimy:
                    break
                for xx in range(tile):
                    x=tx*tile + xx
                    if x>=dimx:
                        break
                    points.append( (x/(dimx-1), 1-y/(dimy-1)) )
            yield (ty*tile, tx*tile), np.array(points, dtype='float64')

def blit(dst, src, dst_y, dst_x):
    rows,cols = src.shape[:2]
    dst[dst_y:dst_y+rows , dst_x:dst_x+cols] = src

def render(dim, sample):
    tiles_per_proc = 256  # each process gets this many tiles
    tile = 32
    ntx,nty,coords_n = next(coords_gen(dim, tile, 'count'))
    coords = coords_gen(dim, tile, 'gen')
    print('Buffer size:', dim)
    print(f'Tile size: {tile}x{tile} = {tile*tile}')
    print('Tiles:', coords_n)
    print('Tiles/process:', coords_n/NPROC)

    with multiprocessing.Pool(NPROC) as pool:
        buf = np.zeros(dim, dtype='float64')
        with tqdm(total=dim[0]*dim[1]) as progress:
            for tile_pos, result in pool.imap(sample, coords, chunksize=tiles_per_proc):
                p=len(result)
                blit(buf, result.reshape((tile,tile)), *tile_pos)
                progress.update(p)

    buf = buf.reshape((dim[0],dim[1],-1))
    return buf

def render_ss(dim, sample, ss):
    dim2=(dim[0]*ss, dim[1]*ss)
    buf=render(dim2, sample)
    if ss>1:
        print('Downsampling...')
        buf=ndimage.zoom(buf, (1/ss,1/ss,1), order=3, prefilter=True)
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

def save(fn, buf, cvbuf):
    if fn.endswith('.bin'):
      fn2=fn+'.u16'
      with open(fn+'.txt', 'w') as f:
        print(fn, buf.dtype, buf.shape, file=f)
        print(fn2, cvbuf.dtype, cvbuf.shape, file=f)
      buf.tofile(fn)
      cvbuf.tofile(fn2)
    else:
      cv2.imwrite(fn, cvbuf)

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
    dim=(args.resolution[1], args.resolution[0])
    n_samples=ss*ss*dim[0]*dim[1]
    NPROC=args.nproc

    print('Using processes:', NPROC)
    print('Samples/pixel', ss**2)
    print('Total samples to calculate:', n_samples)
    print('Outputs', args.output)

    t0=time.time()

    fractal=Julia(c=(0.37+0.1j), viewbox=viewbox_at((0.165,0.12),0.15,dim[1],dim[0]))
    buf=render_ss(dim, fractal.sample, ss)

    t1=time.time()
    elapsed=t1-t0
    time_per_sample=1e6*elapsed/n_samples

    print(f'Elapsed: {elapsed:.3f} s')
    print(f'µs/sample: {time_per_sample:.6f} s')

    print('Output conversion...')
    cvbuf = format_output(buf)

    for filename in args.output:
      print('Saving', filename)
      save(filename, buf, cvbuf)

if __name__=='__main__':
    main()

