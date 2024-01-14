import os
import time
import numpy as np
import cv2
import multiprocessing
import multiprocessing.dummy
from scipy.interpolate import CubicSpline
from scipy.stats.qmc import Halton
from argparse import ArgumentParser
from math import log, sqrt
from tqdm import tqdm

try:
    import cupy
except ImportError:
    print('failed to import cupy')

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
MAX_SAMPLE_BATCH=1024

class Julia:
    def __init__(self, c, viewbox, max_iter=1000):
        """ viewbox: (min real, min imag, max real, max imag) """
        self.c = np.complex128(c)
        self.max_iter = max_iter
        self.view_org = np.array((viewbox[0], viewbox[1]))
        self.view_scale = np.array((viewbox[2]-viewbox[0], viewbox[3]-viewbox[1]))

    def sample(self, tile):
        whatever, points = tile
        if len(points) <= MAX_SAMPLE_BATCH:
            values = self.sample1(points)
        else:
            values = np.array([], dtype=tile[1].dtype)
            for B in batched(points, MAX_SAMPLE_BATCH): 
                values = np.concatenate((values, self.sample1(B)))
        return whatever, values

    def sample1(self, points):
        """ https://linas.org/art-gallery/escape/escape.html
        points: 2d numpy array of points (float)
        output: 1d numpy array of intensity
        """
        z = self.view_org + self.view_scale*points
        if z.dtype == 'float64':
            z = z.flatten().view(dtype=np.complex128)
            c = np.complex128(self.c)
        else:
            assert z.dtype == 'float32'
            z = z.flatten().view(dtype=np.complex64)
            c = np.complex64(self.c)
        n = np.zeros(len(points), dtype=int)
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

    def sample_cupy(self, tile):
        whatever, points = tile
        px,py = cupy.asarray(points, dtype='float32').T
        result = cupy.zeros(len(points), dtype='float32')
        result = self.cupy_kernel(px, py, result)
        return whatever, result

    def init_cupy(self):
        c=self.c
        # line 1 is line 16
        # i is illegal variable name (cant redeclare)
        self.cupy_kernel = cupy.ElementwiseKernel(\
in_params = 'float32 px, float32 py',\
out_params ='float32 y', \
operation = f'''
double za=0, zr, zi, zr1, zi1;
int n;
zr = {self.view_org[0]} + {self.view_scale[0]}*(double)px;
zi = {self.view_org[1]} + {self.view_scale[1]}*(double)py;
for(n=0; n<{self.max_iter}; n+=1) {{
  zr1 = zr*zr - zi*zi + {c.real};
  zi1 = 2*zr*zi + {c.imag};
  zr = zr1;
  zi = zi1;
  za = zr*zr + zi*zi;
  if (za > 4.0) break;
}}
za = max(za, 1.0000001);
za = sqrt(za);
y = n - log(log(za)) * {1.0/np.log(2)} ;
''', \
name='julia')

def coords_gen(dim, tile, mode, k, cu=np, dtype='float64'):
    dimy,dimx = dim
    ntx=(dimx+tile-1)//tile
    nty=(dimy+tile-1)//tile
    if mode=='count':
        yield (ntx, nty, ntx*nty)
        return
    subpos=(Halton(d=2).random(k)-0.5)/np.array((dimx,dimy))
    subpos=cu.asarray(subpos)
    for ty in range(nty):
        for tx in range(ntx):
            points=None
            for yy in range(tile):
                y=ty*tile + yy
                if y>=dimy:
                    break
                for xx in range(tile):
                    x=tx*tile + xx
                    if x>=dimx:
                        break
                    co=(x/(dimx-1), 1-y/(dimy-1))
                    p=subpos + cu.array(co, dtype=dtype)
                    points=p if points is None else cu.concatenate((points, p))
            yield (ty*tile, tx*tile), points

def blit(dst, src, dst_y=0, dst_x=0):
    rows,cols = src.shape[:2]
    dst[dst_y:dst_y+rows , dst_x:dst_x+cols] = src

def render_main(dim1, fractal, subsamples, tile, cupy_device=None):
    dim=(dim1[0]+tile-1)//tile*tile , (dim1[1]+tile-1)//tile*tile
    ntx,nty,coords_n = next(coords_gen(dim, tile, 'count', subsamples))
    print('Buffer size:', dim)
    print(f'Tile size: {tile}x{tile}x{subsamples} = {tile*tile*subsamples}')
    print('Tiles:', coords_n)
    print('Tiles/process:', coords_n/NPROC)

    with tqdm(total=dim[0]*dim[1]*subsamples) as progress:
        if cupy_device is not None:
            tiles_per_proc = 64
            threads = min(NPROC, 4)
            print('cupy mode, device:', cupy_device)

            with cupy.cuda.Device(cupy_device):
                buf = cupy.zeros(dim, dtype='float32') # fail here if not enough GPU memory
                del buf
                buf = None
                buf_row = []
                fractal.init_cupy()
                coords = coords_gen(dim, tile, 'gen', subsamples, cu=np, dtype='float32')
                with tqdm(total=dim[0]*dim[1]*subsamples) as progress:
                    with multiprocessing.dummy.Pool(threads) as pool:
                        for tile_pos, result in pool.imap(fractal.sample_cupy, coords, chunksize=tiles_per_proc):
                            p = len(result)
                            mean = result.reshape((tile,tile,subsamples)).mean(axis=2,keepdims=True).reshape((tile,tile))
                            buf_row += [mean]
                            if len(buf_row) == ntx:
                                temp = cupy.concatenate(buf_row, axis=1)
                                buf = temp if buf is None else cupy.concatenate((buf, temp), axis=0)
                                buf_row = []
                            progress.update(p)
                buf = cupy.asnumpy(buf)
        else:
            print('CPU-only mode. Using processes:', NPROC)
            tiles_per_proc = 256
            buf = np.zeros(dim, dtype='float32')

            coords = coords_gen(dim, tile, 'gen', subsamples, cu=np, dtype='float64')
            with multiprocessing.Pool(NPROC) as pool:
                for tile_pos, result in pool.imap(fractal.sample, coords, chunksize=tiles_per_proc):
                    p=len(result)
                    mean = result.reshape((tile,tile,subsamples)).mean(axis=2,keepdims=True).reshape((tile,tile))
                    blit(buf, mean, *tile_pos)
                    progress.update(p)

    buf = buf.reshape((dim[0],dim[1],-1))
    buf = buf[:dim1[0],:dim1[1],:]
    return buf

def colorize(buf):
    def hexcolor(x):
        return np.array([int(x[i:i+2],16)/255.0 for i in (4,2,0)])
    colormap = [
        (0.00, (0,0,0)),
        (0.02, (0,0,0)),
        (0.07, hexcolor('64dd0a')*0.30),
        (0.90, hexcolor('64dd0a')),
        #(0.90, hexcolor('ffc900')),
        (1.00, (1.0,1.0,1.0)),
    ]
    cs = CubicSpline(*zip(*colormap))
    m = cs(buf)
    m = np.clip(m, 0, 1)
    return m.reshape((*buf.shape[:2], 3))

def normalize(buf):
    #fixed constant here keeps brightness the same with different resolutions
    return (buf-1)/950;
    lo,hi = buf.min(), buf.max()
    return (buf - lo) / (hi - lo)

def add_noise(buf):
    fn='image/blue.png' # from https://graemephi.github.io/posts/some-low-discrepancy-noise-functions/
    if os.path.exists(fn):
        n = cv2.imread(fn).astype('float64') - 127.5
        h,w = buf.shape[:2]
        nh,nw = n.shape[:2]
        for y in range((h+nh-1)//nh):
            y0=y*nh
            y1=min(y0+nh, h)
            for x in range((w+nw-1)//nw):
                x0=x*nw
                x1=min(x0+nw, w)
                buf[y0:y1 , x0:x1 , :] += n[:y1-y0 , :x1-x0]
    else:
        print('did not find', fn, 'using triangular noise instead')
        n = 128
        buf += np.random.triangular(left=-n, mode=0, right=n, size=buf.shape)
    return buf

def format_output(buf):
    buf = normalize(buf)
    buf = colorize(buf)

    hi=0xffff
    buf *= hi

    # noise fixes gradients by dithering when shit software downsamples image to 8bit
    buf = add_noise(buf)

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
    ap.add_argument('-c', '--cupy', default=False, action='store_true',
      help='Calculate on video card using cupy')
    ap.add_argument('-d', '--cupy-device', default=0, type=int,
      help='Set device index [0]', metavar='ID')
    ap.add_argument('-t', '--tile-size', default=4, type=int,
      help='Tile size NxN [4]', metavar='N')
    args=ap.parse_args()

    ss=args.supersample
    ss=ss*ss
    dim=(args.resolution[1], args.resolution[0])
    n_samples=ss*dim[0]*dim[1]
    NPROC=args.nproc

    print('Samples/pixel', ss)
    print('Total samples to calculate:', n_samples)
    print('Outputs', args.output)

    t0=time.time()

    fractal=Julia(c=(0.37+0.1j), viewbox=viewbox_at((0.165,0.12),0.15,dim[1],dim[0]))
    buf=render_main(dim, fractal, ss, args.tile_size, cupy_device=args.cupy_device if args.cupy else None)

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

