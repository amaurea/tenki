import numpy as np, argparse
from enlib import enmap, utils
parser = argparse.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("template")
parser.add_argument("omap")
parser.add_argument("-O", "--order", type=int,   default=3)
parser.add_argument("-m", "--mode",  type=str,   default="constant")
parser.add_argument("-M", "--mem",   type=float, default=1e8)
args = parser.parse_args()

imap  = enmap.read_map(args.imap)
shape, wcs = enmap.read_map_geometry(args.template)
omap  = enmap.zeros(shape, wcs, imap.dtype)

blockpix = np.product(shape[:-2])*shape[-1]
bsize = max(1,utils.nint(args.mem/(blockpix*imap.dtype.itemsize)))

nblock = (shape[-2]+bsize-1)//bsize
for b in range(nblock):
	r1, r2 = b*bsize, (b+1)*bsize
	osub = omap[...,r1:r2,:]
	omap[...,r1:r2,:] = enmap.project(imap, osub.shape, osub.wcs, order=args.order, mode=args.mode)
#o = enmap.project(m, t.shape, t.wcs, order=args.order, mode=args.mode)
enmap.write_map(args.omap, omap)
