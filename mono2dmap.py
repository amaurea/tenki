from __future__ import division, print_function
import numpy as np, argparse, os
from enlib import utils, enmap
parser = argparse.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("template", nargs="?")
parser.add_argument("odir")
parser.add_argument("-t", "--tsize", type=int, default=720)
parser.add_argument("-s", "--slice", type=str, default=None)
args = parser.parse_args()
tfile = args.template or args.imap

tshape  = (args.tsize,args.tsize)
imap    = enmap.read_map(args.imap)
utils.mkdir(args.odir)
ntile   = tuple([(s+t-1)//t for s,t in zip(imap.shape[-2:],tshape)])

for ty in range(ntile[0]):
	y1 = ty*tshape[0]
	y2 = min((ty+1)*tshape[0], imap.shape[-2])
	for tx in range(ntile[1]):
		x1 = tx*tshape[1]
		x2 = min((tx+1)*tshape[1], imap.shape[-1])
		print(args.odir + "/tile%03d_%03d.fits" % (ty,tx))
		enmap.write_map(args.odir + "/tile%03d_%03d.fits" % (ty,tx), imap[...,y1:y2,x1:x2])

## Find the bounds of our tiling
#shape, wcs = enmap.read_map_geometry(tfile)
#box        = enmap.box(shape, wcs, corner=False)
#tshape     = (args.tsize,args.tsize)
#ntile      = tuple([(s+t-1)//t for s,t in zip(shape[-2:],tshape)])
#
#utils.mkdir(args.odir)
#opathfmt = args.odir + "/tile%(y)03d_%(x)03d.fits"
#
#retile.retile(args.imap, opathfmt, otilenum=ntile, ocorner=box[0],
#		otilesize=tshape, verbose=True, slice=args.slice)
