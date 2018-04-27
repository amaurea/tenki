from __future__ import division, print_function
import numpy as np, argparse, os
from enlib import retile, mpi, utils, enmap
parser = argparse.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("template", nargs="?")
parser.add_argument("odir")
parser.add_argument("-t", "--tsize", type=int, default=675)
parser.add_argument("-s", "--slice", type=str, default=None)
args = parser.parse_args()
tfile = args.template or args.imap

# Find the bounds of our tiling
shape, wcs = enmap.read_map_geometry(tfile)
box        = enmap.box(shape, wcs, corner=False)
tshape     = (args.tsize,args.tsize)
ntile      = tuple([(s+t-1)//t for s,t in zip(shape[-2:],tshape)])

utils.mkdir(args.odir)
opathfmt = args.odir + "/tile%(y)03d_%(x)03d.fits"

retile.retile(args.imap, opathfmt, otilenum=ntile, ocorner=box[0],
		otilesize=tshape, verbose=True, slice=args.slice)
