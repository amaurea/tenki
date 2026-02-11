import argparse
parser = argparse.ArgumentParser()
parser.add_argument("fpfile")
parser.add_argument("geometry")
parser.add_argument("omap")
parser.add_argument("-C", "--cols", type=str,   default="0,1,2", help="0-based index of xi, eta and value columns")
parser.add_argument("-r", "--psize",type=float, default=1, help="Point radius in arcmin")
parser.add_argument("-b", "--bgval",type=float, default=0)
parser.add_argument(      "--dp",   default=0, action="count")
parser.add_argument(      "--sp",   default=0, action="count")
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils

dtype = np.float32 if args.sp >= args.dp else np.float64
shape, wcs = enmap.read_map_geometry(args.geometry)
fpfile = args.fpfile
# Make it easy to pipe result of row-filter to command
if fpfile == "-": fpfile = "/dev/stdin"
cols   = utils.parse_ints(args.cols)
data   = np.loadtxt(fpfile, usecols=cols).T
poss   = data[1::-1]*utils.degree
# The value we append represents the unhit background
vals   = np.concatenate([data[2],[args.bgval]]).astype(dtype)

rmap, domains = enmap.distance_from(shape, wcs, poss, domains=True, rmax=args.psize*utils.arcmin)
omap = enmap.enmap(vals[domains], wcs)
enmap.write_map(args.omap, omap)
