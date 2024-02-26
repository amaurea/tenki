import numpy as np, argparse, os, healpy
from enlib import utils, enmap, curvedsky, log, reproject
parser = argparse.ArgumentParser()
parser.add_argument("ihealmap")
parser.add_argument("template")
parser.add_argument("oenmap")
parser.add_argument("-n", "--ncomp", type=int, default=1)
parser.add_argument("-i", "--first", type=int, default=0)
parser.add_argument("-v", "--verbosity", type=int, default=2)
parser.add_argument("-r", "--rot",   type=str, default=None)
parser.add_argument("-l", "--lmax",  type=int, default=0)
parser.add_argument("-m", "--method",type=str, default="harm")
parser.add_argument("-u", "--unit",  type=float, default=1)
parser.add_argument("-O", "--order", type=int, default=1)
parser.add_argument("-e", "--extensive", action="store_true")
args = parser.parse_args()

log_level = log.verbosity2level(args.verbosity)
L = log.init(level=log_level)

ncomp = args.ncomp
shape, wcs = enmap.read_map_geometry(args.template)
shape = (ncomp,)+shape[-2:]
dtype = np.float32

assert ncomp == 1 or ncomp == 3, "Only 1 or 3 components supported"

# Read the input maps
L.info("Reading " + args.ihealmap)
hmap = np.atleast_2d(healpy.read_map(args.ihealmap, field=tuple(range(args.first,args.first+ncomp))))
hmap = hmap.astype(dtype)
mask    = hmap < -1e20
hmap[mask] = np.mean(hmap[~mask])
if args.unit != 1: hmap /= args.unit
# Perform transform
L.info("Reprojecting")
omap = reproject.healpix2map(hmap, shape, wcs, rot=args.rot, method=args.method, order=args.order, extensive=args.extensive, verbose=args.verbosity>1)
del hmap

# And output
L.info("Writing " + args.oenmap)
enmap.write_map(args.oenmap, omap)
