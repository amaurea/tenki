import argparse
parser = argparse.ArgumentParser()
parser.add_argument("box", type=str)
parser.add_argument("ofile")
parser.add_argument("-r", "--res",  type=float, default=0.5)
parser.add_argument("-p", "--proj", type=str, default="car")
parser.add_argument("-f", "--full", action="store_true")
parser.add_argument("-V", "--variant", type=str, default=None)
args = parser.parse_args()
import numpy as np
from pixell import utils, enmap
box = utils.parse_box(args.box)*utils.degree
shape, wcs = enmap.geometry(box, res=args.res*utils.arcmin, proj=args.proj)
if   args.variant is None: pass
elif args.variant == "fejer1": wcs.wcs.crpix[1] -= 0.5
else: raise ValueError("Unknown variant '%s'" % str(args.variant))
if args.full:
	enmap.write_map(args.ofile, enmap.zeros(shape, wcs, np.uint8))
else:
	enmap.write_map_geometry(args.ofile, shape, wcs)
