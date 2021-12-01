import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dec1", type=float)
parser.add_argument("dec2", type=float)
parser.add_argument("ra1",  type=float)
parser.add_argument("ra2",  type=float)
parser.add_argument("ofile")
parser.add_argument("-r", "--res",  type=float, default=0.5)
parser.add_argument("-p", "--proj", type=str, default="car")
args = parser.parse_args()
import numpy as np
from pixell import utils, enmap
box = np.array([[args.dec1,args.ra1],[args.dec2,args.ra2]])*utils.degree
shape, wcs = enmap.geometry(box, res=args.res*utils.arcmin, proj=args.proj)
enmap.write_map(args.ofile, enmap.zeros(shape, wcs, np.uint8))
