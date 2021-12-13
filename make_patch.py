import argparse
parser = argparse.ArgumentParser()
parser.add_argument("box", type=str)
parser.add_argument("ofile")
parser.add_argument("-r", "--res",  type=float, default=0.5)
parser.add_argument("-p", "--proj", type=str, default="car")
args = parser.parse_args()
import numpy as np
from pixell import utils, enmap
box = utils.parse_box(args.box)*utils.degree
shape, wcs = enmap.geometry(box, res=args.res*utils.arcmin, proj=args.proj)
enmap.write_map(args.ofile, enmap.zeros(shape, wcs, np.uint8))
