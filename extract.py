import argparse
parser = argparse.ArgumentParser()
parser.add_argument("box")
parser.add_argument("ifiles", nargs="+")
parser.add_argument("out")
args = parser.parse_args()
import numpy as np, os, glob
from pixell import enmap, utils

ifiles = sum([sorted(glob.glob(ifile)) for ifile in args.ifiles],[])
box    = utils.parse_box(args.box)*utils.degree

for fi, ifile in enumerate(ifiles):
	if len(ifiles) == 1: ofile = args.out
	else:                ofile = args.out + "/" + os.path.basename(ifile)
	print(ofile)
	map = enmap.read_map(ifile, box=box)
	enmap.write_map(ofile, map)
	del map
