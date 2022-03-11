import argparse
parser = argparse.ArgumentParser()
parser.add_argument("box_or_template")
parser.add_argument("ifiles", nargs="+")
parser.add_argument("out")
args = parser.parse_args()
import numpy as np, os, glob
from pixell import enmap, utils

ifiles = sum([sorted(glob.glob(ifile)) for ifile in args.ifiles],[])
kwargs = {}
try:
	kwargs["box"] = utils.parse_box(args.box_or_template)*utils.degree
except ValueError:
	kwargs["geometry"] = enmap.read_map_geometry(args.box_or_template)

for fi, ifile in enumerate(ifiles):
	if len(ifiles) == 1: ofile = args.out
	else:                ofile = args.out + "/" + os.path.basename(ifile)
	print(ofile)
	map = enmap.read_map(ifile, **kwargs)
	enmap.write_map(ofile, map)
	del map
