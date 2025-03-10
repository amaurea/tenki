import argparse
parser = argparse.ArgumentParser()
parser.add_argument("box_or_template")
parser.add_argument("ifiles", nargs="+")
parser.add_argument("out")
parser.add_argument("-F", "--fix-wcs", action="store_true")
parser.add_argument("--op", type=str, default=None)
args = parser.parse_args()
import numpy as np, os
from pixell import enmap, utils, wcsutils

ifiles = sum([sorted(utils.glob(ifile)) for ifile in args.ifiles],[])
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
	if args.fix_wcs:
		map.wcs = wcsutils.fix_wcs(map.wcs)
	if args.op is not None:
		map = eval(args.op, {"m":map,"enmap":enmap,"utils":utils,"np":np},np.__dict__)
	enmap.write_map(ofile, map)
	del map
