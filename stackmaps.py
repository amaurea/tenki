import numpy as np, argparse
from enlib import enmap, log
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("ofile")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

L = log.init(level=log.DEBUG if args.verbose else log.ERROR)

maps = []
for ifile in args.ifiles:
	L.info("Reading %s" % ifile)
	maps.append(enmap.read_map(ifile))
L.info("Stacking")
maps = enmap.samewcs(maps, maps[0])
L.info("Writing %s" % args.ofile)
enmap.write_map(args.ofile, maps)
