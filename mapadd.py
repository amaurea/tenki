import numpy as np, argparse
from enlib import enmap, log
parser = argparse.ArgumentParser()
parser.add_argument("imaps", nargs="+")
parser.add_argument("omap")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

L = log.init(level=log.DEBUG if args.verbose else log.ERROR)
def nonan(a):
	res = a.copy()
	res[np.isnan(res)] = 0
	return res
L.info("Reading %s" % args.imaps[0])
m = nonan(enmap.read_map(args.imaps[0]))
for mif in args.imaps[1:]:
	L.info("Reading %s" % mif)
	m += nonan(enmap.read_map(mif))
L.info("Writing %s" % args.omap)
enmap.write_map(args.omap, m)
