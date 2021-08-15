from __future__ import division, print_function
import numpy as np, argparse
from pixell import enmap, utils
parser = argparse.ArgumentParser()
parser.add_argument("imaps_and_hits", nargs="+", help="map map map ... hits hits hits ... unless --transpose, in which case it's map hits map hits map hits ...")
parser.add_argument("omap")
parser.add_argument("ohit")
parser.add_argument("-T", "--transpose",     action="store_true")
args = parser.parse_args()

n = len(args.imaps_and_hits)//2
if not args.transpose:
	imaps = args.imaps_and_hits[:n]
	ihits = args.imaps_and_hits[n:]
else:
	imaps = args.imaps_and_hits[0::2]
	ihits = args.imaps_and_hits[1::2]

omap = None
for imapfile, ihitfile in zip(imaps,ihits):
	print("Reading %s" % imapfile)
	imap = enmap.read_map(imapfile)
	print("Reading %s" % ihitfile)
	ihit = enmap.read_map(ihitfile)
	if omap is None:
		omap = imap*0
		ohit = ihit*0
	omap += imap*ihit
	ohit += ihit

print("Solving")
with utils.nowarn():
	omap /= ohit
	omap[~np.isfinite(omap)] = 0

print("Writing %s" % args.omap)
enmap.write_map(args.omap, omap)
print("Writing %s" % args.ohit)
enmap.write_map(args.ohit, ohit)
