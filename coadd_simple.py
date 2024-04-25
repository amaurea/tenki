from __future__ import division, print_function
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imaps_and_hits", nargs="+", help="map map map ... hits hits hits ... unless --transpose, in which case it's map hits map hits map hits ...")
parser.add_argument("omap")
parser.add_argument("ohit")
parser.add_argument("-T", "--transpose",     action="store_true")
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils

n = len(args.imaps_and_hits)//2
if not args.transpose:
	imaps = args.imaps_and_hits[:n]
	ihits = args.imaps_and_hits[n:]
else:
	imaps = args.imaps_and_hits[0::2]
	ihits = args.imaps_and_hits[1::2]

omap = 0
ohit = 0
for imapfile, ihitfile in zip(imaps,ihits):
	print("Reading %s" % imapfile)
	imap  = enmap.read_map(imapfile)
	print("Reading %s" % ihitfile)
	ihit  = enmap.read_map(ihitfile).preflat[0]
	omap += imap*ihit
	ohit += ihit
	del imap, ihit

print("Solving")
with utils.nowarn():
	omap /= ohit
	omap = np.nan_to_num(omap, copy=False, nan=0, posinf=0, neginf=0)

print("Writing %s" % args.omap)
enmap.write_map(args.omap, omap); del omap
print("Writing %s" % args.ohit)
enmap.write_map(args.ohit, ohit); del ohit
