import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("omap")
parser.add_argument("-l", "--lknee", type=int,   default=2000)
parser.add_argument("-a", "--alpha", type=float, default=-3)
parser.add_argument("-b", "--beam",  type=str,   default="2.0")
parser.add_argument("-F", "--filter",type=int,   default=1)
parser.add_argument("-A", "--apod",  type=int,   default=0)
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils, uharm
from enlib  import dory

# typically we have lknee = 2000/3000/4000 at f090, f150, f220

beam = dory.get_beam(args.beam)
beam /= np.max(beam)
map  = enmap.read_map(args.imap)
# get rid of nans
map[~np.isfinite(map)] = 0
if args.apod:
	map = enmap.apod(map, args.apod, fill="crossfade")
uht  = uharm.UHT(map.shape, map.wcs)
l    = np.maximum(uht.l, 0.5)
if args.filter > 0: F = (1+(l/args.lknee)**args.alpha)**-1 * uht.lprof2hprof(beam)
mask = map[0] != 0
for I in utils.nditer(map.shape[:-3]):
	# Filter T
	if args.filter > 0:
		map[I][0]  = uht.harm2map(uht.hmul(F, uht.map2harm(map[I][0]), inplace=True))
	# QU -> EB
	map[I][1:] = uht.harm2map(uht.map2harm(map[I][1:], spin=[2]), spin=[0,0])
map *= mask
enmap.write_map(args.omap, map)
