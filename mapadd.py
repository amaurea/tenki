from __future__ import division, print_function
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imaps", nargs="+")
parser.add_argument("omap")
parser.add_argument("-m", "--mean",    action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-s", "--scale",   type=str, default=None)
args = parser.parse_args()
import numpy as np, os
from enlib import enmap, log, mpi, utils

scales = np.full(len(args.imaps), 1.0)
if args.scale:
	for i, word in enumerate(args.scale.split(":")):
		scales[i:] = float(word)

comm  = mpi.COMM_WORLD
def nonan(a):
	res = a.copy()
	res[np.isnan(res)] = 0
	return res
def add_maps(imaps, omap):
	if args.verbose: print("Reading %s" % imaps[0])
	m = nonan(enmap.read_map(imaps[0]))*scales[0]
	for scale, mif in zip(scales[1:],imaps[1:]):
		if args.verbose: print("Reading %s" % mif)
		m2 = nonan(enmap.read_map(mif, geometry=(m.shape,m.wcs)))*scale
		n  = min(len(m.preflat),len(m2.preflat))
		m.preflat[:n] += m2.preflat[:n]
	if args.mean: m /= len(imaps)
	if args.verbose: print("Writing %s" % omap)
	enmap.write_map(omap, m)
def get_tilenames(dir):
	return sorted([name for name in os.listdir(dir) if name.endswith(".fits") or name.endswith(".hdf")])

# Two cases: Normal enmaps or dmaps
if not os.path.isdir(args.imaps[0]):
	# Normal monotlithic map
	if comm.rank == 0:
		add_maps(args.imaps, args.omap)
else:
	# Dmap. Each name is actually a directory, but they
	# all have compatible tile names.
	tilenames = get_tilenames(args.imaps[0])
	utils.mkdir(args.omap)
	for tilename in tilenames[comm.rank::comm.size]:
		timaps = ["%s/%s" % (imap,tilename) for imap in args.imaps]
		print("%3d %s" % (comm.rank, tilename))
		add_maps(timaps, args.omap + "/" + tilename)
	if args.verbose: print("Done")
