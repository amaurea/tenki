import numpy as np, argparse, os
from enlib import enmap, log, mpi, utils
parser = argparse.ArgumentParser()
parser.add_argument("imaps", nargs="+")
parser.add_argument("omap")
parser.add_argument("-m", "--mean",    action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

comm  = mpi.COMM_WORLD
def nonan(a):
	res = a.copy()
	res[np.isnan(res)] = 0
	return res
def add_maps(imaps, omap):
	if args.verbose: print "Reading %s" % imaps[0]
	m = nonan(enmap.read_map(imaps[0]))
	for mif in imaps[1:]:
		if args.verbose: print "Reading %s" % mif
		m += nonan(enmap.read_map(mif))
	if args.mean: m /= len(imaps)
	if args.verbose: "Writing %s" % omap
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
		print "%3d %s" % (comm.rank, tilename)
		add_maps(timaps, args.omap + "/" + tilename)
	if args.verbose: print"Done"
