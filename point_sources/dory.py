# This program is a simple alternative to nemo. It uses the same
# constant-covariance noise model which should make it fast but
# suboptimal. I wrote this because I was having too much trouble
# with nemo. The aim is to be fast and simple to implement.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("idiv")
parser.add_argument("odir")
parser.add_argument("-m", "--mask",    type=str,   default=None)
parser.add_argument("-b", "--beam",    type=str,   default="1.4")
parser.add_argument("-R", "--regions", type=str,   default=None)
parser.add_argument("-a", "--apod",    type=int,   default=30)
parser.add_argument("--apod-margin",   type=int,   default=10)
parser.add_argument("-s", "--nsigma",  type=float, default=3.5)
parser.add_argument("-p", "--pad",     type=int,   default=60)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-o", "--output",  type=str,   default="full,reg")
args = parser.parse_args()
import numpy as np
from enlib import dory
from enlib import enmap, utils, bunch, mpi, fft

comm       = mpi.COMM_WORLD
shape, wcs = enmap.read_map_geometry(args.imap)
beam       = dory.get_beam(args.beam)
regions    = dory.get_regions(args.regions, shape, wcs)
ps_res     = 200
utils.mkdir(args.odir)

# Process each region. Will do mpi here for now. This is a bit lazy - it
# completely skips any mpi speedups for single-region maps. Everything is
# T-only for now too
results  = []
map_keys = ["map","snmap","model","resid","resid_snmap"]
for ri in range(comm.rank, len(regions), comm.size):
	reg_fid = regions[ri]
	reg_pad = dory.pad_region(reg_fid, args.pad)
	print "%3d region %3d %5d %5d %6d %6d" % (comm.rank, ri+1, reg_fid[0,0], reg_fid[1,0], reg_fid[0,1], reg_fid[1,1])
	try:
		imap   = enmap.read_map(args.imap, pixbox=reg_pad).preflat[0]
		idiv   = enmap.read_map(args.idiv, pixbox=reg_pad).preflat[0]
		if args.mask: idiv *= (1-enmap.read_map(args.mask, pixbox=reg_pad).preflat[0])
		if "debug" in args.output: dump_prefix = args.odir + "/region_%02d_" % ri
		else:                      dump_prefix = None
		result = dory.find_srcs(imap, idiv, beam, apod=args.apod, apod_margin=args.apod_margin,
				snmin=args.nsigma, verbose=args.verbose, dump=dump_prefix)
		# FIXME: artifacts are act-specific
		result = dory.prune_artifacts(result)
	except Exception as e:
		print "Exception for task %d region %d: %s" % (comm.rank, ri, e.message)
		raise
	# Write region output
	if "reg" in args.output:
		prefix = args.odir + "/region_%02d_" % ri
		dory.write_catalog(prefix + "cat.txt" , result.cat)
		for name in map_keys:
			enmap.write_map(prefix + name + ".fits", result[name])
		if "full" in args.output:
			results.append(result)

if "full" in args.output:
	# First build the full catalog
	if len(results) > 0:
		my_cat   = np.concatenate([result.cat for result in results])
	else:
		my_cat   = np.zeros([0], dory.cat_dtype)
	tot_cat  = dory.allgather_catalog(my_cat, comm)
	# FIXME: duplicate merging radius depends on beam size
	tot_cat  = dory.merge_duplicates(tot_cat)
	if comm.rank == 0:
		print "Writing catalogue"
		dory.write_catalog(args.odir + "/cat.txt", tot_cat)
	# Then build the full maps
	for name in map_keys:
		if comm.rank == 0: print "Writing %s" % name
		merged = dory.merge_maps_onto([result[name] for result in results], shape, wcs, comm,
				root=0, crop=args.apod+args.apod_margin)
		if comm.rank == 0: enmap.write_map(args.odir + "/%s.fits" % name, merged)
		del merged
