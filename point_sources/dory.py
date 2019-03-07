# This program is a simple alternative to nemo. It uses the same
# constant-covariance noise model which should make it fast but
# suboptimal. I wrote this because I was having too much trouble
# with nemo. The aim is to be fast and simple to implement.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["find","fit"], help="What operation to perform. 'find': Find sources in map and build a new catalog. 'fit': Fit source amplitudes in map using an existing catalog. In this case the icat argument must be provided.")
parser.add_argument("imap", help="The map to find sources in. Should be enmap-compatible.")
parser.add_argument("idiv", help="The inverse variance per pixel of imap.")
parser.add_argument("icat", nargs="?", help="The input point source catalog for amplitude fitting. Only used in 'fit' mode.")
parser.add_argument("odir", help="The directory to write the output to. Will be crated if necessary.")
parser.add_argument("-m", "--mask",    type=str,   default=None, help="The mask to apply when finding sources, typically a galactic mask to avoid identifying dust as sources. Should be the same shape as the map. 0 is unmasked, 1 is masked.")
parser.add_argument("-b", "--beam",    type=str,   default="1.4",help="The beam of the map. Should be a 1d harmonic transfer function.")
parser.add_argument("-R", "--regions", type=str,   default=None, help="Which regions to consider to have comogeneous noise correlations. 'full': Use whole map, 'tile:npix': split map into npix*npix sized tiles. Or specify the file name to a ds9 region file containing boxes.")
parser.add_argument("-a", "--apod",    type=int,   default=30, help="The width of the apodization region, in pixels.")
parser.add_argument("--apod-margin",   type=int,   default=10, help="How far away from the apod region a source should be to be valid.")
parser.add_argument("-s", "--nsigma",  type=float, default=3.5, help="The number a sigma a source must be to be included in the catalog when finding sources.")
parser.add_argument("-p", "--pad",     type=int,   default=60, help="The number of pixels to extend each region by in each direciton, to avoid losing sources at region boundaries. Should be larger than apod+apod_margin")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-o", "--output",  type=str,   default="full,reg", help="What types of output to write to the output directory. Comma-separated list. 'full': Output the final, merged quantities. 'reg': Output per-region results. 'debug': Output lots of debug maps.")
parser.add_argument(      "--ncomp",   type=int,   default=1, help="The number of stokes components to fit for in fit mode.")
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

def divdiag(div):
	if   div.ndim == 2: return div.preflat
	elif div.ndim == 3: return div
	elif div.ndim == 4: return enmap.samewcs(np.einsum("aayx->ayx",div),div)
	else: raise ValueError("Invalid div shape: %s" % div.shape)

if args.mode == "find":
	# Point source finding mode
	results  = []
	map_keys = ["map","snmap","model","resid","resid_snmap"]
	# Mpi-parallelization over regions is simple, but a bit lazy. It means that there will be
	# no speedup for single-region maps
	for ri in range(comm.rank, len(regions), comm.size):
		reg_fid = regions[ri]
		reg_pad = dory.pad_region(reg_fid, args.pad)
		print "%3d region %3d %5d %5d %6d %6d" % (comm.rank, ri+1, reg_fid[0,0], reg_fid[1,0], reg_fid[0,1], reg_fid[1,1])
		try:
			# We only use T to find sources in find mode for now. P usually has much lower S/N and
			# doesn't help much.
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

elif args.mode == "fit":
	icat = dory.read_catalog(args.icat)
	for ri in range(comm.rank, len(regions), comm.size):
		reg_fid = regions[ri]
		reg_pad = dory.pad_region(reg_fid, args.pad)
		print "%3d region %3d %5d %5d %6d %6d" % (comm.rank, ri+1, reg_fid[0,0], reg_fid[1,0], reg_fid[0,1], reg_fid[1,1])
		try:
			# We support polarization here, but treat each component independently
			imap   = enmap.read_map(args.imap, pixbox=reg_pad).preflat[:args.ncomp]
			idiv   = divdiag(enmap.read_map(args.idiv, pixbox=reg_pad))[:args.ncomp]
			if args.mask: idiv *= (1-enmap.read_map(args.mask, pixbox=reg_pad).preflat[0])
			if "debug" in args.output: dump_prefix = args.odir + "/region_%02d_" % ri
			else:                      dump_prefix = None
			result = dory.fit_srcs(imap, idiv, icat, beam, apod=args.apod, apod_margin=args.apod_margin,
					verbose=args.verbose, dump=dump_prefix)
		except Exception as e:
			print "Exception for task %d region %d: %s" % (comm.rank, ri, e.message)
			raise
