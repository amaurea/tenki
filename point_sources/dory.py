# This program is a simple alternative to nemo. It uses the same
# constant-covariance noise model which should make it fast but
# suboptimal. I wrote this because I was having too much trouble
# with nemo. The aim is to be fast and simple to implement.

import argparse, sys
help_general = """Usage:
  dory find     imap idiv odir
  dory fit      imap idiv icat odir
  dory subtract imap icat omap [omodel]

Dory is a simple point source finding/subtracting tool with 3 main modes.
* Find mode takes a map and its white noise inverse covariance and produces a point source catalog
* Fit mode is similar, but fits the amplitudes of point sources given in an input catalog to the provided map
* Subtract mode subtracts the point sources given in the catalog from the input map, producing a source free map.
All of these rely on extra parameters like the beam (-b) and frequency (-f). Run dory with one of the modes
for more details (e.g. "dory find")."""
if len(sys.argv) < 2:
	sys.stderr.write(help_general + "\n")
	sys.exit(1)
mode = sys.argv[1]

parser = argparse.ArgumentParser(description=help_general)
parser.add_argument("mode", choices=["find","fit","subtract"])
if mode == "find":
	parser.add_argument("imap", help="The map to find sources in. Should be enmap-compatible.")
	parser.add_argument("idiv", help="The inverse variance per pixel of imap.")
	parser.add_argument("odir", help="The directory to write the output to. Will be crated if necessary.")
elif mode == "fit":
	parser.add_argument("imap", help="The map to fit sources in. Should be enmap-compatible.")
	parser.add_argument("idiv", help="The inverse variance per pixel of imap.")
	parser.add_argument("icat", help="The input point source catalog for amplitude fitting.")
	parser.add_argument("odir", help="The directory to write the output to. Will be crated if necessary.")
elif mode == "subtract":
	parser.add_argument("imap", help="The map to subtract sources from. Should be enmap-compatible.")
	parser.add_argument("icat", help="The catalog of sources to be subtracted.")
	parser.add_argument("omap", help="The resulting source-subtracted map")
	parser.add_argument("omodel", nargs="?", default=None, help="The source model that was subtracted (optional)")
else: pass
parser.add_argument("-m", "--mask",    type=str,   default=None, help="The mask to apply when finding sources, typically a galactic mask to avoid identifying dust as sources. Should be the same shape as the map. 0 is unmasked, 1 is masked.")
parser.add_argument("-b", "--beam",    type=str,   default="1.4",help="The beam of the map. Should be a 1d harmonic transfer function.")
parser.add_argument("-f", "--freq",    type=float, default=150,  help="The observing frequency in GHz. Only matters for flux calculations.")
parser.add_argument("-R", "--regions", type=str,   default=None, help="Which regions to consider to have comogeneous noise correlations. 'full': Use whole map, 'tile:npix': split map into npix*npix sized tiles. Or specify the file name to a ds9 region file containing boxes.")
parser.add_argument("-a", "--apod",    type=int,   default=30, help="The width of the apodization region, in pixels.")
parser.add_argument("--apod-margin",   type=int,   default=10, help="How far away from the apod region a source should be to be valid.")
parser.add_argument("-s", "--nsigma",  type=float, default=3.5, help="The number a sigma a source must be to be included in the catalog when finding sources.")
parser.add_argument("-p", "--pad",     type=int,   default=60, help="The number of pixels to extend each region by in each direciton, to avoid losing sources at region boundaries. Should be larger than apod+apod_margin")
parser.add_argument("-P", "--prior",   type=float, default=1.0, help="The strength of the input prior in fit mode. Actually the inverse of the source variability assumed, so 0 means the source will be assumed to be infinitely variable, and hence the input database amplitudes don't add anything to the new fit. infinity means that the source is completley stable, and the input statistics add in inverse variance to the measurements. The default is 1, which means that the input database contributes only at the 1 sigma level")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-o", "--output",  type=str,   default="full,reg", help="What types of output to write to the output directory. Comma-separated list. 'full': Output the final, merged quantities. 'reg': Output per-region results. 'debug': Output lots of debug maps. 'maps': Output maps and not just catalogs at each level.")
parser.add_argument(      "--ncomp",   type=int,   default=1, help="The number of stokes components to fit for in fit mode.")
parser.add_argument("--hack",          type=float, default=0)
args = parser.parse_args()
import numpy as np, os
from enlib import dory
from enlib import enmap, utils, bunch, mpi, fft, pointsrcs

comm       = mpi.COMM_WORLD
shape, wcs = enmap.read_map_geometry(args.imap)
beam       = dory.get_beam(args.beam)
regions    = dory.get_regions(args.regions, shape, wcs)

def divdiag(div):
	if   div.ndim == 2: return div.preflat
	elif div.ndim == 3: return div
	elif div.ndim == 4: return enmap.samewcs(np.einsum("aayx->ayx",div),div)
	else: raise ValueError("Invalid div shape: %s" % div.shape)

def get_div(div, ci):
	if len(div) > ci: return div[ci]
	else: return div[0]*2

def get_beam_profile(beam, nsamp=10001, rmax=0, tol=1e-7):
	# First do a low-res run to find rmax
	if not rmax:
		r0   = np.linspace(0, np.pi, nsamp)
		br0  = utils.beam_transform_to_profile(beam, r0, normalize=True)
		imax = min(len(r0)-1,np.where(br0 > tol)[0][-1]+1)
		rmax = r0[imax]
	# Then get the actual profile
	r    = np.linspace(0, rmax, nsamp)
	br   = utils.beam_transform_to_profile(beam, r, normalize=True)
	B    = np.array([r,br])
	return B

def work_around_stupid_mpi4py_bug(imap):
	imap.dtype = np.dtype('=' + imap.dtype.char)
	return imap

if args.mode == "find":
	# Point source finding mode
	results  = []
	map_keys = ["map","snmap","model","resid","resid_snmap"]
	utils.mkdir(args.odir)
	# Mpi-parallelization over regions is simple, but a bit lazy. It means that there will be
	# no speedup for single-region maps
	for ri in range(comm.rank, len(regions), comm.size):
		reg_fid = regions[ri]
		reg_pad = dory.pad_region(reg_fid, args.pad)
		print "%3d region %3d/%d %5d %5d %6d %6d" % (comm.rank, ri+1, len(regions), reg_fid[0,0], reg_fid[1,0], reg_fid[0,1], reg_fid[1,1])
		try:
			# We only use T to find sources in find mode for now. P usually has much lower S/N and
			# doesn't help much. Should add it later, though.
			imap   = enmap.read_map(args.imap, pixbox=reg_pad).preflat[0]
			idiv   = enmap.read_map(args.idiv, pixbox=reg_pad).preflat[0]
			if args.mask:
				mshape, mwcs = enmap.read_map_geometry(args.mask)
				mbox  = enmap.pixbox_of(mwcs, imap.shape, imap.wcs)
				idiv *= (1-enmap.read_map(args.mask, pixbox=mbox).preflat[0])
			if "debug" in args.output: dump_prefix = args.odir + "/region_%02d_" % ri
			else:                      dump_prefix = None
			result = dory.find_srcs(imap, idiv, beam, freq=args.freq, apod=args.apod, apod_margin=args.apod_margin,
					snmin=args.nsigma, verbose=args.verbose, dump=dump_prefix)
			# FIXME: artifacts are act-specific
			result = dory.prune_artifacts(result)
		except Exception as e:
			print "Exception for task %d region %d: %s" % (comm.rank, ri, e.message)
			raise
		# Write region output
		if "reg" in args.output:
			prefix = args.odir + "/region_%02d_" % ri
			dory.write_catalog_fits(prefix + "cat.fits" , result.cat)
			dory.write_catalog_txt(prefix + "cat.txt" , result.cat)
			if "maps" in args.output:
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
			dory.write_catalog_fits(args.odir + "/cat.fits", tot_cat)
			dory.write_catalog_txt(args.odir + "/cat.txt", tot_cat)
		# Then build the full maps
		if "maps" in args.output:
			for name in map_keys:
				if comm.rank == 0: print "Writing %s" % name
				merged = dory.merge_maps_onto([result[name] for result in results], shape, wcs, comm,
						root=0, crop=args.apod+args.apod_margin)
				if comm.rank == 0: enmap.write_map(args.odir + "/%s.fits" % name, merged)
				del merged

elif args.mode == "fit":
	icat    = dory.read_catalog(args.icat)
	reg_cats = []
	utils.mkdir(args.odir)
	for ri in range(comm.rank, len(regions), comm.size):
		reg_fid = regions[ri]
		reg_pad = dory.pad_region(reg_fid, args.pad, fft=True)
		print "%3d region %3d/%d %5d %5d %6d %6d" % (comm.rank, ri+1, len(regions), reg_fid[0,0], reg_fid[1,0], reg_fid[0,1], reg_fid[1,1])
		try:
			# We support polarization here, but treat each component independently
			imap   = enmap.read_map(args.imap, pixbox=reg_pad).preflat[:args.ncomp]
			idiv   = divdiag(enmap.read_map(args.idiv, pixbox=reg_pad))[:args.ncomp]
			if args.mask:
				mshape, mwcs = enmap.read_map_geometry(args.mask)
				mbox  = enmap.pixbox_of(mwcs, imap.shape, imap.wcs)
				idiv *= (1-enmap.read_map(args.mask, pixbox=mbox).preflat[0])
			if "debug" in args.output: dump_prefix = args.odir + "/region_%02d_" % ri
			else:                      dump_prefix = None
			imap[~np.isfinite(imap)] = 0
			idiv[~np.isfinite(idiv)] = 0
			idiv[idiv<0] = 0
			# Get our flux conversion factor
			beam2d = dory.calc_2d_beam(beam, imap.shape, imap.wcs)
			barea  = dory.calc_beam_transform_area(beam2d)
			fluxconv = utils.flux_factor(barea, args.freq*1e9)/1e6
			reg_cat = None
			for ci in range(len(imap)):
				# Build an amplitude prior from our input catalog fluxes
				prior    = dory.build_prior(icat.flux[:,ci]/fluxconv, icat.dflux[:,ci]/fluxconv, 1/args.prior)
				src_pos  = np.array([icat.dec,icat.ra]).T
				print imap.shape, idiv.shape, ci
				fit_inds, amp, icov = dory.fit_src_amps(imap[ci], get_div(idiv,ci), src_pos, beam, prior=prior, apod=args.apod,
						apod_margin=args.apod_margin, verbose=args.verbose, dump=dump_prefix, hack=args.hack, region=ri)
				if reg_cat is None:
					reg_cat = icat[fit_inds].copy()
					reg_cat.amp = reg_cat.damp = reg_cat.flux = reg_cat.dflux = 0
				reg_cat.amp[:,ci]  = amp
				reg_cat.damp[:,ci] = np.diag(icov)**-0.5
			reg_cat.flux  = reg_cat.amp*fluxconv
			reg_cat.dflux = reg_cat.damp*fluxconv
			# Sort by S/N catalog order
			reg_cat = reg_cat[np.argsort(reg_cat.amp[:,0]/reg_cat.damp[:,0])[::-1]]
			# Write region output
			if "reg" in args.output:
				prefix = args.odir + "/region_%02d_" % ri
				dory.write_catalog_fits(prefix + "cat.fits", reg_cat)
				dory.write_catalog_txt (prefix + "cat.txt",  reg_cat)
			if "full" in args.output:
				reg_cats.append(reg_cat)
		except Exception as e:
			print "Exception for task %d region %d: %s" % (comm.rank, ri, e.message)
			raise
	if "full" in args.output:
		if len(reg_cats) > 0: my_cat = np.concatenate(reg_cats)
		else: my_cat = np.zeros([0], dory.cat_dtype)
		tot_cat  = dory.allgather_catalog(my_cat, comm)
		tot_cat  = dory.merge_duplicates(tot_cat)
		# Sort by S/N catalog order
		tot_cat = tot_cat[np.argsort(tot_cat.amp[:,0]/tot_cat.damp[:,0])[::-1]]
		if comm.rank == 0:
			print "Writing catalogue"
			dory.write_catalog_fits(args.odir + "/cat.fits", tot_cat)
			dory.write_catalog_txt (args.odir + "/cat.txt",  tot_cat)
elif args.mode == "subtract":
	icat      = dory.read_catalog(args.icat)
	beam_prof = get_beam_profile(beam)
	barea     = dory.calc_beam_profile_area(beam_prof)
	fluxconv  = utils.flux_factor(barea, args.freq*1e9)/1e6
	# Reformat the catalog to the format sim_srcs takes
	srcs      = np.concatenate([[icat.dec, icat.ra], icat.flux.T/fluxconv],0).T
	# Evaluate the model in regions which we can mpi parallelize over
	models    = []
	omaps     = []
	for ri in range(comm.rank, len(regions), comm.size):
		reg_fid = regions[ri]
		reg_pad = dory.pad_region(reg_fid, args.pad)
		print "%3d region %3d/%d %5d %5d %6d %6d" % (comm.rank, ri+1, len(regions), reg_fid[0,0], reg_fid[1,0], reg_fid[0,1], reg_fid[1,1])
		map    = enmap.read_map(args.imap, pixbox=reg_pad)
		map    = work_around_stupid_mpi4py_bug(map)
		model  = pointsrcs.sim_srcs(map.shape, map.wcs, srcs, beam_prof, dtype=map.dtype, pixwin=True,verbose=args.verbose)
		omaps.append(map-model)
		if args.omodel: models.append(model)
		del model, map
	if comm.rank == 0: print "Merging map"
	omap  = dory.merge_maps_onto(omaps, shape, wcs, comm, root=0, crop=args.pad)
	del omaps
	if comm.rank == 0: print "Writing map"
	if comm.rank == 0: enmap.write_map(args.omap, omap)
	del omap
	if args.omodel:
		if comm.rank == 0: print "Merging model"
		model = dory.merge_maps_onto(models, shape, wcs, comm, root=0, crop=args.pad)
		del models
		if comm.rank == 0: print "Writing model"
		if comm.rank == 0: enmap.write_map(args.omodel, model)
		del model
