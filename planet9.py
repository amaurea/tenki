# Main program for the planet 9 search. Will have three modes of operation:
# * map, which will output point source free rhs, div, ktrue, beam for each
#    time chunk
# * filter, which will produce S/N maps (or chisquare maps) for each of these,
# * find, which will do the likelihood search given the chisquare maps
# Since the first step has multiple outputs, it's easiest to have it output
# a directory with standardized file names in it. Those directories can then
# be inputs and outputs for filter step. Find can take a list of directories
# or chisquare files. Filter could in theory be combined with map, but it
# wants a different parallelization (dmaps with tiles aren't good for SHTs.
# could use tile ffts, but let's keep things simple)

import argparse, sys
help_general = """Usage:
  planet9 map     sel area odir [prefix]
  planet9 filter  dirs
  planet9 find    dirs odir

  planet9 searches for low S/N, slowly moving objects in the outer solar system by
  making maps in chunks of time during which the objects won't have much
  time to move ("map" operation), matched filtering these to produce a chisquare
  per pixel ("filter" operation), and searching through the orbital parameter space
  to find candidates ("find" operation)."""

if len(sys.argv) < 2:
	sys.stderr.write(help_general + "\n")
	sys.exit(1)
mode = sys.argv[1]

# Handle each mode. These are practically separate programs, but I keep them in one
# command to reduce clutter.
if mode == "map":
	import numpy as np
	from enlib import utils, config, mpi, scanutils, sampcut, pmat, mapmaking, nmat
	from enact import filedb, actdata, actscan
	config.default("map_bits",    32, "Bit-depth to use for maps and TOD")
	config.default("downsample",   1, "Factor with which to downsample the TOD")
	config.default("map_sys",  "cel", "Coordinate system for the maps")
	parser = config.ArgumentParser()
	parser.add_argument("sel")
	parser.add_argument("area")
	parser.add_argument("odir")
	parser.add_argument("prefix", nargs="?",  default=None)
	parser.add_argument("--dt",   type=float, default=3)
	parser.add_argument("--srcs", type=str,   default=None)
	parser.add_argument("-S", "-corr-spacing", type=float, default=2)
	args = parser.parse_args()

	comm  = mpi.COMM_WORLD
	utils.mkdir(args.odir)
	shape, wcs = enmap.read_map_geometry(args.area)
	wshape = (3,)+shape[-2:]
	dtype = np.float32 if config.get("map_bits") == 32 else np.float64
	root = args.odir + "/" + (args.prefix + "_" if args.prefix else "")
	sys  = config.get("map_sys")

	# Should we use distributed maps?
	npix = shape[-2]*shape[-1]
	use_dmap = npix > 5e7

	utils.mkdir(root + "log")
	logfile   = root + "log/log%03d.txt" % comm.rank
	log_level = log.verbosity2level(config.get("verbosity"))
	L = log.init(level=log_level, file=logfile, rank=comm.rank)

	def merge_nearby(srcs, rlim=2*utils.arcmin):
		"""given source parameters which might contain duplicates, detect these duplicates
		and merge them to produce a single catalog with no duplicates. sources are considered
		duplicates if they are within rlim of each other."""
		pos    = np.array([srcs[:,1]*np.cos(srcs[:,0]), srcs[:,1]])
		tree   = spatial.cKDTree(pos)
		groups = tree.query_ball_tree(tree, rlim)
		done   = np.zeros(len(cat),bool)
		ocat   = []
		for gi, group in enumerate(groups):
			# remove everything that's done
			group = np.array(group)
			group = group[~done[group]]
			if len(group) == 0: continue
			# Let amplitude be the max in the group, and the position be the
			# amp-weighted mean
			gsrcs  = srcs[group]
			osrc   = gsrcs[0]*0
			weight = gsrcs[:,2]
			best   = np.argmax(np.abs(weight))
			osrc[:2] = np.sum(gsrcs[:,:2]*weight,0)/np.sum(weight)
			osrc[2:] = gsrcs[best]
			done[group] = True
			ocat.append(osrc)
		ocat = np.array(ocat)
		return ocat

	def cut_bright_srcs(scan, srcs, alim_include=1e4, alim_size=100):
		"""Cut sources in srcs that are brighter than alim in uK"""
		srcs_cut = srcs[srcs[:,2]>alim_include]
		if len(srcs) == 0: return scan
		tod  = np.zeros((scan.ndet,scan.nsamp), np.float32)
		psrc = pmat.PmatPtsrc(scan, srcs_cut)
		psrc.forward(tod, srcs)
		cut  = sampcut.from_mask(tod > alim_size)
		scan.cut *= cut
		scan.cut_noisest *= cut
		return scan

	class NmatWindowed(nmat.NoiseMatrix):
		def __init__(self, nmat_inner, windows):
			self.inner  = nmat_inner
			self.windows = windows
		def apply(self, tod, white=False):
			for window in self.windows: window(tod)
			if not white: self.inner.apply(tod)
			else:         self.inner.white(tod)
			for window in self.windows: window(tod)
			return tod
		def white(self, tod): return self.apply(tod, white=True)

	class PsrcSimple:
		def __init__(self, scan, srcparam):
			self.inner = pmat.PmatPtsrc(scan, srcparam)
			self.srcparam = srcparam
		def forward(tod, amps, tmul=None, pmul=None):
			params = self.srcparam.copy()
			params[:,2:5] = 0; params[:,2] = amps
			self.inner.forward(tod, params, tmul=tmul, pmul=pmul)
			return tod
		def backward(tod, tmul=None):
			params = self.srcparam.copy()
			self.inner.backward(tod, params, pmul=0, tmul=tmul)
			return params[:,2]

	def choose_corr_points(shape, wcs, spacing):
		# Set up the points where we will measure the correlation matrix
		box       = enmap.box(shape, wcs)
		dstep     = spacing
		corr_pos = []
		for dec in np.arange(box[0,0]+dstep/2.0, box[1,0], dstep):
			astep = spacing/np.cos(dec)
			for ra in np.arange(box[0,1]+astep/2.0, box[1,1], astep):
				corr_pos.append([dec,ra])
		corr_pos = np.array(samps)
		return corr_pos
		
	filedb.init()
	db     = filedb.scans.select(args.sel)
	ids    = db.ids
	mjd    = utils.ctime2mjd(db.data["t"])
	chunks = utils.find_equal_groups(mjd//args.dt)
	corr_pos = choose_corr_points(shape, wcs, args.corr_spacing*utils.degree)

	# How to parallelize? Could do it over chunks. Usually there will be more chunks than
	# mpi tasks. But there will still be many tods per chunk too (about 6 tods per hour
	# and 72 hours per chunk gives 432 tods per chunk). That's quite a bit for one mpi
	# task to do. Could paralellize over both... No, keep things simple. Parallelize over tods
	# in a chunk, and make sure that nothing breaks if some tasks don't have anything to do.
	L.info("Processing %d chunks" % len(chunks))
	for ci, chunk in enumerate(chunks):
		#### 1. Distribute and read in all our scans
		L.info("Scanning chunk %3d/%d with ids %s .. %s" % (ci+1, len(chunks), ids[chunk[0]], ids[chunk[-1]]))
		myinds = np.array(chunk)[comm.rank::comm.size]
		myinds, myscans = scanutils.read_scans(ids, myinds, actscan.ACTScan, db.data, downsample=config.get("downsample"))
		myinds = np.array(myinds, int)
		# Find the cost and bbox of each successful tod
		costs  = np.zeros(len(chunk), int)
		boxes  = np.zeros([len(chunk),2,2],np.float)
		for ind, scan in zip(myinds, myscans):
			sizes[ind] = scan.det*scan.nsamp
			boxes[ind] = scanutils.calc_sky_bbox_scan(scan, sys)
		costs  = utils.allreduce(costs, comm)
		boxes  = utils.allreduce(boxes, comm)
		# Disqualify empty scans
		bad    = costs == 0
		L.info("Rejected %d bad tods" % (np.sum(bad)))
		inds   = np.array(chunk)[~bad]
		costs, boxes = costs[~bad], boxes[~bad]
		# Redistribute
		if not use_dmap:
			myinds = scanutils.distribute_scans2(inds, costs, None, comm)
		else:
			myinds, mysubs, mybbox = scanutils.distribute_scans2(inds, costs, boxes, comm)
		L.info("Rereading shuffled scans")
		del myscans # scans do take up some space, even without the tod being read in
		myinds, myscans = scanutils.read_scans(ids, myinds, actscan.ACTScan, db.data, downsample=config.get("downsample"))

		#### 2. Prepare our point source database and the corresponding cuts
		src_override = pointsrcs.read(args.src) if args.src else None
		for scan in myscans:
			scan.srcparam = merge_nearby(pointsrcs.src2param(src_override if src_override is not None else src.pointsrcs))
			cut_bright_srcs(scan, scan.srcparam)

		#### 3. Process our tods ####
		apply_window = mapmaking.FilterWindow(config.get("tod_window"))
		if not use_dmap: area = enmap.zeros(wshape, wcs, dtype)
		else:            area = Moo
		# only work will be 3,ny,nx. The rest are scalar. Will copy in-out as necessary
		work  = area*0
		rhs   = area[0]*0
		div   = area[0]*0
		kvals = np.zeros(len(corr_pos), dtype)
		umap  = area[0]*0; umap[0] = 1
		for scan in myscans:
			# Read the tod
			tod = scan.get_samples().astype(dtype)
			tod = utils.deslope(tod)

			# Build and apply the noise model
			apply_window(scan, tod)
			scan.noise = scan.noise.update(tod, scan.srate)
			scan.noise.apply(tod)
			apply_window(scan, tod)
			# Build the window into the noise model, so we don't have to drag it around
			scan.noise = NmatWindowed(scan.noise, [apply_window])
			def apply_cut(tod, inplace=True): return sampcut.gapfill_const(scan.cut, tod, inplace=inplace)

			# Measure the point source amplitudes. We have ensured that they
			# are reasonably independent, so this is the diagonal part of
			# amps = (P'N"P)"P'N"d, where N"d is what our tod is now.
			psrc = PsrcSimple(scan, scan.srcparam)
			wtod    = tod.copy()
			src_rhs = psrc.backward(apply_tod(wtod))
			src_div = psrc.backward(apply_tod(scan.noise.apply(psrc.forward(wtod, src_rhs*0+1, tmul=0))))
			src_amp = src_rhs/src_div
			# And subtract the point sources
			tod -= scan.noise.apply(psrc.forward(wtod, src_amp, tmul=0))
			del wtod, src_rhs, src_div

			if not use_dmap: pmap = pmat.PmatMap(scan, area)
			else:            pmap = ...

			# Build our rhs map
			apply_cut(tod)
			pmap.backward(tod, work, pmul=0); rhs += work[0]
			# Build our div map
			work[:] = 0; work[0] = umap
			pmap.forward(tod, work, tmul=0)
			scan.noise.white(tod)
			apply_cut(tod)
			pmap.backward(tod, work, pmul=0); div += work[0]
			# Build our sparsely sampled BPNPB
			samp_srcs = np.zeros([len(corr_pos),3])
			samp_srcs[:,:2] = corr_pos[:,::-1]
			psamp = PsrcSimple(scan, pointsrcs.src2param(samp_srcs))
			psamp.forward(tod, src_amps*0+1, tmul=0) # PB
			apply_cut(tod)
			scan.noise.apply(tod)                    # NPB
			apply_cut(tod)
			kvals += psrc.backward(tod)              # BPNPB

		#### 4. mpi reduce
		if not use_dmap:
			rhs = utils.allreduce(rhs, comm)
			div = utils.allreduce(div, comm)
			kvals = utils.allreduce(kvals, comm)
		else:
			moo

		#### 4. Output our results
		ctime0 = int(utils.mjd2ctime(mjd[chunk[0]]//args.dt * args.dt))
		cdir   = root + str(ctime0)
		utils.mkdir(cdir)
		if not use_dmap:
			if comm.rank == 0:
				enmap.write_map(cdir + "/rhs.fits", rhs)
				enmap.write_map(cdir + "/div.fits", div)
		else:
			moo
		if comm.rank == 0:
			np.savetxt(cdir + "/kvals.txt", np.array([corr_pos[:,1], corr_pos[:,0], kvals]), fmt="%10.5f %10.5f %15.7e")
	L.info("Done")
