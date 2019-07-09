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

# TODO: Switch from B to R = BFS, where F is the flux conversion and S is the
# spectral index correction.

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
	# Map mode. Process the actual time-ordered data, producing rhs.fits, div.fits and info.fits
	# for each time-chunk.
	import numpy as np, h5py
	from scipy import spatial, integrate
	from enlib import utils, enmap, dmap, config, mpi, scanutils, sampcut, pmat, mapmaking, nmat, log, pointsrcs, gapfill
	from enact import filedb, actdata, actscan
	config.default("map_bits",    32, "Bit-depth to use for maps and TOD")
	config.default("downsample",   1, "Factor with which to downsample the TOD")
	config.default("map_sys",  "cel", "Coordinate system for the maps")
	config.default("verbosity",    1, "Verbosity")
	parser = config.ArgumentParser()
	parser.add_argument("map", help="dummy")
	parser.add_argument("sel")
	parser.add_argument("area")
	parser.add_argument("odir")
	parser.add_argument("prefix", nargs="?",  default=None)
	parser.add_argument(      "--dt",           type=float, default=3)
	parser.add_argument("-T", "--Tref",         type=float, default=40)
	parser.add_argument(      "--fref",         type=float, default=150)
	parser.add_argument(      "--srcs",         type=str,   default=None)
	parser.add_argument(      "--srclim",       type=float, default=1e4)
	parser.add_argument("-S", "--corr-spacing", type=float, default=2)
	parser.add_argument(      "--srcsub",       type=int,   default=1)
	parser.add_argument("-M", "--mapsub",       type=str,   default=None)
	args = parser.parse_args()

	comm  = mpi.COMM_WORLD
	utils.mkdir(args.odir)
	shape, wcs = enmap.read_map_geometry(args.area)
	wshape = (3,)+shape[-2:]
	dtype = np.float32 if config.get("map_bits") == 32 else np.float64
	root = args.odir + "/" + (args.prefix + "_" if args.prefix else "")
	sys  = config.get("map_sys")
	# Bias source amplitudes 0.1% towards their fiducial value
	amp_prior = 1e-3

	# Should we use distributed maps?
	npix = shape[-2]*shape[-1]
	use_dmap = npix > 5e7

	utils.mkdir(root + "log")
	logfile   = root + "log/log%03d.txt" % comm.rank
	log_level = log.verbosity2level(config.get("verbosity"))
	L = log.init(level=log_level, file=logfile, rank=comm.rank)

	def calc_beam_area(beam_profile):
		r, b = beam_profile
		return integrate.simps(2*np.pi*r*b,r)

	def merge_nearby(srcs, rlim=2*utils.arcmin):
		"""given source parameters which might contain duplicates, detect these duplicates
		and merge them to produce a single catalog with no duplicates. sources are considered
		duplicates if they are within rlim of each other."""
		pos    = np.array([srcs[:,1]*np.cos(srcs[:,0]), srcs[:,0]]).T
		tree   = spatial.cKDTree(pos)
		groups = tree.query_ball_tree(tree, rlim)
		done   = np.zeros(len(srcs),bool)
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
			osrc[:2] = np.sum(gsrcs[:,:2]*weight[:,None],0)/np.sum(weight)
			osrc[2:] = gsrcs[best,2:]
			done[group] = True
			ocat.append(osrc)
		ocat = np.array(ocat)
		return ocat

	def cut_bright_srcs(scan, srcs, alim_include=1e4, alim_size=10):
		"""Cut sources in srcs that are brighter than alim in uK"""
		srcs_cut = srcs[srcs[:,2]>alim_include]
		if len(srcs) == 0: return scan
		tod  = np.zeros((scan.ndet,scan.nsamp), np.float32)
		psrc = pmat.PmatPtsrc(scan, srcs_cut)
		psrc.forward(tod, srcs)
		cut  = sampcut.from_mask(tod > alim_size)
		scan.cut *= cut
		scan.cut_noiseest *= cut
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
		def forward(self, tod, amps, tmul=None, pmul=None):
			params = self.srcparam.copy()
			params[:,2:5] = 0; params[:,2] = amps
			self.inner.forward(tod, params, tmul=tmul, pmul=pmul)
			return tod
		def backward(self, tod, tmul=None):
			params = self.srcparam.copy()
			self.inner.backward(tod, params, pmul=0, tmul=tmul)
			return params[:,2]

	def choose_corr_points(shape, wcs, spacing):
		# Set up the points where we will measure the correlation matrix
		box       = np.sort(enmap.box(shape, wcs),0)
		dstep     = spacing
		corr_pos = []
		for dec in np.arange(box[0,0]+dstep/2.0, box[1,0], dstep):
			astep = spacing/np.cos(dec)
			for ra in np.arange(box[0,1]+astep/2.0, box[1,1], astep):
				corr_pos.append([dec,ra])
		corr_pos = np.array(corr_pos)
		return corr_pos

	def defmean(arr, defval=0):
		return np.mean(arr) if len(arr) > 0 else defval
		
	filedb.init()
	db     = filedb.scans.select(args.sel)
	ids    = db.ids
	mjd    = utils.ctime2mjd(db.data["t"])
	chunks = utils.find_equal_groups(mjd//args.dt)
	chunks = [np.sort(chunk) for chunk in chunks]
	chunks = [chunks[i] for i in np.argsort([c[0] for c in chunks])]
	corr_pos = choose_corr_points(shape, wcs, args.corr_spacing*utils.degree)

	# How to parallelize? Could do it over chunks. Usually there will be more chunks than
	# mpi tasks. But there will still be many tods per chunk too (about 6 tods per hour
	# and 72 hours per chunk gives 432 tods per chunk). That's quite a bit for one mpi
	# task to do. Could paralellize over both... No, keep things simple. Parallelize over tods
	# in a chunk, and make sure that nothing breaks if some tasks don't have anything to do.
	L.info("Processing %d chunks" % len(chunks))
	for ci, chunk in enumerate(chunks):
		#### 1. Distribute and read in all our scans
		chunk_ids = ids[chunk]
		chunk_mjd = mjd[chunk]
		L.info("Scanning chunk %3d/%d with %4d tods from %s" % (ci+1, len(chunks), len(chunk), ids[chunk[0]]))
		myinds = np.arange(len(chunk))[comm.rank::comm.size]
		myinds, myscans = scanutils.read_scans(chunk_ids, myinds, actscan.ACTScan, filedb.data, downsample=config.get("downsample"))
		myinds = np.array(myinds, int)
		# Find the cost and bbox of each successful tod
		costs  = np.zeros(len(chunk), int)
		boxes  = np.zeros([len(chunk),2,2],np.float)
		for ind, scan in zip(myinds, myscans):
			costs[ind] = scan.ndet*scan.nsamp
			boxes[ind] = scanutils.calc_sky_bbox_scan(scan, sys)
		costs  = utils.allreduce(costs, comm)
		boxes  = utils.allreduce(boxes, comm)
		# Disqualify empty scans
		bad    = costs == 0
		L.info("Rejected %d bad tods" % (np.sum(bad)))
		inds   = np.where(~bad)[0]
		costs, boxes = costs[~bad], boxes[~bad]
		ntod   = len(inds)

		if ntod == 0:
			L.info("Chunk %d has no tods. Skipping" % (ci+1))
			continue

		# Redistribute
		if not use_dmap:
			myinds = scanutils.distribute_scans2(inds, costs, comm)
		else:
			myinds, mysubs, mybbox = scanutils.distribute_scans2(inds, costs, comm, boxes)
		L.info("Rereading shuffled scans")
		del myscans # scans do take up some space, even without the tod being read in
		myinds, myscans = scanutils.read_scans(chunk_ids, myinds, actscan.ACTScan, filedb.data, downsample=config.get("downsample"))
		if args.srcsub:
			#### 2. Prepare our point source database and the corresponding cuts
			src_override = pointsrcs.read(args.srcs) if args.srcs else None
			for scan in myscans:
				scan.srcparam = merge_nearby(pointsrcs.src2param(src_override if src_override is not None else scan.pointsrcs))
				cut_bright_srcs(scan, scan.srcparam, alim_include=args.srclim)

		#### 3. Process our tods ####
		apply_window = mapmaking.FilterWindow(config.get("tod_window"))
		if not use_dmap: area = enmap.zeros(wshape, wcs, dtype)
		else:
			geo  = dmap.DGeometry(wshape, wcs, dtype=dtype, bbox=mybbox, comm=comm)
			area = dmap.zeros(geo)

		# Set up our signal. We do this instead of building the pmat manually
		# to make it easy to support both maps and dmaps
		if not use_dmap: signal = mapmaking.SignalMap(myscans, area, comm)
		else:            signal = mapmaking.SignalDmap(myscans, mysubs, area, comm)

		# Get the input sky map that we will subtract. We do this because the CMB+CIB
		# are the same from tod to tod, but we can't support intra-tod correlated noise,
		# so we have to get rid of it. This is not optimal, but it shouldn't be far off.
		if args.mapsub:
			sshape, swcs = enmap.read_map_geometry(args.mapsub)
			pixbox = enmap.pixbox_of(swcs, shape, wcs)
			if not use_dmap: refmap = enmap.read_map(args.mapsub, pixbox=pixbox).astype(dtype)
			else:            refmap = dmap.read_map (args.mapsub, pixbox=pixbox).astype(dtype)
			refmap = signal.prepare(refmap)

		# only work will be 3,ny,nx. The rest are scalar. Will copy in-out as necessary
		work  = signal.work()
		rhs   = area[0]
		div   = rhs.copy()
		wrhs  = signal.prepare(rhs)
		wdiv  = signal.prepare(div)
		kvals = np.zeros(len(corr_pos), dtype)
		freqs, bareas = np.zeros(ntod), np.zeros(ntod)
		for si, scan in zip(myinds, myscans):
			L.debug("Processing %s" % scan.id)
			# Read the tod
			tod = scan.get_samples().astype(dtype)
			tod = utils.deslope(tod)

			if args.mapsub:
				# Subtract the reference map.
				signal.forward(scan, tod, refmap, tmul=1, mmul=-1)

			# Build and apply the noise model
			apply_window(scan, tod)
			scan.noise = scan.noise.update(tod, scan.srate)
			scan.noise.apply(tod)
			apply_window(scan, tod)
			# Build the window into the noise model, so we don't have to drag it around
			scan.noise = NmatWindowed(scan.noise, [lambda tod: apply_window(scan, tod)])
			def apply_cut(tod, inplace=True): return sampcut.gapfill_const(scan.cut, tod, inplace=inplace)

			if args.srcsub:
				# Measure the point source amplitudes. We have ensured that they
				# are reasonably independent, so this is the diagonal part of
				# amps = (P'N"P)"P'N"d, where N"d is what our tod is now.
				psrc = PsrcSimple(scan, scan.srcparam)
				wtod    = tod.copy()
				src_rhs = psrc.backward(apply_cut(wtod))
				src_div = psrc.backward(apply_cut(scan.noise.apply(apply_cut(psrc.forward(wtod, src_rhs*0+1, tmul=0)))))
				# Bias slightly towards input value. This helps avoid problems with sources
				# that are hit by only a handful of samples, and are therefore super-uncertain.
				src_amp_old = scan.srcparam[:,2]
				src_div_old = defmean(src_div[src_div>0], 0)*amp_prior
				src_rhs += src_amp_old*src_div_old
				src_div += src_div_old
				with utils.nowarn():
					src_amp = src_rhs/src_div
					src_amp[~np.isfinite(src_amp)] = 0
				# And subtract the point sources
				psrc.forward(wtod, src_amp, tmul=0)
				# Keep our gapfilling semi-consistent
				gapfill.gapfill_linear(wtod, scan.cut, inplace=True)
				scan.noise.apply(wtod)
				tod -= wtod
				del wtod, src_rhs, src_div

			# Build our rhs map
			apply_cut(tod)
			signal.backward(scan, tod, work, mmul=0); wrhs += work[0]
			# Build our div map
			work[:] = 0; work[0] = 1
			signal.forward(scan, tod, work, tmul=0)
			scan.noise.white(tod)
			apply_cut(tod)
			signal.backward(scan, tod, work, mmul=0); wdiv += work[0]
			# Build our sparsely sampled BPNPB
			samp_srcs = np.zeros([len(corr_pos),8])
			samp_srcs[:,:2]  = corr_pos
			samp_srcs[:,2]   = 1
			samp_srcs[:,5:7] = 1
			psamp = PsrcSimple(scan, samp_srcs)
			psamp.forward(tod, np.full(len(corr_pos),1.0), tmul=0) # PB
			apply_cut(tod)
			scan.noise.apply(tod)         # NPB
			apply_cut(tod)
			kvals += psamp.backward(tod)  # BPNPB
			# kvals is now the samp-sum of the sample icov over the squared beam
			# if iN were 1, then this would basically be the number of samples
			# that hit each source. if iN were ivar, then this would be the
			# inverse variance for the amplitude of each source

		#### 4. mpi reduce
		signal.finish(rhs, wrhs)
		signal.finish(div, wdiv)
		kvals = utils.allreduce(kvals, comm)
		del wrhs, wdiv, work, signal, myscans

		# Get the frequency and beam for this chunk. We assume that
		# this is the same for every member of the chunk, so we only need
		# to do this for one scan
		if comm.rank == 0:
			scan       = actscan.ACTScan(filedb.data[chunk_ids[inds[0]]])
			_, dets    = actdata.split_detname(scan.dets)
			beam       = scan.beam
			freq       = scan.array_info.info.nom_freq[dets[0]]
			barea      = calc_beam_area(scan.beam)
			# Get the conversion from ref-freq flux to observed amplitude. This includes
			# dilution by the beam area
			flux2amp   = 1/utils.flux_factor(barea, args.fref*1e9, utils.T_cmb)
			fref2freq  = utils.planck(freq*1e9, args.Tref)/utils.planck(args.fref*1e9, args.Tref)
			rfact      = flux2amp * fref2freq * 1e3 # 1e3 for flux in mJy and amp in uK

		mean_mjd = np.mean(chunk_mjd[inds])

		#### 5. Output our results
		ctime0 = int(utils.mjd2ctime(mjd[chunk[0]]//args.dt * args.dt))
		cdir   = root + str(ctime0)
		utils.mkdir(cdir)

		if not use_dmap:
			if comm.rank == 0:
				enmap.write_map(cdir + "/rhs.fits", rhs)
				enmap.write_map(cdir + "/div.fits", div)
		else:
				dmap.write_map(cdir + "/rhs.fits", rhs, merged=True)
				dmap.write_map(cdir + "/div.fits", div, merged=True)

		if comm.rank == 0:
			with h5py.File(cdir + "/info.hdf", "w") as hfile:
				hfile["kvals"] = np.array([corr_pos[:,1], corr_pos[:,0], kvals*rfact**2]).T
				hfile["beam"]  = beam
				hfile["barea"] = barea
				hfile["freq"]  = freq
				hfile["fref"]  = args.fref
				hfile["Tref"]  = args.Tref
				hfile["rfact"] = rfact
				hfile["ids"]   = chunk_ids[inds]
				hfile["mjd"]   = mean_mjd
				hfile["mjds"]  = chunk_mjd[inds]

elif mode == "filter":
	# Filter mode. This applies the harmonic part of the matched filter, as well as
	# a spline prefilter for fast lookup. For each output directory from the map
	# operation, read in rhs, div and info, and compute and output
	#  frhs = (prefilter) R rhs
	#  kmap = (prefilter) R**2 div * interpol(kvals/sample(R**2 div))
	# With these in hand, the significance is simply frhs/sqrt(kmap)
	# R = rfact * B. To apply B we can either use FFTs or SHTs, depending on the
	# patch size. Should probably apply dust mask too. Or should that be in map?
	# Easiest to do it here, if perhaps a bit imprecise.
	parser = argparse.ArgumentParser()
	parser.add_argument("filter", help="dummy")
	parser.add_argument("dirs", nargs="+")
	parser.add_argument("-l", "--lmax", type=int, default=20000)
	parser.add_argument("-m", "--mask", type=str, default=None)
	args = parser.parse_args()
	import numpy as np, h5py, glob, sys, os, healpy
	from scipy import integrate
	from enlib import mpi
	from pixell import enmap, utils, sharp, curvedsky, bunch

	def apply_beam_fft(map, bl):
		l    = map.modlmap()
		bval = np.interp(l, np.arange(len(bl)), bl, right=0)
		return enmap.ifft(enmap.fft(map)*bval).real

	def apply_beam_sht(map, bl, tol=1e-5):
		lmax = np.where(bl/np.max(bl) > tol)[0][-1]
		ainfo= sharp.alm_info(lmax)
		alm  = curvedsky.map2alm_cyl(map, ainfo=ainfo)
		ainfo.lmul(alm, bl[:lmax+1], out=alm)
		return curvedsky.alm2map_cyl(alm, map, copy=True)

	def get_distortion(map):
		box = map.box()
		dec1, dec2 = box[:,0]
		rmin = min(np.cos(dec1),np.cos(dec2))
		rmax = 1 if not dec1*dec2 > 0 else max(np.cos(dec1),np.cos(dec2))
		return rmax/rmin-1

	def apply_beam(map, bl, max_distortion=0.1):
		if get_distortion(map) > max_distortion:
			print "sht"
			return apply_beam_sht(map, bl)
		else:
			print "fft"
			return apply_beam_fft(map, bl)

	def hget(fname):
		res = bunch.Bunch()
		with h5py.File(fname, "r") as hfile:
			for key in hfile:
				res[key] = hfile[key].value
		return res

	def get_pixsizemap_cyl(shape, wcs):
		# ra step is constant for cylindrical projections
		dra  = np.abs(wcs.wcs.cdelt[0])*utils.degree
		# get the dec for all the pixel edges
		decs = enmap.pix2sky(shape, wcs, [np.arange(shape[-2]+1)-0.5,np.zeros(shape[-2]+1)])[0]
		sins = np.sin(decs)
		sizes= np.abs((sins[1:]-sins[:-1]))*dra
		# make it broadcast with full maps
		sizes = sizes[:,None]
		return sizes

	class Rmat:
		# This represents the linear operator that takes us from
		# a delta function with a given flux (mJy) in the center of each pixel
		# to the observed source in uK in the map
		def __init__(self, shape, wcs, beam_profile, rfact, lmax=20e3, pow=1):
			self.pixarea = get_pixsizemap_cyl(shape, wcs)
			self.bl      = healpy.sphtfunc.beam2bl(beam_profile[1]**pow, beam_profile[0], lmax)
			# We don't really need to factorize out the beam area like this
			self.barea   = self.bl[0]
			self.bl     /= self.barea
			self.rfact   = rfact
			self.pow     = pow
		def apply(self, map):
			# we get a map of fluxes (mJy), and convert it to a map of amplitudes (uK)
			# we then smear those out to produce the source profiles. Smoothing a unit
			# pixel with a beam with area barea results in an amplitude of pixarea/barea,
			# so correct for this factor beforehand.
			rmap = apply_beam((self.rfact**self.pow * self.barea/self.pixarea) * map, self.bl)
			return rmap

	def get_smooth_normalization(frhs, kmap, res=120, tol=2, bsize=1200):
		"""Split into cells of size res*res. For each cell,
		find the normalization factor needed to make the
		mean chisquare 1, where chisquare = frhs**2 / kmap.
		Gapfill outliers, and then interpolate back to full
		resolution. The result will be what kmap needs to be
		multiplied by to get the right chisquare.
		
		This could be made more robust by using a median of mean
		technique in each cell. That would avoid penalizing cells with
		local high signal. But as long as we are noise dominated the
		current approach should be good enough."""
		ny, nx = np.array(frhs.shape[-2:])//res
		mask   = kmap > np.max(kmap)*1e-4
		rblock = frhs[:ny*res,:nx*res].reshape(ny,res,nx,res)
		kblock = kmap[:ny*res,:nx*res].reshape(ny,res,nx,res)
		mblock = mask[:ny*res,:nx*res].reshape(ny,res,nx,res)
		# compute the mean chisquare per block
		with utils.nowarn():
			chisqs = rblock**2 / kblock
			chisqs[~mblock] = 0
			ngood         = np.sum(mblock,(1,3))
			mean_chisqs   = np.sum(chisqs,(1,3))/ngood
		# replace bad entries with typical value
		nmin = res**2 / 3
		good = ngood > nmin
		if np.all(~good): return enmap.full(frhs.shape, frhs.wcs, 1.0, frhs.dtype)
		ref   = np.median(mean_chisqs[good])
		with utils.nowarn():
			good &= np.abs(np.log(mean_chisqs/ref)) < tol
		if np.all(~good): mean_chisqs[:] = ref
		else: mean_chisqs[~good] = np.median(mean_chisqs[good])
		# Turn mean chisqs into kmap scaling factor to get the right chisquare
		norm_lowres = mean_chisqs
		norm_full   = enmap.zeros(frhs.shape, frhs.wcs, frhs.dtype)
		# This loop is just to save memory
		for y1 in range(0, frhs.shape[0], bsize):
			work = norm_full[y1:y1+bsize]
			opix = work.pixmap()
			opix[0] += y1
			ipix = opix/float(res)
			work[:] = norm_lowres.at(ipix, unit="pix", order=1, mode="nearest")
		return norm_full

	comm  = mpi.COMM_WORLD
	scale = 1

	# Storing this takes quite a bit of memory, but it's better than
	# rereading it all the time
	if args.mask: mask = enmap.read_map(args.mask)

	for ind in range(comm.rank, len(args.dirs), comm.size):
		dirpath = args.dirs[ind]
		print "Processing %s" % (dirpath)
		info = hget(dirpath + "/info.hdf")
		# Compute frhs. We multiply by barea becuase both the
		# beam smoothing and rfact divide by the beam area, and we only want to
		# do it once.
		rhs     = enmap.read_map(dirpath + "/rhs.fits")
		R       = Rmat(rhs.shape, rhs.wcs, info.beam, info.rfact, lmax=args.lmax)
		R2      = Rmat(rhs.shape, rhs.wcs, info.beam, info.rfact, lmax=args.lmax, pow=2)
		if args.mask:
			wmask = mask.extract(rhs.shape, rhs.wcs)
			rhs  *= wmask
		frhs = R.apply(rhs); del rhs
		enmap.write_map(dirpath + "/frhs.fits", frhs)
		# Compute our approximate K
		div = enmap.read_map(dirpath + "/div.fits")
		if args.mask: div *= wmask
		RRdiv = R2.apply(div); del div
		# get our correction. The kvals stuff is not really necessary now that
		# we apply an empirical normalization anyway.
		approx_vals = RRdiv.at(info.kvals.T[1::-1], order=1)
		exact_vals  = info.kvals.T[2] * scale**2
		correction  = np.sum(exact_vals*approx_vals)/np.sum(approx_vals**2)
		kmap        = RRdiv * correction; del RRdiv
		norm        = get_smooth_normalization(frhs, kmap)
		kmap       *= norm; del norm
		#enmap.write_map(dirpath + "/norm.fits", norm)
		enmap.write_map(dirpath + "/kmap.fits", kmap)
		enmap.write_map(dirpath + "/sigma.fits", frhs/kmap**0.5)
		del kmap

elif mode == "find":
	# planet finding mode. Takes as input the output directories from the filter step.
	parser = argparse.ArgumentParser()
	parser.add_argument("find", help="dummy")
	parser.add_argument("idirs", nargs="+")
	parser.add_argument("odir")
	parser.add_argument("-O", "--order",     type=int, default=1)
	parser.add_argument("-d", "--downgrade", type=int, default=1)
	parser.add_argument("-m", "--mode",      type=str, default="simple")
	args = parser.parse_args()
	import numpy as np, h5py, time
	from scipy import ndimage
	from enlib import enmap, utils, mpi, parallax, cython, ephemeris

	def solve(rhs, kmap):
		with utils.nowarn():
			sigma = frhs_tot / kmap_tot**0.5
		sigma[kmap_tot < np.max(kmap_tot)*1e-2] = 0
		return sigma

	def get_mjd(idir):
		with h5py.File(idir + "/info.hdf", "r") as hfile:
			return hfile["mjd"].value

	def group_by_year(mjds, dur=365.24, nhist=12):
		# First find the best splitting point via histogram
		mjds   = np.asarray(mjds)
		pix    = (mjds/dur*nhist).astype(int) % nhist
		hist   = np.bincount(pix, minlength=nhist)
		mjd_split = np.argmin(hist)*dur/nhist
		# Then do the actual splitting
		group_ind  = ((mjds-mjd_split)/dur).astype(int)
		return utils.find_equal_groups(group_ind)

	def group_neighbors(mjds, tol=1):
		return utils.find_equal_groups(mjds, tol)

	utils.mkdir(args.odir)
	comm  = mpi.COMM_WORLD
	dtype = np.float32
	# Distances in AU, speeds in arcmin per year
	rmin = 500.; rmax = 1200.; dr =   50.; nr = int(np.ceil((rmax-rmin)/dr))+1
	#vmin =   0.; vmax =    2.; dv = 0.050; nv = int(np.ceil(2*vmax/dv))
	vmin =   0.; vmax =    2.; dv = 0.10; nv = 2*int(np.round(vmax/dv))+1
	mjd0 = 56596

	# Should do the search in tiles. It lets me use fewer maps in regions with less
	# coverage, makes it possible to run the full thing on computers with less memory,
	# and gives better incremental progress. It could also be used to optimize tile size
	# for speed, maybe even gpu. Loop over tiles of constant size as usual, with padding
	# set to the maximum displacement (in our case 0.5 deg would be more than enough).
	# Choose the tile size so that the *padded* tiles have the computationally optimal
	# size.

	if args.mode == "simple":
		# single-level, trivially parallelizable search
		shape, wcs = enmap.downgrade(enmap.read_map(args.idirs[0] + "/frhs.fits"), args.downgrade).geometry
		nmap = len(args.idirs)
		frhss, kmaps, mjds = [], [], []
		for ind in range(comm.rank, nmap, comm.size):
			idir = args.idirs[ind]
			print "Reading %s" % idir
			frhs = enmap.read_map(idir + "/frhs.fits").astype(dtype)
			if args.downgrade > 1: frhs = enmap.downgrade(frhs, args.downgrade)
			kmap = enmap.read_map(idir + "/kmap.fits").astype(dtype)
			if args.downgrade > 1: kmap = enmap.downgrade(kmap, args.downgrade)
			with h5py.File(idir + "/info.hdf", "r") as hfile: mjd = hfile["mjd"].value
			frhss.append(frhs)
			kmaps.append(kmap)
			mjds.append(mjd)
		nlocal = len(frhss)
		# To apply the parallax displacement we need to know the Earth's position
		# relative to the sun in cartesian equatorial coordinates. This is simply
		# the negative of the sun's position relative to the earth.
		earth_pos = -ephemeris.ephem_vec("Sun", mjds).T

		# unshifted
		frhs_tot = enmap.zeros(shape, wcs, dtype)
		kmap_tot = enmap.zeros(shape, wcs, dtype)
		for mi in range(nlocal):
			frhs_tot += frhss[mi]
			kmap_tot += kmaps[mi]
		if comm.size > 1:
			frhs_tot = utils.allreduce(frhs_tot, comm)
			kmap_tot = utils.allreduce(kmap_tot, comm)
		if comm.rank == 0:
			sigma = solve(frhs_tot, kmap_tot)
			enmap.write_map(args.odir + "/sigma_plain.fits", sigma)

		sigma_max = None
		for ri in range(nr):
			r = rmin + ri*dr
			# ok, given this distance we can compute the parallaxed positions. They differ
			# for each map because they come from different times. The parallax computation
			# is a bit slow, and doing it here dilutes that cost, but it also means that we
			# now have to store nmap copies of the posmap. But the parallax displacement will
			# change slowly with position, so can't we compute low-res ones, and just do linear
			# interpolation as needed? Looks like the parallax computation is only about 4-5x slower
			# than interpolation.
			for vy in np.linspace(-vmax,vmax,nv):
				for vx in np.linspace(-vmax,vmax,nv):
			#for vy in [-vmax,0,vmax]:
			#	for vx in [-vmax,0,vmax]:
					vmag = (vy**2+vx**2)**0.5
					if vmag > vmax: continue
					vy_rad = vy*utils.arcmin/utils.yr2days
					vx_rad = vx*utils.arcmin/utils.yr2days
					frhs_tot[:] = 0
					kmap_tot[:] = 0
					t1 = time.time()
					for mi in range(nlocal):
						# Find the offset as a function of time
						dmjd = mjds[mi] - mjd0
						off = [vy_rad*dmjd, vx_rad*dmjd]
						cython.displace_map(frhss[mi], earth_pos[mi], r, off, omap=frhs_tot)
						cython.displace_map(kmaps[mi], earth_pos[mi], r, off, omap=kmap_tot)
					if comm.size > 1:
						frhs_tot = utils.allreduce(frhs_tot, comm)
						kmap_tot = utils.allreduce(kmap_tot, comm)
					t2 = time.time()
					if comm.rank == 0:
						print "%5.0f %5.2f %5.2f  %8.3f ms" % (r, vy, vx, (t2-t1)*1e3)
						sigma = solve(frhs_tot, kmap_tot)
						# Collapsing to sigma here means that we lose information about the parameters
						# for which the match was found, which would increase the amount of work we have
						# to do later. It's better to run the object detection directly on each sigma map.
						# It should be pretty fast anyway. What threshold should I use? We want to output
						# enough candidates to be able to model the background of false positives (which
						# will probably be a gaussian).
						if sigma_max is None: sigma_max = sigma
						else: sigma_max = np.maximum(sigma_max, sigma)
					#enmap.write_map(args.odir + "/sigma_%+05.0f_%+05.2f_%+05.2f.fits" % (r, vy, vx), sigma)

		if comm.rank == 0:
			enmap.write_map(args.odir + "/sigma_max.fits", sigma_max)
			#cands = sigma > args.nsigma

	else:
		# hierarchical approach. Potentially a factor 30 or so speed increase, but involves
		# multiple levels of interpolation, each of which comes with a bit of signal loss.
		# Let's try the simple approach first, and see if it's workable.
		raise NotImplementedError
