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
mjd0 = 57174

# Handle each mode. These are practically separate programs, but I keep them in one
# command to reduce clutter.
if mode == "map":
	# Map mode. Process the actual time-ordered data, producing rhs.fits, div.fits and info.fits
	# for each time-chunk.
	import numpy as np, os
	from enlib import utils
	with utils.nowarn(): import h5py
	from enlib import planet9, enmap, dmap, config, mpi, scanutils, sampcut, pmat, mapmaking
	from enlib import log, pointsrcs, gapfill, ephemeris
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
	parser.add_argument(      "--srclim",       type=float, default=500)
	parser.add_argument("-S", "--corr-spacing", type=float, default=2)
	parser.add_argument(      "--srcsub",       type=int,   default=1)
	parser.add_argument("-M", "--mapsub",       type=str,   default=None)
	parser.add_argument("-I", "--inject",       type=str,   default=None)
	parser.add_argument(      "--only",         type=str)
	parser.add_argument(      "--static",       action="store_true")
	parser.add_argument("-c", "--cont",         action="store_true")
	args = parser.parse_args()

	comm  = mpi.COMM_WORLD
	utils.mkdir(args.odir)
	shape, wcs = enmap.read_map_geometry(args.area)
	wshape = (3,)+shape[-2:]
	dtype = np.float32 if config.get("map_bits") == 32 else np.float64
	root = args.odir + "/" + (args.prefix + "_" if args.prefix else "")
	sys  = config.get("map_sys")
	ym   = utils.arcmin/utils.yr2days
	# Bias source amplitudes 0.1% towards their fiducial value
	amp_prior = 1e-3

	only = [int(word) for word in args.only.split(",")] if args.only else []

	# Should we use distributed maps?
	npix = shape[-2]*shape[-1]
	use_dmap = npix > 5e7

	utils.mkdir(root + "log")
	logfile   = root + "log/log%03d.txt" % comm.rank
	log_level = log.verbosity2level(config.get("verbosity"))
	L = log.init(level=log_level, file=logfile, rank=comm.rank)

	filedb.init()
	db     = filedb.scans.select(args.sel)
	ids    = db.ids
	mjd    = utils.ctime2mjd(db.data["t"])
	chunks = utils.find_equal_groups(mjd//args.dt)
	chunks = [np.sort(chunk) for chunk in chunks]
	chunks = [chunks[i] for i in np.argsort([c[0] for c in chunks])]
	corr_pos = planet9.choose_corr_points(shape, wcs, args.corr_spacing*utils.degree)

	if args.inject:
		inject_params = np.loadtxt(args.inject,ndmin=2) # [:,{ra0,dec0,R,vx,vy,flux}]

	# How to parallelize? Could do it over chunks. Usually there will be more chunks than
	# mpi tasks. But there will still be many tods per chunk too (about 6 tods per hour
	# and 72 hours per chunk gives 432 tods per chunk). That's quite a bit for one mpi
	# task to do. Could paralellize over both... No, keep things simple. Parallelize over tods
	# in a chunk, and make sure that nothing breaks if some tasks don't have anything to do.
	L.info("Processing %d chunks" % len(chunks))
	for ci, chunk in enumerate(chunks):
		ctime0 = int(utils.mjd2ctime(mjd[chunk[0]]//args.dt * args.dt))
		if only and ctime0 not in only: continue
		cdir   = root + str(ctime0)
		if args.cont and os.path.exists(cdir + "/info.hdf"): continue
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
				scan.srcparam = pointsrcs.src2param(src_override if src_override is not None else scan.pointsrcs)
				scan.srcparam = planet9.merge_nearby(scan.srcparam)
				planet9.cut_bright_srcs(scan, scan.srcparam, alim_include=args.srclim)

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

		# Get the frequency and beam for this chunk. We assume that
		# this is the same for every member of the chunk, so we only need
		# to do this for one scan
		scan       = actscan.ACTScan(filedb.data[chunk_ids[inds[0]]])
		_, dets    = actdata.split_detname(scan.dets)
		beam       = scan.beam
		freq       = scan.array_info.info.nom_freq[dets[0]]
		barea      = planet9.calc_beam_area(scan.beam)
		# Get the conversion from ref-freq flux to observed amplitude. This includes
		# dilution by the beam area
		flux2amp   = 1/utils.flux_factor(barea, args.fref*1e9, utils.T_cmb)
		fref2freq  = utils.planck(freq*1e9, args.Tref)/utils.planck(args.fref*1e9, args.Tref)
		rfact      = flux2amp * fref2freq * 1e3 # 1e3 for flux in mJy and amp in uK

		# only work will be 3,ny,nx. The rest are scalar. Will copy in-out as necessary
		work  = signal.work()
		rhs   = area[0]
		div   = rhs.copy()
		wrhs  = signal.prepare(rhs)
		wdiv  = signal.prepare(div)
		kvals = np.zeros(len(corr_pos), dtype)
		freqs, bareas = np.zeros(ntod), np.zeros(ntod)

		bleh = True and args.inject
		if bleh:
			sim_rhs = np.zeros(len(inject_params))
			sim_div = np.zeros(len(inject_params))

		for si, scan in zip(myinds, myscans):
			L.debug("Processing %s" % scan.id)
			# Read the tod
			tod = scan.get_samples().astype(dtype)
			tod = utils.deslope(tod)

			if args.mapsub:
				# Subtract the reference map. If the reference map is not source free,
				# then this could reintroduce strong point sources that were cut earlier.
				# To avoid this we do another round of gapfilling
				signal.forward(scan, tod, refmap, tmul=1, mmul=-1)
				gapfill.gapfill_joneig(tod, scan.cut, inplace=True)

			# Inject simulated signal if requested
			if args.inject:
				dmjd = scan.mjd0-mjd0
				earth_pos = -ephemeris.ephem_vec("Sun", scan.mjd0)[:,0]
				# Set the position and amplitude uK of each simulated source
				sim_srcs = np.zeros([len(inject_params),8])
				# TODO: inject and analyze with no displacement to see if interpolation is the
				# cause of the low bias in amplitude.
				sim_srcs[:,:2] = inject_params[:,1::-1]*utils.degree
				if not args.static:
					sim_srcs[:,:2] = planet9.displace_pos(sim_srcs[:,:2].T, earth_pos, inject_params.T[2], inject_params.T[4:2:-1]*ym*dmjd).T
				#print "params", inject_params[:,5]
				#print "rfact", rfact
				sim_srcs[:,2]  = inject_params[:,5]*rfact
				sim_srcs[:,5:7] = 1
				psim = planet9.PsrcSimple(scan, sim_srcs)
				psim.forward(tod, sim_srcs[:,2], tmul=1.0)

			# Build and apply the noise model
			apply_window(scan, tod)
			scan.noise = scan.noise.update(tod, scan.srate)
			scan.noise.apply(tod)
			apply_window(scan, tod)
			# Build the window into the noise model, so we don't have to drag it around
			scan.noise = planet9.NmatWindowed(scan.noise, [lambda tod: apply_window(scan, tod)])
			def apply_cut(tod, inplace=True): return sampcut.gapfill_const(scan.cut, tod, inplace=inplace)

			if args.srcsub:
				# Measure the point source amplitudes. We have ensured that they
				# are reasonably independent, so this is the diagonal part of
				# amps = (P'N"P)"P'N"d, where N"d is what our tod is now.
				psrc = planet9.PsrcSimple(scan, scan.srcparam)
				wtod    = tod.copy()
				src_rhs = psrc.backward(apply_cut(wtod))
				# For some reason this manages to be slightly negative sometimes. Might be because the sources are
				# not sufficiently separated. We avoid this by forcing it to be minimum 0
				src_div = psrc.backward(apply_cut(scan.noise.apply(apply_cut(psrc.forward(wtod, src_rhs*0+1, tmul=0)))))
				src_div = np.maximum(src_div, 0)
				# Bias slightly towards input value. This helps avoid problems with sources
				# that are hit by only a handful of samples, and are therefore super-uncertain.
				src_amp_old = scan.srcparam[:,2]
				src_div_old = planet9.defmean(src_div[src_div>0], 1e-5)*amp_prior
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
			if bleh: sim_rhs += psim.backward(tod)
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
			psamp = planet9.PsrcSimple(scan, samp_srcs)
			psamp.forward(tod, np.full(len(corr_pos),1.0), tmul=0) # PB
			apply_cut(tod)
			scan.noise.apply(tod)         # NPB
			apply_cut(tod)
			kvals += psamp.backward(tod)  # BPNPB
			# kvals is now the samp-sum of the sample icov over the squared beam
			# if iN were 1, then this would basically be the number of samples
			# that hit each source. if iN were ivar, then this would be the
			# inverse variance for the amplitude of each source
			if bleh:
				psim.forward(tod, sim_div*0+1, tmul=0)
				apply_cut(tod)
				scan.noise.apply(tod)
				apply_cut(tod)
				sim_div += psim.backward(tod)

		#### 4. mpi reduce
		signal.finish(rhs, wrhs)
		signal.finish(div, wdiv)
		kvals = utils.allreduce(kvals, comm)
		del wrhs, wdiv, work, signal, myscans
		if bleh:
			sim_rhs = utils.allreduce(sim_rhs, comm)
			sim_div = utils.allreduce(sim_div, comm)
			sim_rhs *= rfact; sim_div *= rfact**2
			sim_amp = sim_rhs/sim_div
			sim_damp= sim_div**-0.5
			if comm.rank == 0:
				for i in range(len(sim_amp)):
					print "%15.7e %15.7e %8.3f %8.3f %8.3f" % (sim_rhs[i], sim_div[i], sim_amp[i], sim_damp[i], sim_amp[i]/sim_damp[i])

		mean_mjd = np.mean(chunk_mjd[inds])

		#### 5. Output our results
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
				if bleh:
					hfile["sim_rhs"] = sim_rhs
					hfile["sim_div"] = sim_div

elif mode == "inject":
	# Signal injection mode. This used rhs, div and info to inject a fake source
	# with a given flux, distance and velocity into the rhs maps. The output will
	# be in a different directory, to avoid overwriting.
	#
	# Injecting in map space like this is an approximation compared to injecting
	# directly in the TOD (with the --inject parameter in "map"), but it's pretty
	# accurate, maybe a few percent off.
	parser = argparse.ArgumentParser()
	parser.add_argument("inject", help="dummy")
	parser.add_argument("paramfile")
	parser.add_argument("dirs", nargs="+")
	parser.add_argument("odir")
	parser.add_argument("-c", "--cont", action="store_true")
	parser.add_argument("-l", "--lmax", type=int, default=20000)
	args = parser.parse_args()
	import numpy as np, glob, sys, os
	from enlib import mpi, planet9, utils, enmap, ephemeris, pointsrcs

	dirs   = sum([glob.glob(dname) for dname in args.dirs],[])

	comm   = mpi.COMM_WORLD
	params = np.loadtxt(args.paramfile,ndmin=2) # [:,{ra0,dec0,R,vx,vy,flux}]
	ym     = utils.arcmin/utils.yr2days

	for ind in range(comm.rank, len(dirs), comm.size):
		idirpath = dirs[ind]
		odirpath = args.odir + "/" + os.path.basename(idirpath)
		if args.cont and os.path.isfile(odirpath + "/rhs.fits"): continue
		print "Processing %s" % (idirpath)
		info    = planet9.hget(idirpath + "/info.hdf")
		rhs     = enmap.read_map(idirpath + "/rhs.fits")
		div     = enmap.read_map(idirpath + "/div.fits")
		dmjd    = info.mjd - mjd0
		earth_pos = -ephemeris.ephem_vec("Sun", info.mjd)[:,0]
		# Set the position and amplitude uK of each simulated source
		srcs    = np.zeros([len(params),3])
		srcs[:,:2] = planet9.displace_pos(params.T[1::-1]*utils.degree, earth_pos, params.T[2], params.T[4:2:-1]*ym*dmjd).T
		srcs[:,2]  = params[:,5]*info.rfact
		sim  = pointsrcs.sim_srcs(rhs.shape, rhs.wcs, srcs, info.beam)
		## (Could also sim ths using planet9 rmat)
		#R       = planet9.Rmat(rhs.shape, rhs.wcs, info.beam, info.rfact, lmax=args.lmax)
		#dmap    = rhs*0
		#for i, src in enumerate(srcs):
		#	planet9.add_delta(dmap, dmap.sky2pix(src[:2]), params[i,5])
		#sim2 = R.apply(dmap)
		rhs += div*sim
		# And output
		utils.mkdir(odirpath)
		planet9.hput(odirpath + "/info.hdf", info)
		enmap.write_map(odirpath + "/rhs.fits", rhs)
		enmap.write_map(odirpath + "/div.fits", div)
		del rhs, div, info, sim

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
	parser.add_argument("-c", "--cont", action="store_true")
	args = parser.parse_args()
	from enlib import utils
	with utils.nowarn(): import h5py
	import numpy as np, glob, sys, os, healpy
	from scipy import ndimage
	from enlib import enmap, mpi, planet9

	comm  = mpi.COMM_WORLD
	scale = 1
	dirs  = sum([glob.glob(dname) for dname in args.dirs],[])

	# Storing this takes quite a bit of memory, but it's better than
	# rereading it all the time
	if args.mask: mask = enmap.read_map(args.mask)

	for ind in range(comm.rank, len(dirs), comm.size):
		dirpath = dirs[ind]
		if args.cont and os.path.isfile(dirpath + "/kmap.fits"): continue
		print "Processing %s" % (dirpath)
		info = planet9.hget(dirpath + "/info.hdf")
		# Compute frhs. We multiply by barea becuase both the
		# beam smoothing and rfact divide by the beam area, and we only want to
		# do it once.
		rhs     = enmap.read_map(dirpath + "/rhs.fits")
		R       = planet9.Rmat(rhs.shape, rhs.wcs, info.beam, info.rfact, lmax=args.lmax)
		R2      = planet9.Rmat(rhs.shape, rhs.wcs, info.beam, info.rfact, lmax=args.lmax, pow=2)
		if args.mask:
			wmask = 1-mask.extract(rhs.shape, rhs.wcs)
			rhs  *= wmask
		frhs = R.apply(rhs)
		unhit  = rhs==0
		del rhs
		# Compute our approximate K
		div = enmap.read_map(dirpath + "/div.fits")
		if args.mask: div *= wmask
		RRdiv = R2.apply(div); del div
		RRdiv = np.maximum(RRdiv, max(0,np.max(RRdiv)*1e-10))
		# get our correction. The kvals stuff is not really necessary now that
		# we apply an empirical normalization anyway. The empirical normalization is
		# typically around 1.2-1.5, so the kvals have pretty much the right unit
		approx_vals = RRdiv.at(info.kvals.T[1::-1], order=1)
		exact_vals  = info.kvals.T[2] * scale**2
		correction  = np.sum(exact_vals*approx_vals)/np.sum(approx_vals**2)
		pos = np.array([-2,10])*utils.degree #
		#val = 3.73
		#print "A", RRdiv.at(pos), val, val/RRdiv.at(pos)
		kmap        = RRdiv * correction; del RRdiv
		#print "B", kmap.at(pos), val, val/kmap.at(pos)
		#norm        = planet9.get_smooth_normalization(frhs, kmap)
		#kval = kmap.at(pos, order=0); nval = norm.at(pos, order=0)
		#print "A %8.4f %8.4f" % (kval, nval)
		#kmap       *= norm; del norm
		#print "C", kmap.at(pos), val, val/kmap.at(pos)
		# Kill all unhit values. This is only necessary because kmap is approximate, and
		# thus doesn't agree perfectly with frhs about how the smoothing smears out the
		# signal in the holes. By setting them explicitly to zero there we avoid dividing
		# by very small numbers there.
		kmap[:]     = np.maximum(kmap, max(np.max(kmap)*1e-4, 1e-12))
		kmap[unhit] = 0
		frhs[unhit] = 0
		# Kmap should contain the noise ivar in mJy at the reference frequency.
		# We can compare this to an approximation based on div alone, which is
		# what my forecast was based on. Given white noise with some ivar per
		# pixel


		#enmap.write_map(dirpath + "/norm.fits", norm)
		enmap.write_map(dirpath + "/frhs.fits", frhs)
		enmap.write_map(dirpath + "/kmap.fits", kmap)
		with utils.nowarn(): sigma = frhs/kmap**0.5
		enmap.write_map(dirpath + "/sigma.fits", sigma)
		del kmap, sigma, unhit

elif mode == "find":
	# planet finding mode. Takes as input the output directories from the filter step.
	parser = argparse.ArgumentParser()
	parser.add_argument("find", help="dummy")
	parser.add_argument("idirs", nargs="+")
	parser.add_argument("area")
	parser.add_argument("odir")
	parser.add_argument("-O", "--order",     type=int, default=1)
	parser.add_argument("-d", "--downgrade", type=int, default=1)
	parser.add_argument("-m", "--mode",      type=str, default="simple")
	parser.add_argument("-T", "--tsize",     type=int, default=1200)
	parser.add_argument("-P", "--pad",       type=int, default=60)
	parser.add_argument("-N", "--npertile",  type=int, default=-1)
	parser.add_argument("-R", "--rsearch",   type=str, default="500:1200:50")
	parser.add_argument("-V", "--vsearch",   type=str, default="0:2:0.1")
	parser.add_argument("-v", "--verbose",   action="store_true")
	parser.add_argument(      "--static",    action="store_true")
	parser.add_argument("-s", "--snmin",     type=float, default=5)
	args = parser.parse_args()
	import numpy as np, time, os, glob
	from enlib import utils
	with utils.nowarn(): import h5py
	from scipy import ndimage, stats
	from enlib import enmap, utils, mpi, parallax, cython, ephemeris, statdist, planet9

	utils.mkdir(args.odir)
	comm  = mpi.COMM_WORLD
	dtype = np.float32
	shape, wcs = enmap.read_map_geometry(args.area)
	tsize, pad = args.tsize, args.pad

	# Our parameter search space. Distances in AU, speeds in arcmin per year
	ym = utils.arcmin/utils.yr2days
	rmin, rmax, dr = [float(w)    for w in args.rsearch.split(":")]
	vmin, vmax, dv = [float(w)*ym for w in args.vsearch.split(":")]
	nr = int(np.ceil((rmax-rmin)/dr))+1
	nv = 2*int(np.round(vmax/dv))+1

	# How many tiles will we have?
	if tsize == 0: nty, ntx = 1, 1
	else: nty, ntx = (np.array(shape[-2:])+tsize-1)//tsize
	ntile = nty*ntx

	idirs = sum([glob.glob(idir) for idir in args.idirs],[])

	# We can parallelize both over tiles and inside tiles. More mpi tasks
	# per tile does not increase the total memory cost (much), but has
	# diminishing returns due to communication overhead (and you can't go
	# beyond the number of maps the tile has). More mpi tasks across tiles
	# scales well computationally, but memory use goes up proportionally.
	# Let's make it configurable.
	if args.npertile < 0: npertile = len(idirs)//(-args.npertile)
	else:                 npertile = args.npertile
	npertile = max(1,min(comm.size,npertile))
	# Build intra- and inter-tile communicators
	comm_intra = comm.Split(comm.rank//npertile, comm.rank %npertile)
	comm_inter = comm.Split(comm.rank %npertile, comm.rank//npertile)

	if tsize > 0:
		# We're using tiles, so our outputs will be directories
		for name in ["sigma_plain","param_map","limit_map","snmin_map","sigma_map","hit_tot","cands"]:
			utils.mkdir(args.odir + "/" + name)

	# Loop over tiles
	for ti in range(comm_inter.rank, ntile, comm_inter.size):
		ty, tx = ti//ntx, ti%ntx
		if tsize > 0:
			def oname(name):
				fname, fext = os.path.splitext(name)
				return "%s/%s/tile%03d_%03d%s" % (args.odir, fname, ty, tx, fext)
			pixbox = np.array([[ty*tsize-pad,tx*tsize-pad],[(ty+1)*tsize+pad,(tx+1)*tsize+pad]])
		else:
			def oname(name): return args.odir + "/" + name
			pixbox = np.array([[-pad,-pad],[shape[-2]+pad,shape[-1]+pad]])
		# Get the shape of the sliced, downgraded tiles
		tshape, twcs = enmap.slice_geometry(shape, wcs, [slice(pixbox[0,0],pixbox[1,0]),slice(pixbox[0,1],pixbox[1,1])], nowrap=True)
		tshape, twcs = enmap.downgrade_geometry(tshape, twcs, args.downgrade)
		# Read in our tile data
		frhss, kmaps, mjds = [], [], []
		for ind in range(comm_intra.rank, len(idirs), comm_intra.size):
			idir = idirs[ind]
			lshape, lwcs = enmap.read_map_geometry(idir + "/frhs.fits")
			pixbox_loc   = pixbox - enmap.pixbox_of(wcs, lshape, lwcs)[0]
			kmap = enmap.read_map(idir + "/kmap.fits", pixbox=pixbox_loc).astype(dtype)
			if args.downgrade > 1: kmap = enmap.downgrade(kmap, args.downgrade)
			# Skip tile if it's empty
			if np.any(~np.isfinite(kmap)) or np.all(kmap < 1e-10):
				print "skipping %s (nan or unexposed)" % idir
				continue
			if args.verbose: print idir
			frhs = enmap.read_map(idir + "/frhs.fits", pixbox=pixbox_loc).astype(dtype)
			if args.downgrade > 1: frhs = enmap.downgrade(frhs, args.downgrade)
			with h5py.File(idir + "/info.hdf", "r") as hfile: mjd = hfile["mjd"].value
			frhss.append(frhs)
			kmaps.append(kmap)
			mjds.append(mjd)
		nlocal = len(frhss)
		nmap_tile = comm_intra.allreduce(nlocal)
		# To apply the parallax displacement we need to know the Earth's position
		# relative to the sun in cartesian equatorial coordinates. This is simply
		# the negative of the sun's position relative to the earth.
		earth_pos = -ephemeris.ephem_vec("Sun", mjds).T

		# It's useful to have the unshifted output map, to look for poorly
		# subtracted point sources etc
		frhs_tot = enmap.zeros(tshape, twcs, dtype)
		kmap_tot = enmap.zeros(tshape, twcs, dtype)
		hit_tot  = enmap.zeros(tshape, twcs, np.int32)
		for mi in range(nlocal):
			frhs_tot += frhss[mi]
			kmap_tot += kmaps[mi]
		if comm_intra.size > 1:
			frhs_tot = utils.allreduce(frhs_tot, comm_intra)
			kmap_tot = utils.allreduce(kmap_tot, comm_intra)
		klim = np.percentile(kmap_tot,90)*1e-3
		if comm_intra.rank == 0:
			#print "A", np.std(frhs_tot), np.std(kmap_tot)
			#sigma = planet9.solve(frhs_tot, kmap_tot)
			sigma = frhs_tot*0
			cython.solve(frhs_tot, kmap_tot, sigma, klim=klim)
			#print "B", np.std(sigma)
			enmap.write_map(oname("sigma_plain.fits"), planet9.unpad(sigma, pad))
		#print klim

		# Perform the actual parameter search
		if comm_intra.rank == 0:
			sigma_max = sigma-np.inf
			param_max = enmap.zeros((4,)+sigma.shape, sigma.wcs, dtype)
		for ri in range(nr):
			r = rmin + ri*dr
			for vy in np.linspace(-vmax,vmax,nv):
				for vx in np.linspace(-vmax,vmax,nv):
					vmag = (vy**2+vx**2)**0.5
					if vmag < vmin or vmag > vmax: continue
					# Accumulate shifted maps into these
					frhs_tot[:] = 0
					kmap_tot[:] = 0
					t1 = time.time()
					for mi in range(nlocal):
						# Find the offset as a function of time
						dmjd = mjds[mi] - mjd0
						off = [vy*dmjd, vx*dmjd]
						if not args.static:
							cython.displace_map(frhss[mi], earth_pos[mi], r, off, omap=frhs_tot)
							cython.displace_map(kmaps[mi], earth_pos[mi], r, off, omap=kmap_tot)
						else:
							frhs_tot += frhss[mi]
							kmap_tot += kmaps[mi]
					if comm_intra.size > 1:
						frhs_tot = utils.allreduce(frhs_tot, comm_intra)
						kmap_tot = utils.allreduce(kmap_tot, comm_intra)
					t2 = time.time()
					if comm_intra.rank == 0:
						cython.solve(frhs_tot, kmap_tot, sigma, klim=klim)
						cython.update_total(sigma, sigma_max, param_max, hit_tot, kmap_tot, r, vy, vx)
						t3 = time.time()
						print "%2d %5.0f %5.2f %5.2f  %8.3f ms %8.3f ms" % (comm_inter.rank, r, vy/ym, vx/ym, (t2-t1)*1e3, (t3-t2)*1e3)
		if comm_intra.rank == 0:
			# Find the mean value of sigma_max, excluding unexposed pixels. We need this to infer the
			# the effective number of independent samples that were maxed together in sigma_max
			mask  = kmap_tot > klim
			dist_params = planet9.build_dist_map(sigma_max, mask=mask) # paralellize this
			# Find candidates above our threshold. Is returned as a list of [ra,dec,nsigma,r,vy,vx].
			sigma_eff = (sigma_max-dist_params[0])/dist_params[1]
			cands = planet9.find_candidates(sigma_eff, param_max, snmin=snmin, pad=pad//args.downgrade)
			np.savetxt(oname("cands.txt"), np.array([cands[:,1]/utils.degree, cands[:,0]/utils.degree,
				cands[:,2], cands[:,3], cands[:,4], cands[:,5], cands[:,6]/ym, cands[:,7]/ym]).T, fmt="%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f")
			param_max[1:] /= ym
			enmap.write_map(oname("sigma_map.fits"), planet9.unpad(sigma_max, pad))
			enmap.write_map(oname("sigma_eff.fits"), planet9.unpad(sigma_eff, pad))
			enmap.write_map(oname("param_map.fits"), planet9.unpad(param_max, pad))
			# The kmap_tot = da used here only corresponds to the last iteration, but
			# since kmaps are pretty smooth, it shouldn't matter much.
			mask = hit_tot > 0
			da = kmap_tot*0; da[mask] = kmap_tot[mask]**-0.5
			enmap.write_map(oname("limit_map.fits"), planet9.unpad(da * snmin, pad))
			enmap.write_map(oname("snmin_map.fits"), planet9.unpad(snmin, pad))
			enmap.write_map(oname("hit_tot.fits"), planet9.unpad(hit_tot, pad))
