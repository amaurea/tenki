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
from __future__ import division, print_function
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
mjd0 = 57688

# Handle each mode. These are practically separate programs, but I keep them in one
# command to reduce clutter.
if mode == "map":
	# Map mode. Process the actual time-ordered data, producing rhs.fits, div.fits and info.fits
	# for each time-chunk.
	import numpy as np, os, time
	from enlib import utils
	with utils.nowarn(): import h5py
	from enlib import planet9, enmap, dmap, config, mpi, scanutils, sampcut, pmat, mapmaking
	from enlib import log, pointsrcs, gapfill, ephemeris
	from enact import filedb, actdata, actscan, cuts as actcuts
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
	parser.add_argument("-D", "--dayerr",       type=str,   default="-1:1,-2:4")
	parser.add_argument(      "--srclim-day",   type=float, default=150)
	# These should ideally be moved into the general tod autocuts
	parser.add_argument("-a", "--asteroid-file", type=str, default=None)
	parser.add_argument("--asteroid-list", type=str, default=None)
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

	dayerr = np.array([[float(w) for w in tok.split(":")] for tok in args.dayerr.split(",")]).T # [[x1,y1],[x2,y2]]

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
		inject_params = np.loadtxt(args.inject,ndmin=2) # [:,{ra0,dec0,R,vy,vx,flux}]

	asteroids = planet9.get_asteroids(args.asteroid_file, args.asteroid_list)

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
				scan.srcparam, nmerged = planet9.merge_nearby(scan.srcparam)
				planet9.cut_srcs_rad(scan, scan.srcparam[nmerged>1])
				ctime = utils.mjd2ctime(scan.mjd0) + scan.boresight[scan.nsamp//2,0]
				hour  = ctime/3600%24
				isday = hour >= 11 and hour < 23
				if isday:
					planet9.cut_bright_srcs_daytime(scan, scan.srcparam, alim_include=args.srclim_day, errbox=dayerr)
				else:
					planet9.cut_bright_srcs(scan, scan.srcparam, alim_include=args.srclim)
		if asteroids:
			for scan in myscans:
				planet9.cut_asteroids_scan(scan, asteroids)

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
			else:            refmap = dmap.read_map (args.mapsub, pixbox=pixbox, bbox=mybbox, comm=comm).astype(dtype)
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

		bleh = False and args.inject
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
				#gapfill.gapfill_joneig(tod, scan.cut, inplace=True)
				gapfill.gapfill_linear(tod, scan.cut, inplace=True)

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
					sim_srcs[:,:2] = planet9.displace_pos(sim_srcs[:,:2].T, earth_pos, inject_params.T[2], inject_params.T[3:5]*ym*dmjd).T
				#print("params", inject_params[:,5])
				#print("rfact", rfact)
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
				# Compute the raw source amplitudes. Since we're fitting the source amplitude we might
				# as well output them.
				with utils.nowarn():
					src_amp_raw = src_rhs/src_div
					src_amp_raw[~np.isfinite(src_amp_raw)] = 0
					src_std_raw = src_div**-0.5
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
					print("%15.7e %15.7e %8.3f %8.3f %8.3f" % (sim_rhs[i], sim_div[i], sim_amp[i], sim_damp[i], sim_amp[i]/sim_damp[i]))

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
		del rhs, div

		if comm.rank == 0:
			with h5py.File(cdir + "/info.hdf", "w") as hfile:
				hfile["kvals"] = np.array([corr_pos[:,1], corr_pos[:,0], kvals*rfact**2]).T
				hfile["beam"]  = beam
				hfile["barea"] = barea
				hfile["freq"]  = freq
				hfile["fref"]  = args.fref
				hfile["Tref"]  = args.Tref
				hfile["rfact"] = rfact
				hfile["ids"]   = utils.encode_array_if_necessary(chunk_ids[inds])
				hfile["mjd"]   = mean_mjd
				hfile["mjds"]  = chunk_mjd[inds]
				if bleh:
					hfile["sim_rhs"] = sim_rhs
					hfile["sim_div"] = sim_div

elif mode == "inject":
	# Signal injection mode. This uses rhs, div and info to inject a fake source
	# with a given flux, distance and velocity into the rhs maps. The output will
	# be in a different directory, to avoid overwriting.
	#
	# Injecting in map space like this is an approximation compared to injecting
	# directly in the TOD (with the --inject parameter in "map"), but it's pretty
	# accurate, maybe a few percent off.
	#
	# I have tested this both with and without extra filtering in "filter". It's
	# accurate to 2%.
	parser = argparse.ArgumentParser()
	parser.add_argument("inject", help="dummy")
	parser.add_argument("paramfile")
	parser.add_argument("dirs", nargs="+")
	parser.add_argument("odir")
	parser.add_argument("-c", "--cont", action="store_true")
	parser.add_argument("-l", "--lmax", type=int,   default=10000)
	parser.add_argument(      "--mmul", type=float, default=1)
	parser.add_argument(      "--lknee",type=str,   default=None)
	args = parser.parse_args()
	import numpy as np, glob, sys, os
	from enlib import mpi, planet9, utils, enmap, ephemeris, pointsrcs, bench

	dirs   = sorted(sum([glob.glob(dname) for dname in args.dirs],[]))

	comm   = mpi.COMM_WORLD
	params = np.loadtxt(args.paramfile,ndmin=2) # [:,{ra0,dec0,R,vy,vx,flux}]
	ym     = utils.arcmin/utils.yr2days

	def get_div_correction(div, info, lmax=5000):
		# don't need as high lmax because div is pretty smooth
		R2          = planet9.Rmat(div.shape, div.wcs, info.beam, info.rfact, lmax=lmax, pow=2)
		kmap        = R2.apply(div)
		kmap        = np.maximum(kmap, max(0,np.max(kmap)*1e-10))
		approx_vals = kmap.at(info.kvals.T[1::-1], order=1)
		exact_vals  = info.kvals.T[2]
		correction  = np.sum(exact_vals*approx_vals)/np.sum(approx_vals**2)
		return correction

	for ind in range(comm.rank, len(dirs), comm.size):
		idirpath = dirs[ind]
		odirpath = args.odir + "/" + os.path.basename(idirpath)
		if args.cont and os.path.isfile(odirpath + "/rhs.fits"): continue
		print("Processing %s" % (idirpath))
		info    = planet9.hget(idirpath + "/info.hdf")
		rhs     = enmap.read_map(idirpath + "/rhs.fits")
		div     = enmap.read_map(idirpath + "/div.fits")
		dmjd    = info.mjd - mjd0
		earth_pos = -ephemeris.ephem_vec("Sun", info.mjd)[:,0]
		# Set the position and amplitude uK of each simulated source
		srcs       = np.zeros([len(params),3])
		srcs[:,:2] = planet9.displace_pos(params.T[1::-1]*utils.degree, earth_pos, params.T[2], params.T[3:5]*ym*dmjd).T
		srcs[:,2]  = params[:,5]*info.rfact
		with bench.show("sim"):
			sim  = pointsrcs.sim_srcs_dist_transform(rhs.shape, rhs.wcs, srcs, info.beam, ignore_outside=True, verbose=True)
		with bench.show("filter"):
			# Apply approximate mapmaker filter to sim. We use the Beam class for its 
			beam     = planet9.Beam(rhs.shape, rhs.wcs, info.beam, lmax=args.lmax)
			fit      = planet9.setup_noise_fit(rhs, args.lknee, idirpath)
			Falready = planet9.butterworth(beam.l, fit.lknee2, fit.alpha2)**0.5
			# Our normalization assumes no Falready, so compensate for it
			Falready/= np.sum(Falready*beam.lbeam**2*beam.nmode)/np.sum(beam.lbeam**2*beam.nmode)
			sim      = beam.apply(sim, Falready) # applies Falready instead of beam due to 2nd arg
			correction = get_div_correction(div, info)
		if args.mmul != 1:
			rhs *= args.mmul
		rhs += div*sim*correction
		# And output
		utils.mkdir(odirpath)
		planet9.hput(odirpath + "/info.hdf", info)
		enmap.write_map(odirpath + "/rhs.fits", rhs)
		enmap.write_map(odirpath + "/div.fits", div)
		enmap.write_map(odirpath + "/sim.fits", sim)
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
	parser.add_argument("-l", "--lmax", type=int, default=10000)
	parser.add_argument("-m", "--mask", type=str, default=None)
	parser.add_argument("-c", "--cont", action="store_true")
	parser.add_argument("-a", "--asteroid-file", type=str, default=None)
	parser.add_argument("--asteroid-list", type=str, default=None)
	parser.add_argument("-F", "--extra-filter", action="store_true")
	parser.add_argument(      "--lknee",        type=str, default=None)
	parser.add_argument("--planet-list",   type=str,   default="Mercury,Venus,Mars,Jupiter,Saturn,Uranus,Neptune")
	parser.add_argument("--planet-rad",    type=float, default=50)
	parser.add_argument("-P", "--mask-planets",  action="store_true")
	parser.add_argument(      "--no-noise-norm", action="store_true")
	parser.add_argument(      "--noiseref",  type=str,   default=None)
	parser.add_argument("-R", "--ref",       type=str,   default=None, help="Reference map to keep downgrades compatible")
	parser.add_argument("-d", "--downgrade", type=int,   default=1)
	args = parser.parse_args()
	from enlib import utils
	with utils.nowarn(): import h5py
	import numpy as np, glob, sys, os, healpy
	from scipy import ndimage
	from enlib import enmap, mpi, planet9, cython, bench, bunch

	comm  = mpi.COMM_WORLD
	scale = 1
	dirs  = sorted(sum([glob.glob(dname) for dname in args.dirs],[]))

	# Storing this takes quite a bit of memory, but it's better than
	# rereading it all the time
	if args.mask: mask = enmap.read_map(args.mask).astype(bool)
	asteroids = planet9.get_asteroids(args.asteroid_file, args.asteroid_list) # None if not specified

	if args.downgrade > 1:
		if args.ref is None:
			if comm.rank == 0:
				print("Downgrading requires a reference shape to ensure that the pixels remain compatible. Specify with --ref")
			sys.exit(1)
		else:
			ref_shape, ref_wcs = enmap.read_map_geometry(args.ref)

	for ind in range(comm.rank, len(dirs), comm.size):
		dirpath = dirs[ind]
		if args.cont and os.path.isfile(dirpath + "/kmap.fits"): continue
		print("Processing %s" % (dirpath))
		info = planet9.hget(dirpath + "/info.hdf")
		# First we'll build frhs = F*R*rhs
		rhs     = enmap.read_map(dirpath + "/rhs.fits")
		fit     = planet9.setup_noise_fit(rhs, args.lknee, dirpath, disable=not args.extra_filter, dump=True)
		R       = planet9.Rmat(rhs.shape, rhs.wcs, info.beam, info.rfact, lmax=args.lmax, lknee1=fit.lknee1, alpha1=fit.alpha1, lknee2=fit.lknee2, alpha2=fit.alpha2)
		R2      = planet9.Rmat(rhs.shape, rhs.wcs, info.beam, info.rfact, lmax=args.lmax, pow=2)
		wmask   = None
		def nmul(a,b):
			if a is None: return b
			else: return a.__imul__(b)
		if args.mask:
			wmask = nmul(wmask, ~mask.extract(rhs.shape, rhs.wcs))
		if asteroids:
			wmask = nmul(wmask, ~planet9.build_asteroid_mask(rhs.shape, rhs.wcs, asteroids, info.mjds))
		if args.mask_planets:
			wmask = nmul(wmask, ~planet9.build_planet_mask(rhs.shape, rhs.wcs, args.planet_list.split(","), info.mjds, r=args.planet_rad*utils.arcmin))
		if wmask is not None:
			rhs  *= wmask
		frhs   = R.apply(rhs)
		unhit  = rhs==0
		del rhs
		# Compute our approximate K
		div = enmap.read_map(dirpath + "/div.fits")
		if wmask is not None: div *= wmask
		kmap  = R2.apply(div); del div
		kmap  = np.maximum(kmap, max(0,np.max(kmap)*1e-10))

		# UNITS:
		#
		# 1. We want frhs/kmap to be a map with mJy per pixel. This will be the case if we
		# include the info.kvals.T[2] information, which evaluates N" = P'Q"P at sparse points
		# in the maps, and as long as we don't do any extra filtering here. Here Q" is the
		# time-domain noise model.
		#
		# 2. However, we know that the map-maker noise model N is wrong, and underestimates
		# the atmospheric noise. This means that kmap, which is supposed to describe the
		# inverse variance in mJy in each pixel, underestimates the noise. We can measure
		# how wrong it is by looking at the mean chisquare of the map: chi = mean(frhs**2/kmap).
		# The factor by which chi is too high (higher than 1) can be robustly measured as
		# A = mean(frhs**2*kmap)/mean(kmap**2). We can use this to correct kmap as long as
		# we also scale frhs the same way. E.g. (frhs,kmap) -> (frhs*B, kmap*B), which
		# results in chi -> chi*B. We want to divide chi by A, which means that B = 1/A.
		#
		# 3. Instead of just eating all that atmospheric noise, we could apply a filter F
		# to get rid of it. My measurments show that a butterworth filter with
		# lknee = 2000/3000/4000 at f090/f150/f220 works well. It's easy to apply this to
		# frhs -> F*frhs, but to preserve the right map units we need to compensate for
		# this in kmap. It is not obvious what this compensation should be. We can consider
		# the filter to be an extra contibution to the noise model such that N" -> HN"H, where
		# H = sqrt(F). In that case, frhs = R'N"d -> R'HN"Hd approx FR'N"d = F frhs, and
		# kmap = diag(R'N"R) -> diag(R'HN"HR') = diag(HR'N"RH). But we don't have the full N",
		# so the latter is hard to evaluate. Both H and R are fourier-diagonal, though, so
		# in theory they could be treated the same way, and we already handle R. The problem
		# with this is that we have info.kvals.T[2] to correct the approximation we make when
		# handling R, but we don't have anything like this to handle H.
		# A quick approximation is to assume that the original N" is a much gentler highpass
		# filter than F. I think N" in practice underestimates lknee by a factor of 2, which
		# would mean that it removes only 1/4 as much power as F does. Let's call this
		# lknee factor alpha = 0.5. The approximate fractional loss of signal from applying
		# F will be q = mean(FGB)/mean(GB), where F = butter(lknee), G = butter(alpha*lknee),
		# and B is the beam. This mean should be over all modes, not just 1D ells.
		# To compensate for this loss we should let kmap -> kmap*q.
		# So to summarize, our approximation is (frhs,kmap) -> (F*frhs, q*kmap).

		# First use info.kvals to normalize kmap. After this frhs/kmap should have
		# proper mJy units
		approx_vals = kmap.at(info.kvals.T[1::-1], order=1)
		exact_vals  = info.kvals.T[2]
		correction  = np.sum(exact_vals*approx_vals)/np.sum(approx_vals**2)
		kmap *= correction

		# Then compensate for any filter we might have applied. This factor was
		# precomputed in Rmat.
		print("correction %8.5f   R.q %8.5f" % (correction,R.q))
		kmap *= R.q

		# Finally characterize any residual noise under/over-estimation. This does
		# not change the flux units, just the noise and S/N ratio.
		if not args.no_noise_norm:
			if args.noiseref:
				# Compute the normalization from a different frhs map. Useful for noiseless sims
				frhs_noiseest = R.apply(enmap.read_map(args.noiseref + "/" + os.path.basename(dirpath) + "/rhs.fits"))
			else: frhs_noiseest = frhs
			A = planet9.get_normalization(frhs_noiseest, kmap)
			print("A", A)
			del frhs_noiseest
			frhs /= A
			kmap /= A

		# Kill all unhit values. This is only necessary because kmap is approximate, and
		# thus doesn't agree perfectly with frhs about how the smoothing smears out the
		# signal in the holes. By setting them explicitly to zero there we avoid dividing
		# by very small numbers there.
		kmap[:]     = np.maximum(kmap, max(np.max(kmap)*1e-4, 1e-12))
		kmap[unhit] = 0
		frhs[unhit] = 0
		# Sharp edges from the mask appears to be causing problems, so remask masked
		# areas after filtering
		if wmask is not None:
			kmap *= wmask
			frhs *= wmask

		# Kmap should contain the noise ivar in mJy at the reference frequency.
		# We can compare this to an approximation based on div alone, which is
		# what my forecast was based on. Given white noise with some ivar per
		# pixel

		if args.downgrade > 1:
			frhs = planet9.downgrade_compatible(frhs, ref_wcs, args.downgrade)
			kmap = planet9.downgrade_compatible(kmap, ref_wcs, args.downgrade)

		#enmap.write_map(dirpath + "/norm.fits", norm)
		enmap.write_map(dirpath + "/frhs.fits", frhs)
		enmap.write_map(dirpath + "/kmap.fits", kmap)
		sigma = frhs*0
		cython.solve(frhs, kmap, sigma, klim=np.percentile(kmap, 90)*1e-3)
		enmap.write_map(dirpath + "/sigma.fits", sigma)
		sps, ls = (np.abs(enmap.fft(sigma))**2).lbin()
		sps /= np.mean(kmap > 0)
		np.savetxt(dirpath + "/sps.txt", np.array([ls,sps]).T, fmt="%15.7e")
		del kmap, sigma, unhit, wmask

elif mode == "maskmore":
	parser = argparse.ArgumentParser()
	parser.add_argument("maskmore", help="dummy")
	parser.add_argument("dirs", nargs="+")
	parser.add_argument("-m", "--mask", type=str, default=None)
	parser.add_argument("-a", "--asteroid-file", type=str, default=None)
	parser.add_argument("--asteroid-list", type=str, default=None)
	parser.add_argument("--planet-list",   type=str,   default="Mercury,Venus,Mars,Jupiter,Saturn,Uranus,Neptune")
	parser.add_argument("--planet-rad",    type=float, default=50)
	parser.add_argument("-P", "--mask-planets", action="store_true")
	args = parser.parse_args()
	from enlib import utils
	with utils.nowarn(): import h5py
	import numpy as np, glob, sys, os
	from scipy import ndimage
	from enlib import enmap, mpi, planet9

	comm  = mpi.COMM_WORLD
	scale = 1
	dirs  = sorted(sum([glob.glob(dname) for dname in args.dirs],[]))

	# Storing this takes quite a bit of memory, but it's better than
	# rereading it all the time
	if args.mask: mask = enmap.read_map(args.mask).astype(bool)
	asteroids = planet9.get_asteroids(args.asteroid_file, args.asteroid_list)

	for ind in range(comm.rank, len(dirs), comm.size):
		dirpath = dirs[ind]
		print("Processing %s" % (dirpath))
		info    = planet9.hget(dirpath + "/info.hdf")
		frhs    = enmap.read_map(dirpath + "/frhs.fits")
		wmask   = None
		def nmul(a,b):
			if a is None: return b
			else: return a.__imul__(b)
		if args.mask:
			wmask = nmul(wmask, ~mask.extract(frhs.shape, frhs.wcs))
		if asteroids:
			wmask = nmul(wmask, ~planet9.build_asteroid_mask(frhs.shape, frhs.wcs, asteroids, info.mjds))
		if args.mask_planets:
			wmask = nmul(wmask, ~planet9.build_planet_mask(rhs.shape, rhs.wcs, args.planet_list.split(","), info.mjds, r=args.planet_rad))
		if wmask is not None:
			frhs  *= wmask
		enmap.write_map(dirpath + "/frhs.fits", frhs)
		del frhs
		kmap  = enmap.read_map(dirpath + "/kmap.fits")
		a = np.sum(kmap!=0)
		kmap *= wmask
		b = np.sum(kmap!=0)
		print(a-b)
		enmap.write_map(dirpath + "/kmap.fits", kmap)
		del kmap, wmask

elif mode == "extract":
	parser = argparse.ArgumentParser()
	parser.add_argument("extract", help="dummy")
	parser.add_argument("box")
	parser.add_argument("dirs", nargs="+")
	parser.add_argument("odir")
	parser.add_argument("-c", "--cont", action="store_true")
	parser.add_argument("-F", "--fields", type=str, default="rhs,div,frhs,kmap")
	args = parser.parse_args()
	from enlib import utils
	with utils.nowarn(): import h5py
	import numpy as np, glob, sys, os
	from scipy import ndimage
	from enlib import enmap, mpi, planet9

	comm  = mpi.COMM_WORLD
	odir  = args.odir
	dirs  = sorted(sum([glob.glob(dname) for dname in args.dirs],[]))
	# from dec1:dec2,ra1:ra2 to [[dec1,ra1],[dec2,ra2]]
	box   = np.array([[float(c) for c in word.split(":")] for word in args.box.split(",")]).T*utils.degree
	fields = args.fields.split(",")

	shape, wcs = enmap.read_map_geometry(dirs[0]+"/%s.fits" % fields[0])
	shape, wcs = enmap.Geometry(shape, wcs).submap(box)

	if comm.rank == 0:
		utils.mkdir(args.odir)
		enmap.write_map(args.odir + "/area.fits", enmap.zeros(shape, wcs, np.int16))

	for ind in range(comm.rank, len(dirs), comm.size):
		idirpath = dirs[ind]
		odirpath = args.odir + "/" + os.path.basename(idirpath)
		if args.cont and os.path.isfile(odirpath + "/info.hdf"):
			continue
		print("Processing %s" % (idirpath))
		skip = False
		for fname in ["div", "kmap", "rhs", "frhs"]:
			if fname not in fields: continue
			map = enmap.read_map(idirpath + "/%s.fits" % fname, geometry=(shape, wcs))
			if fname in ["div", "kmap"] and np.all(map==0):
				skip = True
				break
			utils.mkdir(odirpath)
			enmap.write_map(odirpath + "/%s.fits" % fname, map)
		if skip: continue
		info  = planet9.hget(idirpath + "/info.hdf")
		planet9.hput(odirpath + "/info.hdf", info)

elif mode == "combine":
	# planet9 map and planet9 filter operate on per-array, per-frequency maps because
	# they need to handle the different beams etc. But the later steps don't need to
	# care about that, and one can therefore save I/O by combining the different dirs
	# corresponding to the same time stamp, which is what planet9 combine does. It
	# can also optionally downgrade the the maps. Unlike the other commands, this one
	# makes some strong assumptions about the directory name format, which is assumed to
	# be season_patch_array_freq_daynight_ctime. idirs with the same season, patch and
	# ctime will be combined.
	parser = argparse.ArgumentParser()
	parser.add_argument("combine", help="dummy")
	parser.add_argument("idirs", nargs="+")
	parser.add_argument("odir")
	parser.add_argument("-R", "--ref",       type=str, default=None, help="Reference map to keep downgrades compatible")
	parser.add_argument("-d", "--downgrade", type=int, default=1)
	parser.add_argument("-v", "--verbose",   action="store_true")
	parser.add_argument("-c", "--cont",      action="store_true")
	args = parser.parse_args()
	import numpy as np, time, os, glob, sys
	from enlib import utils
	with utils.nowarn(): import h5py
	from scipy import ndimage
	from enlib import enmap, utils, mpi, planet9

	utils.mkdir(args.odir)
	comm  = mpi.COMM_WORLD
	dtype = np.float32

	idirs = sorted(sum([glob.glob(idir) for idir in args.idirs],[]))

	if args.downgrade > 1:
		if args.ref is None:
			if comm.rank == 0:
				print("Downgrading requires a reference shape to ensure that the pixels remain compatible. Specify with --ref")
			sys.exit(1)
		else:
			ref_shape, ref_wcs = enmap.read_map_geometry(args.ref)

	# Find groups that need to be combined
	dirgnames = []
	for idir in idirs:
		toks  = os.path.basename(idir).split("_")
		gname = "_".join(toks[:-4]+toks[-1:])
		dirgnames.append(gname)
	gnames, rinds = np.unique(dirgnames, return_inverse=True)
	groups = [[] for gname in gnames]
	for i, ri in enumerate(rinds):
		groups[ri].append(i)
	
	if comm.rank == 0:
		print("Combining %d dirs into %d dirs" % (len(idirs), len(groups)))

	# Then process the groups
	for ind in range(comm.rank, len(groups), comm.size):
		gname = gnames[ind]
		odir = "%s/%s" % (args.odir, gname)
		if args.cont and os.path.isfile(odir + "/frhs.fits") and os.path.isfile(odir + "/kmap.fits") and os.path.isfile(odir + "/info.hdf"): continue
		gdirs = [idirs[i] for i in groups[ind]]
		print("%3d processing %4d/%d %s  %2d" % (comm.rank, ind+1, len(groups), gname, len(gdirs)))
		# Read in and accumulate the individual files
		frhs_tot, kmap_tot, mjds = None, None, []
		for di, gdir in enumerate(gdirs):
			frhs = enmap.read_map("%s/frhs.fits" % gdir)
			if args.downgrade > 1:
				frhs = planet9.downgrade_compatible(frhs, ref_wcs, args.downgrade)  #*args.downgrade**2
			kmap = enmap.read_map("%s/kmap.fits" % gdir)
			if args.downgrade > 1:
				kmap = planet9.downgrade_compatible(kmap, ref_wcs, args.downgrade)  #*args.downgrade**2
			with h5py.File("%s/info.hdf" % gdir, "r") as hfile:
				mjd = hfile["mjd"][()]
			if frhs_tot is None: frhs_tot  = frhs
			else:                frhs_tot += frhs
			if kmap_tot is None: kmap_tot  = kmap
			else:                kmap_tot += kmap
			mjds.append(mjd)
		# And output the combined results
		mjd  = np.mean(mjds)
		utils.mkdir(odir)
		enmap.write_map(odir + "/frhs.fits", frhs_tot)
		enmap.write_map(odir + "/kmap.fits", kmap_tot)
		with h5py.File(odir + "/info.hdf", "w") as hfile:
			hfile["mjd"] = mjd

elif mode == "find":
	# planet finding mode. Takes as input the output directories from the filter step.
	parser = argparse.ArgumentParser()
	parser.add_argument("find", help="dummy")
	parser.add_argument("idirs", nargs="+")
	parser.add_argument("area")
	parser.add_argument("odir")
	parser.add_argument("-O", "--order",     type=int, default=1)
	parser.add_argument("-d", "--downgrade", type=int, default=1)
	parser.add_argument("-m", "--mask",      type=str, default=None)
	parser.add_argument("-T", "--tsize",     type=int, default=1200)
	parser.add_argument("-P", "--pad",       type=int, default=60)
	parser.add_argument("-N", "--npertile",  type=int, default=-1)
	parser.add_argument("-v", "--verbose",   action="count", default=1)
	parser.add_argument("-q", "--quiet",     action="count", default=0)
	parser.add_argument(      "--rref",      type=float, default=300)
	parser.add_argument(      "--rmin",      type=float, default=300)
	parser.add_argument(      "--rmax",      type=float, default=2000)
	parser.add_argument(      "--dr",        type=float, default=20)
	parser.add_argument(      "--vmax",      type=float, default=6.28)
	parser.add_argument(      "--vmin",      type=float, default=4.16)
	parser.add_argument(      "--dv",        type=float, default=0.1)
	parser.add_argument(      "--static",    action="store_true")
	parser.add_argument(      "--rinf",      type=int, default=1, help="Include infinite distance in the r search. Useful for rejecting non-moving objects")
	parser.add_argument("-g", "--full-geo",  action="store_true", help="Read the geometry of each input data set instead of assuming that ones with the same prefix have the same geometry")
	parser.add_argument("-i", "--invert",    action="store_true")
	parser.add_argument("-c", "--cont",      action="store_true")
	parser.add_argument("-f", "--fscale",    type=float, default=1, help="Multiply fluxes by this factor")
	parser.add_argument("-e", "--escale",    type=float, default=1, help="Multiply flux errors by this factor (in addition to that implied by fscale)")
	parser.add_argument(      "--only",      type=str,   default=None)
	parser.add_argument("-S", "--scratch",   type=str,   default=None)
	args = parser.parse_args()
	import numpy as np, time, os, glob
	from enlib import utils
	with utils.nowarn(): import h5py
	from scipy import ndimage, stats
	from enlib import enmap, utils, mpi, parallax, cython, ephemeris, statdist, planet9

	utils.mkdir(args.odir)
	comm  = mpi.COMM_WORLD
	dtype = np.float32
	shape, wcs = planet9.get_geometry(args.area)
	tsize, pad = args.tsize, args.pad
	nphi = abs(utils.nint(360./wcs.wcs.cdelt[0]))
	only = [int(w) for w in args.only.split(",")] if args.only else None
	verbose = args.verbose - args.quiet

	def build_rlist(rref, dr, rmin, rmax, rinf=False):
		# r' = dr0*(r/rref)² => dr/r²[rmin:r] = dr0*dx/rref²[0:x] => 1/r1-1/r = x*dr0/rref²
		# => r = 1/(1/r1 - x*dr0/rref²)
		# what is xmax? xmax = (1/r1-1/r2)*rref²/dr0
		xmax = int(np.ceil((1/rmin-1/rmax)*rref**2/dr))
		x    = np.arange(xmax+1)
		r    = 1/(np.maximum(1/rmin - x*dr/rref**2,1/rmax))
		if rinf: r = np.concatenate([r,[1e9]])
		return r

	# Our parameter search space. Distances in AU, speeds in arcmin per year.
	# smoothing from r step: 0.2' = 1au*dr/r**2/4 => dr = 20 au at r = 300 au
	# vmax, vmin:
	#  based on 2*pi*a**2*(1-e**2)**0.5/((a/au)**1.5 * years)/r**2
	#  this goes as a**0.5, so only a weak function of a. Also weak function of e
	#  as long as e is not cose to 1. Let's assume e in [0,0.7] and a in [400,800].
	#  Then the highest speed is 3.5'/yr @ 400 AU and the lowest is 2.8'/yr. This translates
	#  to 6.2'/yr and 5.0'/yr @ 300 AU, though the actual vmax at 300 AU is a bit lower
	# Let's use 6.5 to 5.0.
	# Update: reading off from fig 15 in the P9 hyp paper, it looks like vref should go
	# from 4.15 to 6.30 to be safe. That's about twice as expensive.sadly. And rmin should
	# be around 300.
	ym = utils.arcmin/utils.yr2days
	# The radius at which the speed and radius steps are specified. Values at other
	# radii are scaled from these
	rref = args.rref
	# The radial stepping
	rlist = build_rlist(rref, args.dr, args.rmin, args.rmax, rinf=args.rinf)
	nr    = len(rlist)
	# The speed bounds as a function of radius
	vmaxs = args.vmax * (rlist/rref)**-2
	vmins = args.vmin * (rlist/rref)**-2
	vmins = np.maximum(0, vmins-args.dv)
	nv    = int(np.ceil(np.max(vmaxs)/args.dv))
	# Convert to physical units
	vmaxs *= ym; vmins *= ym; dv = args.dv*ym
	if comm.rank == 0 and verbose >= 2:
		print("Parameter search space:\n%8s %8s %8s" % ("r", "vmin", "vmax"))
		for r, vmin, vmax in zip(rlist, vmins, vmaxs):
			print("%8.2f %8.2f %8.2f" % (r, vmin/ym, vmax/ym))

	#nparam = 0
	#for ri, (r, vmin, vmax) in enumerate(zip(rlist, vmins, vmaxs)):
	#	for vy in np.arange(-nv,nv+1)*dv:
	#		for vx in np.arange(-nv,nv+1)*dv:
	#			vmag = (vy**2+vx**2)**0.5
	#			if vmag < vmin or vmag > vmax: continue
	#			nparam += 1
	#print("nparam", nparam)
	#1/0

	# How many tiles will we have?
	if tsize == 0: nty, ntx = 1, 1
	else: nty, ntx = (np.array(shape[-2:])+tsize-1)//tsize
	ntile = nty*ntx

	idirs = sorted(sum([glob.glob(idir) for idir in args.idirs],[]))

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
		for name in ["sigma_plain", "param_map", "param_map_full", "sigma_eff", "sigma_eff_full", "hit_tot", "cands"]:
			utils.mkdir(args.odir + "/" + name)

	# Get the pixel bounding box of each input map in terms of our output area
	if args.full_geo:
		pboxes = planet9.read_pboxes(idirs, wcs, comm=comm, verbose=verbose>=2)
	else:
		# Here we assume that dirs with the same prefix have the same geometry
		prefixes = np.array(["_".join(idir.split("_")[:-1]) for idir in idirs])
		upres, inds, rinds = np.unique(prefixes, return_index=True, return_inverse=True)
		uboxes   = planet9.read_pboxes([idirs[i] for i in inds], wcs, upres, comm=comm, verbose=verbose>=2)
		pboxes   = uboxes[rinds]

	# To speed up the likelihood search we will combine all maps that have the same
	# timestamp, since they will all be displaced the same way anyway. So split our
	# idirs into a list of groups with the same time
	tstamps = np.array([idir.split("_")[-1] for idir in idirs])
	utimes, inds, rinds = np.unique(tstamps, return_index=True, return_inverse=True)
	groups = [[] for i in inds]
	for i, ri in enumerate(rinds):
		groups[ri].append(i)

	if args.scratch:
		import shutil, socket
		prefix = args.scratch
		hname  = socket.gethostname()
		# Copy over to fast local system. This node communicator lets us communicate with
		# all processes inside the same node
		node_comm = comm.Split_type(mpi.COMM_TYPE_SHARED)
		# Figure out the full set of dirs that are used on this node
		mydirs = [idirs[i] for g in groups[comm_intra.rank::comm_intra.size] for i in g]
		hdirs  = np.unique(node_comm.allreduce(mydirs))
		if node_comm.rank == 0:
			print("Host %s needs %4d/%d (%3.0f%%) files" % (hname, len(hdirs), len(idirs), 100.0*len(hdirs)/len(idirs)))
		# Then copy them over. Only ncopy tasks will try copying at the same time to avoid
		# trashing the file system
		ncopy = min(4, node_comm.size)
		if node_comm.rank < ncopy:
			for hdir in hdirs[node_comm.rank::ncopy]:
			#for ind in range(node_comm.rank, len(hdirs), node_comm.size):
				#hdir = hdirs[ind]
				utils.mkdir("%s/%s" % (prefix,hdir))
				print("%3d Copying %s to %s" % (comm.rank, hdir, hname))
				for fname in ["frhs.fits", "kmap.fits", "info.hdf"]:
					shutil.copyfile("%s/%s" % (hdir, fname), "%s/%s/%s" % (prefix, hdir, fname))
		# Finally replace the contents of idirs with the new paths
		idirs = ["%s/%s" % (prefix, idir) for idir in idirs]
		comm.Barrier()

	# Loop over tiles
	for ti in range(comm_inter.rank, ntile, comm_inter.size):
		ty, tx = ti//ntx, ti%ntx
		if only and (ty != only[0] or tx != only[1]): continue
		if tsize > 0:
			def oname(name):
				fname, fext = os.path.splitext(name)
				return "%s/%s/tile%03d_%03d%s" % (args.odir, fname, ty, tx, fext)
			pixbox = np.array([[ty*tsize-pad,tx*tsize-pad],[(ty+1)*tsize+pad,(tx+1)*tsize+pad]])
		else:
			def oname(name): return args.odir + "/" + name
			pixbox = np.array([[-pad,-pad],[shape[-2]+pad,shape[-1]+pad]])
		if args.cont and os.path.exists(oname("cands.txt")): continue
		if comm_intra.rank == 0:
			print("group %3d processing tile %3d %3d" % (comm_inter.rank, ty, tx))
		# Get the shape of the sliced, downgraded tiles
		tshape_full, twcs_full = enmap.slice_geometry(shape, wcs, [slice(pixbox[0,0],pixbox[1,0]),slice(pixbox[0,1],pixbox[1,1])], nowrap=True)
		tshape, twcs = enmap.downgrade_geometry(tshape_full, twcs_full, args.downgrade)
		# Read in our mask, if any
		if args.mask:
			mask = enmap.read_map(args.mask, geometry=(tshape_full, twcs_full))
			if args.downgrade > 0: mask = enmap.downgrade(mask, args.downgrade)
			mask = 1-mask # make mask 1 in OK regions and 0 in bad regions
		# Read in our tile data
		frhss, kmaps, mjds = [], [], []
		for gind in range(comm_intra.rank, len(groups), comm_intra.size):
			group = groups[gind]
			gfrhs, gkmap, gmjds = None, None, []
			for i, ind in enumerate(group):
				if not planet9.overlaps(pboxes[ind], pixbox, nphi):
					#print(idirs[ind], "does not overlap", pixbox.reshape(-1), pboxes[ind].reshape(-1))
					continue
				idir = idirs[ind]
				lshape, lwcs = enmap.read_map_geometry(idir + "/frhs.fits")
				pixbox_loc   = pixbox - enmap.pixbox_of(wcs, lshape, lwcs)[0]
				kmap = enmap.read_map(idir + "/kmap.fits", pixbox=pixbox_loc).astype(dtype)
				if args.downgrade > 1: kmap = enmap.downgrade(kmap, args.downgrade)   #*args.downgrade**2
				# Skip tile if it's empty
				if np.any(~np.isfinite(kmap)) or np.all(kmap < 1e-10):
					if verbose >= 2: print("%3d skip %3d %2d %s (nan or unexposed)" % (comm.rank, gind, i, idir))
					continue
				else:
					if verbose >= 2: print("%3d read %3d %2d %s" % (comm.rank, gind, i, idir))
				frhs = enmap.read_map(idir + "/frhs.fits", pixbox=pixbox_loc).astype(dtype)
				if args.downgrade > 1: frhs = enmap.downgrade(frhs, args.downgrade)   #*args.downgrade**2
				if args.invert: frhs *= -1
				with h5py.File(idir + "/info.hdf", "r") as hfile: mjd = hfile["mjd"][()]
				# Apply mask
				if args.mask:
					frhs *= mask
					kmap *= mask
				# Flux correction if any
				frhs /= args.fscale    * args.escale**2
				kmap /= args.fscale**2 * args.escale**2
				if gfrhs is None: gfrhs = frhs*0
				if gkmap is None: gkmap = kmap*0
				gfrhs += frhs
				gkmap += kmap
				gmjds.append(mjd)
			# No data found for this group, so skip the whole thing
			if len(gmjds) == 0:
				continue
			frhss.append(gfrhs)
			kmaps.append(gkmap)
			mjds.append(np.mean(gmjds))

		nlocal = len(frhss)
		ntot   = comm_intra.allreduce(nlocal)

		# Handle case where there's no data
		if ntot == 0:
			if comm_intra.rank == 0:
				# Output dummy stuff in areas we don't hit
				dummy_map   = planet9.unpad(enmap.zeros(tshape, twcs, dtype), pad)
				dummy_param = planet9.unpad(enmap.zeros((nr,5)+tshape, twcs, dtype), pad)
				for name in ["sigma_plain.fits", "sigma_eff.fits", "hit_tot.fits"]:
					enmap.write_map(oname(name), dummy_map)
				enmap.write_map(oname("param_map_full.fits"), dummy_param)
				enmap.write_map(oname("param_map.fits"),      dummy_param[0])
				enmap.write_map(oname("sigma_eff_full.fits"), dummy_param[:,0])
				del dummy_map, dummy_param
				with open(oname("cands.txt"),"w") as ofile: ofile.write("")
				print("%3d nothing to do for %s" % (comm.rank, ti))
			continue
		# Skip tasks that don't have anything to do
		comm_good = comm_intra.Split(nlocal > 0, comm_intra.rank)
		if nlocal == 0:
			print("%3d nothing to do for %s" % (comm.rank, ti))
			continue

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
		if comm_good.size > 1:
			frhs_tot = utils.allreduce(frhs_tot, comm_good)
			kmap_tot = utils.allreduce(kmap_tot, comm_good)
		klim = np.percentile(kmap_tot,90)*1e-3
		if comm_good.rank == 0:
			#print("A", np.std(frhs_tot), np.std(kmap_tot))
			#sigma = planet9.solve(frhs_tot, kmap_tot)
			sigma = frhs_tot*0
			cython.solve(frhs_tot, kmap_tot, sigma, klim=klim)
			enmap.write_map(oname("sigma_plain.fits"), planet9.unpad(sigma, pad))
		#print(klim)

		# Perform the actual parameter search
		if comm_good.rank == 0:
			sigma_max = enmap.full ((nr,) +sigma.shape, sigma.wcs, -np.inf, sigma.dtype)
			param_max = enmap.zeros((nr,5)+sigma.shape, sigma.wcs, dtype)
		ntest = 0
		for ri, (r, vmin, vmax) in enumerate(zip(rlist, vmins, vmaxs)):
			# The search space is typically much bigger for low r than high r. If we
			# just do the raw, unnormalized max over all the different rs, then setting
			# a lower rmin makes us less sensitive to stuff at high r, which isn't good.
			# we can avoid this by making sigma_max and param_max for each r, then
			# normalizing, which penalizes rs with larger search spaces, and then combining
			# into a single overall sigma_max and param_max.
			#
			# This will also impact our limit calculations. The current ones assume that we
			# one can infer the normalization just given a param map. That's no longer true
			# if we rescale each r separately.
			#
			# We could compensate by updating the frhs and kmap parts that are built into
			# params too, but that would sacrifice our ability to get out the actual flux.
			#
			# Maybe a better approach is to expand params from [5,ny,nx] to
			# [nr,5,ny,nx]. This would make it ~10x bigger, though. It's currently 2.5
			# GB big, so that would put it at 25 GB or more, which starts to be a bit big.
			# We do this search tilewise, though, so that full size doesn't matter much.
			#
			# Ok, let's go with the per-r approach
			for vy in np.arange(-nv,nv+1)*dv:
				for vx in np.arange(-nv,nv+1)*dv:
					vmag = (vy**2+vx**2)**0.5
					if vmag < vmin or vmag > vmax: continue
					#def close(a,b): return np.abs(a/b-1) < 0.01
					#if not (close(r,857.143) and close(vy/ym,-0.600) and close(vx/ym, 0.500)):
					#	#print("skipped", r, vy, vx)
					#	continue
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
					if comm_good.size > 1:
						frhs_tot = utils.allreduce(frhs_tot, comm_good)
						kmap_tot = utils.allreduce(kmap_tot, comm_good)
					t2 = time.time()
					if comm_good.rank == 0:
						cython.solve(frhs_tot, kmap_tot, sigma, klim=klim)
						cython.update_total(sigma, sigma_max[ri], param_max[ri], hit_tot, frhs_tot, kmap_tot, r, vy, vx)
						t3 = time.time()
						if verbose >= 1:
							print("%2d %5.0f %5.2f %5.2f  %8.3f ms %8.3f ms" % (comm_inter.rank, r, vy/ym, vx/ym, (t2-t1)*1e3, (t3-t2)*1e3))
					ntest += 1
		if comm_good.rank == 0:
			# Output the full param map with the speed in units of arcmin-per-year. The is the
			# only thins we need to output per-r, the rest can be recovered from it later if
			# such details are needed.
			param_max[:,1:3] /= ym
			enmap.write_map(oname("param_map_full.fits"), planet9.unpad(param_max, pad))
			param_max[:,1:3] *= ym

			# FIXME: The rest is obsolete. Call finish_find to finish the analysis

			## Normalize each r and merge into total quantities
			#mask  = kmap_tot > klim
			#sigma_eff = sigma_max[0]-np.inf
			#param_map = param_max[0]*0
			#hit_dummy = hit_tot*0
			#for ri, r in enumerate(rlist):
			#	# Normalize each before merging
			#	b, a          = planet9.build_mu_sigma_map(sigma_max[ri], mask=mask)
			#	sigma_max[ri] = (sigma_max[ri]-b)/a
			#	cython.merge_param_maps(param_max[ri], sigma_max[ri], param_map, sigma_eff)
			## Normalize again after merging, since we're merging by max instead of just averaging
			#b, a      = planet9.build_mu_sigma_map(sigma_eff, mask=mask)
			#sigma_eff = (sigma_eff-b)/a
			## Find candidates
			#cands = planet9.find_candidates(sigma_eff, param_map, hit_tot, snmin=args.snmin, pad=pad//args.downgrade)
			#np.savetxt(oname("cands.txt"), np.array([cands[:,1]/utils.degree, cands[:,0]/utils.degree,
			#	cands[:,2], cands[:,3], cands[:,4], cands[:,5], cands[:,6]/ym, cands[:,7]/ym, cands[:,8]/ntest]).T, fmt="%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %4.2f")
			#enmap.write_map(oname("param_map.fits"),      planet9.unpad(param_map,      pad))
			#enmap.write_map(oname("sigma_eff.fits"),      planet9.unpad(sigma_eff,      pad))
			#enmap.write_map(oname("sigma_eff_full.fits"), planet9.unpad(sigma_max,      pad))
			#enmap.write_map(oname("hit_tot.fits"),        planet9.unpad(hit_tot,        pad))

			# We will handle the limit stuff elsewhere, since it's not r-dependent and depends on some
			# tunable parameters to get right.

			## The kmap_tot = da used here only corresponds to the last iteration, but
			## since kmaps are pretty smooth, it shouldn't matter much.
			#mask = hit_tot > 0
			#with utils.nowarn():
			#	da = kmap_tot*0; da[mask] = kmap_tot[mask]**-0.5
			## Our selection threshold is sigma_eff > snmin => (sigma-mu)/s > snmin => sigma > snmin*s+mu
			## Hence our sensitivity is da*sigma = da*(snmin*s+mu)
			#limit = da*(args.snmin*a+b)
			#enmap.write_map(oname("limit_map.fits"), planet9.unpad(limit, pad))

elif mode == "classify":
	# Catalog classifying/pruning/masking mode. We read in one or more catalogs, merge then, and
	# then classify them as bad (not moving or other obvious problems) (0), p9-like (2)
	# (has an orbit similar to p9) or some other orbit but otherwise ok (1).
	# The point is that we might want to use a lower S/N threshold for detection for
	# p9-like candidates, since there will be fewer of them because of the smaller parameter
	# space.
	parser = argparse.ArgumentParser()
	parser.add_argument("classify", help="dummy")
	parser.add_argument("icats", nargs="+")
	parser.add_argument("ocat")
	parser.add_argument("-m", "--mask",     type=str,   default=None, help="classify anything inside mask as bad")
	parser.add_argument(      "--mask-pad", type=float, default=3,    help="arcminutes to pad mask by")
	parser.add_argument("-r", "--rmin",     type=float, default=6,    help="minimum distance between cands")
	args = parser.parse_args()
	import numpy as np, time, os, glob
	from enlib import enmap, utils, planet9
	from scipy import spatial
	ym = utils.arcmin/utils.yr2days

	def read_catalog(fname):
		"""ra dec snr flux dflux r vy vx hitfrac"""
		with utils.nowarn():
			cat = np.loadtxt(fname, usecols=range(8), ndmin=2).reshape(-1,8)
		cat[:,:2]  = cat[:,1::-1]*utils.degree
		cat[:,-2:]*= ym
		return cat

	# Merge candidates that are very close to each other, keeping the one with
	# highest S/N
	def merge_candidates(cat, rlim=2*utils.arcmin):
		dec, ra = cat.T[:2]
		pos     = utils.ang2rect([ra,dec]).T
		tree    = spatial.cKDTree(pos)
		groups  = tree.query_ball_tree(tree, rlim)
		done    = np.zeros(len(cat),bool)
		ocat    = []
		nmerged = []
		for gi, group in enumerate(groups):
			group = np.array(group)
			group = group[~done[group]]
			if len(group) == 0: continue
			gcat = cat[group]
			best = np.argmax(gcat[:,2])
			ocat.append(gcat[best])
			nmerged.append(len(group))
			done[group] = True
		ocat = np.array(ocat).reshape(-1,cat.shape[1])
		nmerged = np.array(nmerged)
		return ocat, nmerged

	# Read in all the catalogs and concatenate them
	cat  = [read_catalog(fname) for fname in args.icats]
	cat  = np.concatenate(cat,0)
	cat, nmerged = merge_candidates(cat, rlim=args.rmin*utils.arcmin)

	masked = False
	if args.mask:
		mask = enmap.read_map(args.mask) > 0.5
		if args.mask_pad > 0:
			mask = (~mask).distance_transform(rmax=args.mask_pad*utils.arcmin) <= utils.arcmin
		masked = mask.at(cat.T[:2], order=0) > 0
		print(np.sum(masked), np.mean(masked))
		del mask

	# And apply the classification
	bad  = cat[:,5] >= 1e5
	bad |= masked
	ok   = ~bad
	p9   = planet9.p9_like(ra=cat[:,1], dec=cat[:,0], r=cat[:,5], vy=cat[:,6], vx=cat[:,7])
	status= np.full(len(cat),int)
	status[bad] = 0
	status[ok]  = 1
	status[p9]  = 2

	# Sorty by type, then by S/N
	order = np.lexsort([cat[:,2], status])[::-1]
	cat   = cat[order]
	status= status[order]

	np.savetxt(args.ocat, np.array([cat[:,1]/utils.degree, cat[:,0]/utils.degree,
		cat[:,2], cat[:,3], cat[:,4], cat[:,5], cat[:,6]/ym, cat[:,7]/ym, status]).T,
		fmt="%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %d")

elif mode == "analyse":
	parser = argparse.ArgumentParser()
	parser.add_argument("analyse", help="dummy")
	parser.add_argument("cands")
	parser.add_argument("idirs", nargs="+")
	parser.add_argument("odir")
	parser.add_argument("-O", "--order",     type=int,   default=1)
	parser.add_argument("-d", "--downgrade", type=int,   default=1)
	parser.add_argument("-m", "--mask",      type=str,   default=None)
	parser.add_argument("-T", "--tsize",     type=int,   default=120)
	parser.add_argument("-P", "--pad",       type=int,   default=60)
	parser.add_argument("-n", "--ncand",     type=int,   default=None)
	parser.add_argument("-N", "--nper",      type=int,   default=-1)
	parser.add_argument("-f", "--fscale",    type=float, default=1, help="Multiply fluxes by this factor")
	parser.add_argument("-e", "--escale",    type=float, default=1, help="Multiply flux errors by this factor (in addition to that implied by fscale)")
	parser.add_argument("-v", "--verbose",   action="store_true")
	parser.add_argument("-g", "--full-geo",  action="store_true", help="Read the geometry of each input data set instead of assuming that ones with the same prefix have the same geometry")
	parser.add_argument("-c", "--cont",      action="store_true")
	parser.add_argument(      "--only",      type=str,   default=None)
	args = parser.parse_args()
	import numpy as np, time, os, glob
	from enlib import utils
	with utils.nowarn(): import h5py
	from scipy import ndimage, stats
	from enlib import enmap, utils, mpi, parallax, cython, ephemeris, statdist, planet9

	# Using half-resolution, I've found that the recovered flux is low by a factor 0.62,
	# which one can correct for using --fscale 1.61. This correction is smaller for a much
	# slower full-resolution search. There are also signs of 10%-level position-dependent
	# systematic errors in the recovered flux. The recovered errors are also only accurate at
	# the 10%-level.

	utils.mkdir(args.odir)
	comm  = mpi.COMM_WORLD
	dtype = np.float32
	tsize, pad = args.tsize, args.pad
	only  = [int(w) for w in args.only.split(",")] if args.only else None
	ym    = utils.arcmin/utils.yr2days # 1 arcmin per year, in units of radians per day

	idirs = sorted(sum([glob.glob(idir) for idir in args.idirs],[]))

	# We will make per-candidate maps here, so the reference geometry doesn't really
	# matter as long as all the input maps are compatible with each other. So we will
	# just use the first input file
	_, wcs = enmap.read_map_geometry(idirs[0] + "/frhs.fits")
	nphi  = abs(utils.nint(360./wcs.wcs.cdelt[0]))

	# Parallelization
	if args.nper < 0: nper = len(idirs)//(-args.nper)
	else:             nper = args.nper
	nper = max(1,min(comm.size,nper))
	# Build intra- and inter-tile communicators
	comm_intra = comm.Split(comm.rank//nper, comm.rank %nper)
	comm_inter = comm.Split(comm.rank %nper, comm.rank//nper)

	# Get the pixel bounding box of each input map in terms of our output area
	if args.full_geo:
		pboxes = planet9.read_pboxes(idirs, wcs, comm=comm, verbose=args.verbose)
	else:
		# Here we assume that dirs with the same prefix have the same geometry
		prefixes = np.array(["_".join(idir.split("_")[:-1]) for idir in idirs])
		upres, inds, rinds = np.unique(prefixes, return_index=True, return_inverse=True)
		uboxes   = planet9.read_pboxes([idirs[i] for i in inds], wcs, upres, comm=comm, verbose=args.verbose)
		pboxes   = uboxes[rinds]

	# Read in our candidates. We need the position and motion parameters
	cands = np.loadtxt(args.cands, usecols=[0,1,5,6,7]).reshape(-1,5)
	if args.ncand:
		cands = cands[:args.ncand]
	# Loop over candidates
	for cind in range(comm_inter.rank, len(cands), comm_inter.size):
		if only is not None and cind not in only:
			continue
		odir = "%s/cand_%03d" % (args.odir, cind+1)
		if args.cont and os.path.isfile(odir + "/fluxes.txt"):
			continue
		cand = cands[cind]
		ra, dec = cand[:2]*utils.degree
		r       = cand[2]
		vy, vx  = cand[3:]*ym
		if args.verbose and comm_intra.rank == 0:
			print("%3d %4d/%d ra %8.3f dec %8.3f vy %8.3f vx %8.3f" % (comm_inter.rank, cind+1, len(cands), ra/utils.degree, dec/utils.degree, vy/ym, vx/ym))
		y, x    = utils.nint(enmap.sky2pix(None, wcs, [dec,ra], safe=False))
		pixbox  = np.array([[y-tsize//2-pad,x-tsize//2-pad],[y-tsize//2+tsize+pad,x-tsize//2+tsize+pad]])
		oshape  = pixbox[1]-pixbox[0]
		owcs    = wcs.deepcopy(); owcs.wcs.crpix -= pixbox[0,::-1]
		# Loop over the datasets
		frhs_tot = enmap.zeros(oshape, owcs, dtype)
		kmap_tot = enmap.zeros(oshape, owcs, dtype)
		frhs_tot_plain = enmap.zeros(oshape, owcs, dtype)
		kmap_tot_plain = enmap.zeros(oshape, owcs, dtype)
		fluxes   = np.zeros((len(idirs),2), dtype)
		mjds     = np.zeros(len(idirs), dtype)
		for dind in range(comm_intra.rank, len(idirs), comm_intra.size):
			idir = idirs[dind]
			# Skip idirs that don't overlap
			if not planet9.overlaps(pboxes[dind], pixbox, nphi):
				continue
			# Translate our target pixbox to local pixels. We already have the local pixbox, so this is
			# just a subtraction
			lpixbox = pixbox - pboxes[dind][0]
			# And read the data
			frhs = enmap.read_map(idir + "/frhs.fits", pixbox=lpixbox)
			kmap = enmap.read_map(idir + "/kmap.fits", pixbox=lpixbox)
			with h5py.File(idir + "/info.hdf", "r") as hfile: mjd = hfile["mjd"][()]
			if args.mask:
				mask = 1-enmap.read_map(args.mask, geometry=frhs.geometry)
				frhs *= mask
				kmap *= mask
			# Scale if necessary
			frhs /= args.fscale    * args.escale**2
			kmap /= args.fscale**2 * args.escale**2
			# Evaluate the S/N and flux at the candidate location for each
			klim = np.percentile(kmap,90)*0.1
			rval = frhs.at([dec,ra], order=1)
			kval = kmap.at([dec,ra], order=1)
			mjds[dind] = mjd
			if klim > 0:
				with utils.nowarn():
					flux  = rval/kval
					dflux = kval**-0.5
				fluxes[dind] = [flux,dflux]
				if args.verbose: print("%3d %4d/%d %3d %8.3f %8.3f %s" % (comm_inter.rank, cind+1, len(cands), comm_intra.rank, flux, dflux, os.path.basename(idir)))
			else:
				if args.verbose: print("%3d %4d/%d %3d %-17s %s" % (comm_inter.rank, cind+1, len(cands), comm_intra.rank, "skip", os.path.basename(idir)))
				continue
			# offset frhs and kmap
			earth_pos = -ephemeris.ephem_vec("Sun", mjd).T[0]
			dmjd      = mjd - mjd0
			off       = [vy*dmjd, vx*dmjd]
			print("off", np.array(off)/utils.arcmin)
			cython.displace_map(frhs, earth_pos, r, off, omap=frhs_tot)
			cython.displace_map(kmap, earth_pos, r, off, omap=kmap_tot)
			frhs_tot_plain += frhs
			kmap_tot_plain += kmap
		# And reduce
		frhs_tot = utils.allreduce(frhs_tot, comm_intra)
		kmap_tot = utils.allreduce(kmap_tot, comm_intra)
		frhs_tot_plain = utils.allreduce(frhs_tot_plain, comm_intra)
		kmap_tot_plain = utils.allreduce(kmap_tot_plain, comm_intra)
		fluxes   = utils.allreduce(fluxes,   comm_intra)
		mjds     = utils.allreduce(mjds,     comm_intra)
		# Remove our padding and solve
		frhs_tot = frhs_tot[pad:-pad,pad:-pad]
		kmap_tot = kmap_tot[pad:-pad,pad:-pad]
		frhs_tot_plain = frhs_tot_plain[pad:-pad,pad:-pad]
		kmap_tot_plain = kmap_tot_plain[pad:-pad,pad:-pad]
		klim     = np.percentile(kmap_tot,90)*0.1
		sigma    = frhs_tot*0
		fluxmap  = frhs_tot*0
		sigma_plain    = frhs_tot*0
		fluxmap_plain  = frhs_tot*0
		cython.solve(frhs_tot, kmap_tot,    sigma,   klim=klim)
		cython.solve(frhs_tot, kmap_tot**2, fluxmap, klim=klim**2)
		cython.solve(frhs_tot_plain, kmap_tot_plain,    sigma_plain,   klim=klim)
		cython.solve(frhs_tot_plain, kmap_tot_plain**2, fluxmap_plain, klim=klim**2)
		# Check the empirical S/N, and use it to normalize
		def calc_std_robust(sigma, nblock=10):
			sigma = sigma[sigma!=0]
			sigma = sigma[:sigma.size//nblock*nblock].reshape(nblock,-1)
			return np.median(np.std(sigma,1))
		norm   = calc_std_robust(sigma)
		sigma /= norm
		norm_plain   = calc_std_robust(sigma_plain)
		sigma_plain /= norm_plain
		# Then read off the values at the candidate position
		flux_tot  = fluxmap.at([dec,ra], order=1)
		sigma_tot = sigma.at([dec,ra], order=1)
		dflux_tot = flux_tot/sigma_tot
		# And output the result
		if comm_intra.rank == 0:
			if args.verbose: print("%3d %4d/%d %3d %8.3f %8.3f %8.3f total" % (comm_inter.rank, cind+1, len(cands), comm_intra.rank, sigma_tot, flux_tot, dflux_tot))
			utils.mkdir(odir)
			enmap.write_map(odir + "/sigma.fits",   sigma)
			enmap.write_map(odir + "/fluxmap.fits", fluxmap)
			enmap.write_map(odir + "/sigma_plain.fits",   sigma_plain)
			enmap.write_map(odir + "/fluxmap_plain.fits", fluxmap_plain)
			good = np.where(fluxes[:,1] > 0)[0]
			with open(odir + "/fluxes.txt", "w") as ofile:
				for gi in good:
					ofile.write("%16.6f %6.2f %8.3f %8.3f   %6.2f %8.3f %8.3f\n" % (
						mjds[gi], fluxes[gi,0]/fluxes[gi,1], fluxes[gi,0], fluxes[gi,1],
						sigma_tot, flux_tot, dflux_tot))

elif mode == "limit":
	# Given param_map_full, norm_full and a S/N limit, outputs the corresponding limiting flux
	parser = argparse.ArgumentParser()
	parser.add_argument("limit", help="dummy")
	parser.add_argument("param_map_full")
	parser.add_argument("norm_full")
	parser.add_argument("odir")
	parser.add_argument("-s", "--snlim",  type=float, default=5, help="Actually the z lim using the paper's terms")
	parser.add_argument("-f", "--fscale", type=float, default=1, help="Multiply fluxes by this factor")
	parser.add_argument("-e", "--escale", type=float, default=1, help="Multiply flux errors by this factor (in addition to that implied by fscale)")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument(      "--p9",      action="store_true")
	parser.add_argument("-B", "--bias-map", type=str, default=None)
	args = parser.parse_args()
	import numpy as np, os, glob
	from enlib import enmap, utils, cython, planet9, mpi

	ym   = utils.arcmin/utils.yr2days
	utils.mkdir(args.odir)

	# Read the inverse variance maps. We don't need the rest of param_map_full
	pmap       = enmap.read_map(args.param_map_full)
	ivar_full  = pmap[:,-1]
	ivar_full /= args.fscale**2 * args.escale**2

	if args.bias_map:
		bias_map = enmap.read_map(args.bias_map)
		for ri in range(len(ivar_full)):
			if ri < bias_map.shape[0]: bmap = bias_map[ri]
			else:                      bmap = bias_map[-1]*0.1
			ivar_full[ri] *= bmap.project(ivar_full.shape, ivar_full.wcs, order=1, mode="wrap")**2
	dists      = np.max(pmap[:,0],(-2,-1))
	# Dummy dists in case we read in a file with no valid data. Prevents a warning later.
	if np.all(dists == 0): dists = 1+np.arange(len(dists))
	del pmap
	if args.p9:
		pos = ivar_full.posmap()
		ivar_full *= planet9.p9_like(pos[1], pos[0], posonly=True)
		del pos
	mask = ivar_full[0] !=0
	#print("dflux", np.median(ivar_full[:,mask])**-0.5)
	# except the distance, so get that too

	#print("dists", dists)
	# Use mu, sigma to untransform snlim
	mu, sigma = np.moveaxis(enmap.read_map(args.norm_full),1,0)
	#print("mu", np.median(mu[:,mask],1))
	#print("sigma", np.median(sigma[:,mask],1))
	snlim_raw = args.snlim*sigma + mu
	#print("snlim_raw", np.median(snlim_raw[:,mask],1))
	# And use it to get the flux limit as a function of distance
	with utils.nowarn():
		fluxlim = ivar_full**-0.5 * snlim_raw
		fluxlim[~np.isfinite(fluxlim)] = 0
	#print("fluxlim", np.median(fluxlim[:,mask],1))
	enmap.write_map(args.odir + "/fluxlim.fits", fluxlim)
	# Also get the distance limit for different planet 9 sizes.
	# Linder and Mordasini (2016) estimate that
	# * A 10 Me P9 would have a radius of 3.70 Re and a temperature of 47 K
	# * A  5 Me P9 would have a radius of 2.92 Re and a temperature of 40 K
	# and that the spectrum would be a featureless blackbody at our frequencies
	# (unlike Neptune, which has enhanced emission there).
	# The flux is F = blackbody(freq,T) * disk_solid_angle
	#               = blackbody(freq,T) * pi*R**2/r**2
	# For a reference distance of 500 AU we get:
	# * 10 Me: F0 = 9.40 mJy
	# *  5 Me: F0 = 4.92 mJy
	# F(r) = F0 * (r/500au)**-2 => r = 500au * (F/F0)**-0.5
	# and hence rlim = 500au * (flim/F0)**-0.5. We handle the more complicated flim(r) case
	# in a separate function.
	# Update: Using numbers based on Fortney middle scenario instead.
	rref = 500
	for name, fref in [("M10", 8.47), ("M05", 5.28)]:
		rlim = planet9.calc_rlim(fluxlim, dists, fref, rref)
		#print("rlim", name, np.median(rlim[mask]))
		enmap.write_map("%s/rlim_%s.fits" % (args.odir, name), rlim)

elif mode == "finish_find":
	"""Redo the last, fast steps of the find operation"""
	parser = argparse.ArgumentParser()
	parser.add_argument("finish_find", help="dummy")
	parser.add_argument("param_map_full")
	parser.add_argument("odir")
	parser.add_argument("-s", "--snlim",  type=float, default=3.5)
	parser.add_argument(      "--pad",    type=int,   default=300)
	parser.add_argument("-m", "--mask",   type=str,   default=None)
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-f", "--fscale", type=float, default=1, help="Multiply fluxes by this factor")
	parser.add_argument("-e", "--escale", type=float, default=1, help="Multiply errors by this factor, in addition to that implied by fscale")
	parser.add_argument(      "--no-norm", action="store_true")
	parser.add_argument(      "--norm-dir",type=str, default=None)
	parser.add_argument("-c", "--cont",    action="store_true")
	parser.add_argument("-B", "--bias-map", type=str, default=None, help="Map of oflux/iflux (so 1 for no loss of signal) as a function of r bin")
	parser.add_argument(      "--only",   type=str, default=None)
	args = parser.parse_args()
	import numpy as np, os
	from enlib import enmap, utils, cython, planet9, retile, mpi

	dtype = np.float32
	comm  = mpi.COMM_WORLD
	ym    = utils.arcmin/utils.yr2days
	pad   = args.pad
	single= os.path.isfile(args.param_map_full)
	if args.verbose: print("Reading %s" % (args.param_map))

	if not single:
		for name in ["param_map", "sigma_eff", "sigma_eff_full", "norm_full", "cands"]:
			utils.mkdir(args.odir + "/" + name)

	if args.bias_map:
		bias_map = enmap.read_map(args.bias_map)

	only = [int(w) for w in args.only.split(",")] if args.only else None

	def process(params_full, namefun, tile=None):
		# First loop through each radial bin, compute its S/N, and the corresponding spatial distribution
		# Store the total mu, sigma maps here, so we can quickly construct flux limit maps later. The detection
		# limit will be (snlim*sigma+mu) * kmap**-0.5
		norm_full = enmap.zeros((len(params_full),2)+params_full.shape[-2:], params_full.wcs, params_full.dtype)
		sigma_full= enmap.zeros((len(params_full),)+params_full.shape[-2:], params_full.wcs, params_full.dtype)
		param_max = params_full[0]*0
		sigma_max = params_full[0,0]*0-np.inf

		if args.norm_dir:
			if os.path.isfile(args.norm_dir):
				norm_full = enmap.read_map(args.norm_dir, geometry=norm_full.geometry)
			else:
				norm_full = retile.read_geometry(args.norm_dir + "/tile%(y)03d_%(x)03d.fits", norm_full.shape, norm_full.wcs, verbose=True)
			print("norm", np.median(norm_full,(-2,-1)))

		for ri, params in enumerate(params_full):
			# Build S/N map
			frhs, kmap = params[-2:]
			# Debias if necessary
			if args.bias_map:
				# This is a bit more complicated than it ideally would be because I forgot to
				# include the infinite-distance element in my sims. So for now I will just use
				# the largest element available with an additional big penalty. Sadly this means
				# that the inf element becomes pointless - it can't absorb non-moving stuff any more.
				if ri < bias_map.shape[0]: bmap = bias_map[ri]
				else:                      bmap = bias_map[-1]*0.1
				bias_correction = bmap.project(frhs.shape, frhs.wcs, order=1, mode="wrap")
				frhs *= bias_correction
				kmap *= bias_correction**2
			# Scale if necessary
			frhs /= args.fscale    * args.escale**2
			kmap /= args.fscale**2 * args.escale**2
			klim  = np.float32(np.percentile(kmap,99)*1e-3)
			if klim > 0:
				klim  = np.float32(np.percentile(kmap[kmap>klim],90)*1e-2)
			snmap = cython.solve(frhs, kmap, klim=klim)
			mask  = kmap > klim # true = good
			if args.mask:
				# mask on disk has opposite convention
				mask &= enmap.read_map(args.mask, geometry=params.geometry) < 0.5
			# Grow the mask uses opposite masking convention
			mask  = ~planet9.grow_mask(~mask, 20)
			# Normalize spatially
			if args.no_norm:
				pass
			elif args.norm_dir:
				mu, sigma = norm_full[ri]
				with utils.nowarn():
					snmap = (snmap-mu)/sigma * mask
					snmap[~np.isfinite(snmap)] = 0
			else:
				spat_mu, spat_sigma = planet9.build_mu_sigma_map(snmap, mask=mask)
				snmap = (snmap - spat_mu)/spat_sigma * mask
				# Normalize survival function (or at least the high-S/N tail part of it)
				dist_mu, dist_sigma = planet9.find_sf_correction(snmap, mask=mask)
				snmap  = (snmap - dist_mu)/dist_sigma * mask
				# Update total normalization record. We have done ((sn-spat_mu)/spat_sigma-dist_mu)/dist_sigma,
				# so in total we have: (sn - (spat_mu+dist_mu*spat_sigma))/(spat_sigma*dist_sigma)
				norm_full[ri,0] = spat_sigma*dist_mu + spat_mu
				norm_full[ri,1] = spat_sigma*dist_sigma

			sigma_full[ri]  = snmap
			# Merge into overall S/N
			cython.merge_param_maps(params, snmap, param_max, sigma_max)

		# Ok, we now have the merged, normalized S/N map. Use these to find
		# candidates.
		cands = planet9.find_candidates(sigma_max, param_max, snmin=args.snlim, pad=pad)
		np.savetxt(namefun("cands.txt"), np.array([cands[:,1]/utils.degree, cands[:,0]/utils.degree,
				cands[:,2], cands[:,3], cands[:,4], cands[:,5], cands[:,6]/ym, cands[:,7]/ym, cands[:,8]]).T, fmt="%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %4.2f")
		# Output the merged maps too
		enmap.write_map(namefun("param_map.fits"), planet9.unpad(param_max, pad))
		enmap.write_map(namefun("sigma_eff.fits"), planet9.unpad(sigma_max, pad))
		enmap.write_map(namefun("norm_full.fits"), planet9.unpad(norm_full, pad))
		enmap.write_map(namefun("sigma_eff_full.fits"), planet9.unpad(sigma_full, pad))
		if tile is not None:
			ty, tx = tile
			print("%4d processed tile %3d %3d. Found %5d candidates." % (comm.rank, ty, tx, len(cands)))

	if single:
		params_full = enmap.read_map(args.param_map_full)
		params_full[:,1:3] *= ym
		def oname(name): return os.path.join(args.odir, name)
		process(params_full, oname)
	else:
		# The tile stuff is ugly and should be redesigned
		tpath = args.param_map_full + "/tile%(y)03d_%(x)03d.fits"
		(ty1, tx1), (ty2, tx2) = retile.find_tile_range(tpath)
		tyxs = [(ty,tx) for ty in range(ty1,ty2) for tx in range(tx1,tx2)]
		for ind in range(comm.rank, len(tyxs), comm.size):
			ty, tx = tyxs[ind]
			if only and not (ty == only[0] and tx == only[1]): continue
			def oname(name):
				fname, fext = os.path.splitext(name)
				return "%s/%s/tile%03d_%03d%s" % (args.odir, fname, ty, tx, fext)
			if args.cont and os.path.isfile(oname("sigma_eff_full.fits")): continue
			params_full = retile.read_retile(tpath, (ty,tx), margin=pad).astype(dtype)
			# params_full is [nr,{r,vy,vx,frhs,kmap},y,x]. But go back to rads/s
			params_full[:,1:3] *= ym
			process(params_full, oname, tile=(ty,tx))
		comm.Barrier()
	if comm.rank == 0:
		print("Done")

# Below is a note rescued from the old version of limit
#
# Ideally we have
#  flux  = frhs/kmap
#  dflux = kmap**-0.5
#  sigma = flux/dflux
# However, there are some issues.
#
# 1. The recovered flux is lower than the injeted flux, by a factor of about 0.8 for
# full-res analysis and about 0.6 for half-res analysis. This might just be due to
# the pixel window, interpolation and limited resolution in the parameter search.
# Can check this with a more detailed fit for each candidate.
#
# 2. The scatter in recovered fluxes is OK before correcting for the factor above, so
# dflux is also wrong by about the same factor. To be precise, the scatter is OK in
# nose dominated regime. In strongly signal dominated regime there appears to be some
# position-dependent bias that increases the scatter.
#
# 3. The plain sigma doesn't take into account the effect of the parameter search.
# We can use planet9.build_mu_sigma_map to infer sigma_eff. But see #4
#
# 4. Ultimately, our flux limit is set by how many false positives we are willing to consider.
# I had 100 candidates above 6.6 sigma_eff (which would be 6.0 after dividing by 1.1).
# This is more than one should expect for a gaussian field. Even if each pixel were
# independent, which they aren't, I would only expect 0.2 6 sigma detections by chance.
# More quantitatively, I have 22 hits in a 0.33176 wide bin centered at 5.46. For a
# properly normalized gaussien field, a 5.46 sigma fluctuation should have a 4.6e-8
# chance of happening per draw, implying 5e8 independent samples in the map, which
# is about 20x too high. So the normalization isn't right.
#
# Let's say that we accept U false positives in our search. Hence, for each of our N
# independent spots in the map we want to know at what flux threshold there is a
# U/N chance of having a false detection. That level is our detection limit.
# For homogeneous search space (e.g. equal relative weight form each time period
# everywhere in the map) we could do this simply by computing sigma_map and
# finding the level at which we have U candidates (excluding clear outliers).
# Then our flux limit is simply dflux*sigma_lim (assuming dflux is correct, of course)
#
# If we can't assume homogeneity of sigma_lim, then we need to find a position-dependent
# version, and that's difficult. Since U/N is so small we can't expect to have any
# candidates inside any reasonably-sized region. One therefore has to extrapolate from
# lower threshold, and that requires knowing something about the distribution.
# The input field is pretty gaussian, but during the search we displace and combine them,
# taking the maximum across many parameter values. Some input maps move more than others
# during this process, making it hard to estimate how many independent gaussian numbers
# were combined and maxed in the process.
#
# Empirically the distribution looks like a bit like two gaussians glued together, with
# the upper part being steeper than the lower part. What build_mu_sigma_map does is to fit
# parameters such that sigma_eff = (sigma_map - b)/a. One can then find the sigma_eff_lim
# that gives U candidates, resulting in sigma_map_lim = sigma_eff_lim*a+b, and finally
# flux_lim = dflux * (sigma_eff_lim*a+b).
# This is already how I find the candidates. So I just need to compute dflux, sigma_map
# and the coefficients.
#
# This doesn't need to actually reflect the real distribution of the values.
# I spent a a day or two trying to model the true distribution, but it's
# tricky and not really worth the effort. Instead of sigma_eff being the
# quantiles of the true distribution, we should just think of them as being
# some arbitrary detection statistic, and the transformation with b,a above is just there
# to make it more homogeneous than the plain sigma_map is.
