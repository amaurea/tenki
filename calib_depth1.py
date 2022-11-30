import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("planck1")
parser.add_argument("planck2")
parser.add_argument("depth1_maps", nargs="+")
parser.add_argument("ofile")
parser.add_argument("--pbeam", type=str, default=None)
parser.add_argument("--abeam", type=str, default=None)
parser.add_argument("--lmin",  type=int, default=1000)
parser.add_argument("--lmax",  type=int, default=2000)
parser.add_argument("--mask",  type=str, default=None)
parser.add_argument("--lknee", type=float, default=2000)
parser.add_argument("--alpha", type=float, default=-3)
parser.add_argument("--apod",  type=float, default=0.25)
parser.add_argument("--debug", type=str,   default=None)
parser.add_argument("--cache", type=str,   default=None)
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils, curvedsky, mpi, uharm, bunch

def read_beam(fname, lmax=None):
	beam = np.loadtxt(fname, usecols=(1,))
	nl   = lmax+1 if lmax is not None else None
	beam = utils.regularize_beam(beam, nl=nl)
	beam /= np.max(beam)
	return beam

def whole_ratio(r, n):
	# Really stupid solution, but not bottleneck
	nsegs = np.arange(1,n)
	nsegs = nsegs[n%nsegs==0]
	score = (n/nsegs-r)**2
	nseg  = nsegs[np.argmin(score)]
	ri    = max(1, n//nseg)
	return ri

def downsample(imap, factor, ref=[0,0]):
	return enmap.downgrade(imap, factor, ref=ref)

def highpass(imap, lmin):
	"""Hard highpass of map imap, cutting off modes below lmin"""
	alm = curvedsky.map2alm(imap, lmax=lmin, tweak=True)
	sub = curvedsky.alm2map(alm, imap*0, tweak=True)
	return imap-sub

def build_noise_model(imap, ivar, lout, bsize=200):
	"""Build a simple noise model from imap, assuming it's nosie-dominated.
	We don't want to measure too much as it will bias us, so we fit a small
	number of degrees of freedom."""
	wmap    = imap * ivar**0.5
	ps2d    = np.abs(enmap.fft(wmap))**2 / np.mean(ivar>0)
	ps1d, l = ps2d.lbin(bsize=bsize)
	C       = np.interp(lout, l, ps1d)
	return C

def get_maps(mapfile, down, cache=None):
	if cache is not None:
		cmapfile = cache + "/" + os.path.basename(mapfile)
		try:
			map  = enmap.read_map(cmapfile)
			ivar = enmap.read_map(utils.replace(cmapfile, "map.fits", "ivar.fits"))
			return map, ivar
		except (IOError, FileNotFoundError): pass
	imap_raw = enmap.read_map(mapfile, sel=(0,))
	ivar_raw = enmap.read_map(utils.replace(mapfile, "map.fits", "ivar.fits")).preflat[0]
	# Downgrade with inverse variance weighting
	ivar     = downsample(ivar_raw, down)
	with utils.nowarn():
		imap   = downsample(imap_raw*ivar_raw, down)/ivar
		utils.remove_nan(imap)
	if not allow_day:
		tmap_raw = enmap.read_map(utils.replace(mapfile, "map.fits", "time.fits"))
		info     = bunch.read(utils.replace(mapfile, "map.fits", "info.hdf"))
		with utils.nowarn():
			tmap   = downsample(tmap_raw*ivar_raw, down)/ivar
			del tmap_raw
			utils.remove_nan(tmap)
		hour_map = ((tmap.astype(np.float64)+info.t)/3600)%24
		is_day = (hour_map > 11)&(hour_map<23)
		ivar  *= ~is_day
	del imap_raw, ivar_raw
	ivar *= down**2
	if cache is not None:
		cmapfile = cache + "/" + os.path.basename(mapfile)
		enmap.write_map(cmapfile, imap)
		enmap.write_map(utils.replace(cmapfile, "map.fits", "ivar.fits"), ivar)
	return imap, ivar

def fit_noisy_template_constcorr(template1, template2, imap, ivar, iC, tbeam=None, mbeam=None, uht=None, debug=None):
	"""Perform the fit amp = sum(template*iN*map)/sum(template1*iN*template2)
	where iN = sqrt(iC)*ivar*sqrt(iC) and template = 0.5*(template1+template2).
	Two realizations of the template are taken in to remove noise bias in the fit.
	The templates should still be low-noise, though, as their noise is not propagated
	into the errors.

	The optional arguments tbeam and mbeam specify the beams of the templates and map
	respectively, with shape [nl]"""
	if uht is None: uht = uharm.UHT(*imap.geometry)
	# Go to alms
	talm1 = uht.map2harm(template1)
	talm2 = uht.map2harm(template2)
	malm  = uht.map2harm(imap)
	# Convolve to common beam
	if mbeam is not None:
		uht.hmul(mbeam, talm1, inplace=True)
		uht.hmul(mbeam, talm2, inplace=True)
	if tbeam is not None:
		uht.hmul(tbeam, malm, inplace=True)
		iC *= tbeam**-2
	if debug:
		enmap.write_map(debug + "_t.fits", (template1+template2)/2)
		enmap.write_map(debug + "_m.fits", imap)
		enmap.write_map(debug + "_t_reconv.fits", uht.harm2map((talm1+talm2)/2))
		enmap.write_map(debug + "_m_reconv.fits", uht.harm2map(malm))
	# Do the fit
	hC   = iC**0.5
	t1hC = uht.harm2map(uht.hmul(hC,talm1))
	t2hC = uht.harm2map(uht.hmul(hC,talm2))
	thC  = 0.5*(t1hC+t2hC)
	mhC  = uht.harm2map(uht.hmul(hC,malm))
	if debug:
		enmap.write_map(debug + "_thCw.fits", ivar**0.5*thC)
		enmap.write_map(debug + "_mhCw.fits", ivar**0.5*mhC)
	rhs  = np.sum(thC*ivar*mhC)
	div  = np.sum(t1hC*ivar*t2hC)
	with utils.nowarn():
		amp  = utils.without_nan(rhs/div)
		damp = div**-0.5
	return amp, damp

comm     = mpi.COMM_WORLD
mapfiles = sum([sorted(utils.glob(fname)) for fname in args.depth1_maps],[])
nmap     = len(mapfiles)
allow_day= False

# Calculate target resolution. We will downsample to roughly
# this resolution to make things faster and save memory. This will induce
# pixel window, but it should cancel overall
targ_res = 2*np.pi/args.lmax/2 / 2
# Get the current resolution. We assume that all maps match this
pshape, pwcs = enmap.read_map_geometry(args.planck1)
res      = np.max(np.abs(pwcs.wcs.cdelt))*utils.degree
# Use this to compute the downgrade factor
down     = max(1,whole_ratio(targ_res/res,utils.nint(2*np.pi/res)))
apod_rad = args.apod*utils.degree

# Read the beams
pbeam = None if args.pbeam is None else read_beam(args.pbeam, lmax=args.lmax)
abeam = None if args.abeam is None else read_beam(args.abeam, lmax=args.lmax)
# Read the planck maps. A bit messy due to caching
planck_read = False
if args.cache:
	try:
		planck1 = enmap.read_map(args.cache + "/" + os.path.basename(args.planck1))
		planck2 = enmap.read_map(args.cache + "/" + os.path.basename(args.planck2))
		mask = None if args.mask is None else enmap.read_map(args.cache + "/" + os.path.basename(args.mask))>0
		planck_read = True
	except (IOError, FileNotFoundError) as e: pass
comm.Barrier()
if not planck_read:
	# Read the planck maps. Should be compatible with depth-1 maps
	planck1 = downsample(enmap.read_map(args.planck1, sel=(0,)).astype(np.float32, copy=False), down)
	planck2 = downsample(enmap.read_map(args.planck2, sel=(0,)).astype(np.float32, copy=False), down)
	# If we have a mask, read it too
	mask = None if args.mask is None else downsample(enmap.read_map(args.mask).preflat[0], down)>0
	if args.cache and comm.rank == 0:
		enmap.write_map(args.cache + "/" + os.path.basename(args.planck1), planck1)
		enmap.write_map(args.cache + "/" + os.path.basename(args.planck2), planck2)
		if args.mask is not None:
			enmap.write_map(args.cache + "/" + os.path.basename(args.mask), mask.astype(np.uint8))

# Apply a gentle highpass to the planck maps to reduce edge ringing
planck1 = highpass(planck1, args.lmin//2)
planck2 = highpass(planck2, args.lmin//2)

utils.mkdir(os.path.dirname(args.ofile))
if args.debug: utils.mkdir(args.debug)
amps  = np.zeros(nmap)
damps = np.zeros(nmap)
times = np.zeros(nmap)

for ind in range(comm.rank, nmap, comm.size):
	mapfile  = mapfiles[ind]
	t        = int(os.path.basename(mapfile).split("_")[1])
	imap,ivar= get_maps(mapfile, down, cache=args.cache)
	# Get the overlapping planck subsets
	psub1 = planck1.extract(*imap.geometry)
	psub2 = planck2.extract(*imap.geometry)
	# Use the mask to construct an apodization, which we apply to all input maps
	# We then mark all apodized regions as zero-weight in the ivar
	if mask is not None:
		masksub = mask.extract(*imap.geometry)
		ivar   *= 1-masksub
		apod    = enmap.apod_mask(ivar>0, apod_rad)
		#enmap.write_map("test_apod.fits", apod)
		for obj in [imap,psub1,psub2]: obj *= apod
		ivar   *= apod == 1
	# Skip empty
	if np.any(ivar>0):
		# Make the pixel windows consistent.
		# ACT    starts with win_full. After downgrade: win_down
		# Planck starts with no win (since it's in the beam). After: win_down/win_full
		# So we can make everything consistent by applying win_full to planck
		# This also has the advantage that we don't distort the noise in imap
		psub1 = enmap.apply_window(psub1, scale=down)
		psub2 = enmap.apply_window(psub2, scale=down)
		# Build our noise model. Do not want to build this
		# from the data itself to avoid bias. So will just hardcode something
		uht   = uharm.UHT(*imap.geometry, mode="curved", lmax=args.lmax, tweak=True)
		#with utils.nowarn():
		#	iC  = (1+(uht.l/args.lknee)**args.alpha)**-1
		iC    = 1/build_noise_model(imap, ivar, uht.l)
		# Apply multipole cut
		iC[uht.l<args.lmin] = 0
		iC[uht.l>args.lmax] = 0
		# And actually perform the fit
		debug = None if args.debug is None else args.debug + "/" + os.path.basename(mapfile)[:-5]
		amp, damp = fit_noisy_template_constcorr(psub1, psub2, imap, ivar, iC, tbeam=pbeam, mbeam=abeam, uht=uht, debug=debug)
	else:
		amp, damp = 0, np.inf
	amps [ind] = amp
	damps[ind] = damp
	times[ind] = t
	print("%10.0f %8.3f %8.3f %s" % (t, amp, damp, os.path.basename(mapfile)))

amps  = utils.allreduce(amps,  comm)
damps = utils.allreduce(damps, comm)
times = utils.allreduce(times, comm)
if comm.rank == 0:
	with open(args.ofile, "w") as ofile:
		for i, (mapfile, t, amp, damp) in enumerate(zip(mapfiles, times, amps, damps)):
			ofile.write("%10.0f %8.3f %8.3f %s\n" % (t, amp, damp, os.path.basename(mapfile)))
