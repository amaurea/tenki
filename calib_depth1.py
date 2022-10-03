import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("planck1")
parser.add_argument("planck2")
parser.add_argument("depth1_maps", nargs="+")
parser.add_argument("odir")
parser.add_argument("--pbeam", type=str, default=None)
parser.add_argument("--abeam", type=str, default=None)
parser.add_argument("--lmin",  type=int, default=1000)
parser.add_argument("--lmax",  type=int, default=2000)
parser.add_argument("--mask",  type=str, default=None)
parser.add_argument("--lknee", type=float, default=2000)
parser.add_argument("--alpha", type=float, default=-3)
parser.add_argument("--apod",  type=float, default=0.5)
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils, curvedsky, mpi, uharm

def read_beam(fname, lmax=None):
	beam = np.loadtxt(fname, usecols=(1,))
	nl   = lmax+1 if lmax is not None else None
	beam = utils.regularize_beam(beam, nl=nl)
	beam /= np.max(beam)
	return beam

def downsample(imap, factor, ref=[0,0]):
	return enmap.downgrade(imap, factor, ref=ref)

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

# Calculate target resolution. We will downsample to roughly
# this resolution to make things faster and save memory
targ_res = 2*np.pi/args.lmax/2
# Get the current resolution. We assume that all maps match this
pshape, pwcs = enmap.read_map_geometry(args.planck1)
res      = np.max(np.abs(pwcs.wcs.cdelt))*utils.degree
# Use this to compute the downgrade factor
down     = max(1,utils.floor(targ_res/res))
apod_rad = args.apod*utils.degree

# Read the beams
pbeam = None if args.pbeam is None else read_beam(args.pbeam, lmax=args.lmax)
abeam = None if args.abeam is None else read_beam(args.abeam, lmax=args.lmax)
# Read the planck maps. Should be compatible with depth-1 maps
planck1 = downsample(enmap.read_map(args.planck1, sel=(0,)).astype(np.float32, copy=False), down)
planck2 = downsample(enmap.read_map(args.planck2, sel=(0,)).astype(np.float32, copy=False), down)
# If we have a mask, read it too
mask = None if args.mask is None else downsample(enmap.read_map(args.mask).preflat[0], down)>0

utils.mkdir(args.odir)
amps  = np.zeros(nmap)
damps = np.zeros(nmap)
times = np.zeros(nmap)

for ind in range(comm.rank, nmap, comm.size):
	mapfile  = mapfiles[ind]
	t        = int(os.path.basename(mapfile).split("_")[1])
	# Read in full-resolution maps
	imap_raw = enmap.read_map(mapfile, sel=(0,))
	ivar_raw = enmap.read_map(utils.replace(mapfile, "map.fits", "ivar.fits")).preflat[0]
	# Downgrade with inverse variance weighting
	ivar     = downsample(ivar_raw, down)
	with utils.nowarn():
		imap   = downsample(imap_raw*ivar_raw, down)/ivar
		utils.remove_nan(imap)
	# Deconvolve pixel window, and clean up the outside area for
	# plotting purposes
	enmap.apply_window(imap, -1)
	imap[ivar==0]=0
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
	# Build our noise model. Do not want to build this
	# from the data itself to avoid bias. So will just hardcode something
	uht   = uharm.UHT(*imap.geometry, mode="curved", lmax=args.lmax, tweak=True)
	with utils.nowarn():
		iC  = (1+(uht.l/args.lknee)**args.alpha)**-1
	# Apply multipole cut
	iC[uht.l<args.lmin] = 0
	iC[uht.l>args.lmax] = 0
	# And actually perform the fit
	amp, damp = fit_noisy_template_constcorr(psub1, psub2, imap, ivar, iC, tbeam=pbeam, mbeam=abeam, uht=uht, debug=args.odir + "/" + os.path.basename(mapfile)[:-5])
	amps [ind] = amp
	damps[ind] = damp
	times[ind] = t
	print("%10.0f %8.3f %8.3f %s" % (t, amp, damp, os.path.basename(mapfile)))

amps  = utils.allreduce(amps,  comm)
damps = utils.allreduce(damps, comm)
times = utils.allreduce(times, comm)
if comm.rank == 0:
	with open(args.odir + "/fits.txt", "w") as ofile:
		for i, (mapfile, t, amp, damp) in enumerate(zip(mapfiles, times, amps, damps)):
			ofile.write("%10.0f %8.3f %8.3f %s\n" % (t, amp, damp, os.path.basename(mapfile)))
