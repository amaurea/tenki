"""Measure pointing offsets given depth-1 maps and an input
catalog. This program takes the simple approach of requiring
each source to be indivdually detectable. This lets us get
the postions without any non-linear search, and gives us
per-source measurements, but has the cost of not getting
pointing measurements for tods with no bright sources. For
SO f090 all-tube maps, this should not be a problem though.
I can write fancier programs later if necessary."""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("cat")
parser.add_argument("rhofiles", nargs="+")
parser.add_argument("ofile")
parser.add_argument("-s", "--snlim",   type=float, default=5)
parser.add_argument("-v", "--verbose", default=1, action="count")
parser.add_argument("-q", "--quiet",   default=0, action="count")
parser.add_argument("-b", "--bsize",   type=float, default=2.1, help="Beam FWHM in arcmin. Used to estimate position accuracy")
parser.add_argument(      "--noinfo",  action="store_true")
args = parser.parse_args()
import numpy as np, os
from pixell import enmap, utils, mpi, bunch, colors
from astropy.io import fits
from scipy import ndimage
from enlib import coordinates

def read_pointing_catalog(fname):
	if fname.endswith(".fits"):
		d = fits.open(fname)[1].data
		return np.array([d.ref_ra*utils.degree, d.ref_dec*utils.degree,
			d.base_flux, d.base_dflux]).T
	else:
		cat = np.loadtxt(fname, usecols=(0,1,2,3))
		cat[:,:2] *= utils.degree
		return cat

def read_tmap(fname):
	tmap   = enmap.read_map(fname)
	header = enmap.read_fits_header(fname)
	tref   = float(header["TREF"])
	return tmap, tref

def find_srcs(rho, kappa, snlim=5, snlim0=2):
	rho  = rho.preflat[0]
	kappa= kappa.preflat[0]
	snr  = rho/kappa**0.5
	labels, nlabel = ndimage.label(snr > snlim0)
	iall = np.arange(1,nlabel+1)
	# measure center-of-mass position. This is a good estimate for the
	# point source position, but not infallible
	pixs = np.array(ndimage.center_of_mass(snr, labels, iall)).T
	if pixs.size == 0: return np.zeros((0,4))
	poss = rho.pix2sky(pixs)
	# measure the flux
	rho_vals   = rho.at(poss)
	kappa_vals = kappa.at(poss)
	snr_vals = rho_vals/kappa_vals**0.5
	flux  = rho_vals/kappa_vals
	dflux = kappa_vals**-0.5
	# make sure the peak is actually in our region, to avoid ring artifacts
	ipixs = utils.nint(pixs)
	peak_label = labels[ipixs[0],ipixs[1]]
	good  = (peak_label == iall)&(snr_vals>=snlim)
	# construct output catalog
	cat = np.concatenate([poss[::-1,good], flux[None,good], dflux[None,good]]).T
	return cat

def cat2fcart(cat, rscale=1, fscale=1):
	coords       = np.zeros((len(cat),4))
	coords[:,:3] = utils.ang2rect(cat.T[:2]).T/rscale
	coords[:,3]  = np.log(cat.T[2])/fscale
	return coords

def match_srcs(cat, refcat, rmax=4*utils.arcmin, variability=5):
	"""Match entries in cat with refcat, up to a maximum angular
	distance of rmax, and maximum log-flux distance of log(variability),
	returning matching pairs and distances"""
	# Build 4D pos-flux coordinates
	coords    = cat2fcart(cat,    rscale=rmax, fscale=np.log(variability))
	refcoords = cat2fcart(refcat, rscale=rmax, fscale=np.log(variability))
	matches   = utils.crossmatch(coords, refcoords, mode="closest", rmax=1)
	return np.array(matches, dtype=int).T.reshape(2,-1)

def format_row(row, refcat, rhofiles):
	ind, refi, dra, ddec, daz, del_, t, ra, dec, flux, dflux, az, el, obs_az, obs_waz, obs_el, obs_roll = row
	ind, refi = int(ind), int(refi)
	snr    = flux/dflux
	posacc = args.bsize/snr
	ref_ra, ref_dec, ref_flux = refcat[refi,:3]
	base   = os.path.basename(utils.replace(rhofiles[ind], "_rho.fits", ""))
	# Let's start with what we observe directly, then supporting information later.
	# Part 1. Raw observations: t, ra, dec, snr, flux, Δra+err, Δdec+err
	msg  = "%10.0f  %8.3f %7.3f %7.2f %7.1f %7.1f  %6.3f %6.3f %6.3f %6.3f" % (
			t, ra/utils.degree, dec/utils.degree, snr, flux, dflux,
			dra/utils.arcmin, posacc, ddec/utils.arcmin, posacc)
	# Part 2. Horizontal version: az, el, Δaz+err, Δel+err
	msg += "  %8.3f %7.3f  %6.3f %6.3f %6.3f %6.3f" % (
			az/utils.degree, el/utils.degree,
			daz/utils.arcmin, posacc, del_/utils.arcmin, posacc)
	# Part 3. Boresight: baz waz bel roll
	msg += "  %7.2f %7.2f %6.2f %7.2f" % (
			obs_az/utils.degree, obs_waz/utils.degree,
			obs_el/utils.degree, obs_roll/utils.degree)
	# Part 4. Reference: ref_id, ref_ra, ref_dec, ref_flux
	msg += "  %5d %8.3f %7.3f %7.1f" % (
			refi, ref_ra/utils.degree, ref_dec/utils.degree, ref_flux)
	# Part 5: name
	msg += "  %s" % base
	return msg

def normalize_hor(hor, bel):
	if bel > np.pi/2:
		hor[:,0] += np.pi
		hor[:,1]  = np.pi-hor[:,1]
	hor[:,0] = utils.rewind(hor[:,0])
	return hor

comm     = mpi.COMM_WORLD
verbosity= args.verbose - args.quiet
refcat   = read_pointing_catalog(args.cat) # [{ra,dec,flux,dflux},nsrc]
rhofiles = sum([sorted(utils.glob(fname)) for fname in args.rhofiles],[])
nfile    = len(rhofiles)

# collect output here
output = []

for ind in range(comm.rank, nfile, comm.size):
	base  = utils.replace(rhofiles[ind], "_rho.fits", "")
	if args.verbose >= 1:
		print("%s%4d %4d/%d Processing %s%s" % (colors.lgreen, comm.rank, ind+1, nfile, base, colors.reset))
	rho   = enmap.read_map(base + "_rho.fits")
	kappa = enmap.read_map(base + "_kappa.fits")
	kappa = np.maximum(kappa, np.max(kappa)*0.01)
	tmap, tref = read_tmap(base + "_time.fits")
	if not args.noinfo:
		info  = bunch.read("_".join(base.split("_")[:-2]) + "_info.hdf")
		# We need the commanded elevation, since this can be >90°, which
		# we can't recover from our coordinate transform. Also nice to have roll
		obs_az   = info.obstab[0]["az"]*utils.degree
		obs_waz  = info.obstab[0]["waz"]*utils.degree
		obs_el   = info.obstab[0]["el"]*utils.degree
		obs_roll = info.obstab[0]["roll"]*utils.degree
	else:
		obs_az   = 0
		obs_waz  = 0
		obs_el   = 0
		obs_roll = 0

	# Find srcs and sort by snr
	mycat = find_srcs(rho, kappa, snlim=args.snlim) # [nsrc,{ra,dec,flux,dflux}]
	mycat = mycat[np.argsort(mycat[:,2]/mycat[:,3])[::-1]]
	if len(mycat) == 0 and args.verbose >= 1:
		print("%4d %4d/%s No objects found in %s" % (comm.rank, ind+1, nfile, base))
		continue

	# Crossmatch with our input catalog
	myinds, refinds = match_srcs(mycat, refcat)
	if len(myinds) == 0 and args.verbose >= 1:
		print("%4d %4d/%s None of %d detected objects matched in %s" % (comm.rank, ind+1, nfile, len(mycat), base))
		continue
	nmatched = len(myinds)
	# Get the time each object was hit. Uses observed coordinates
	times = tmap.at(mycat.T[1::-1,myinds], order=1).astype(np.float64)+tref

	# Restrict to valid times. This is probably not necessary, since
	# we shouldn't detect objects if there's a hole in the map anyway
	tgood = times>=tref
	myinds, refinds, times = myinds[tgood], refinds[tgood], times[tgood]
	if len(myinds) == 0 and args.verbose >= 1:
		print("%4d %4d/%s None of %d matches had valid times in %s" % (comm.rank, ind+1, nfile, nmatched, base))
		continue

	# Transform to horizontal coordinates
	refhor = coordinates.transform("cel", "hor", refcat.T[:2,refinds], time=utils.ctime2mjd(times)).T
	myhor  = coordinates.transform("cel", "hor",  mycat.T[:2, myinds], time=utils.ctime2mjd(times)).T
	refhor = normalize_hor(refhor, obs_el)
	myhor  = normalize_hor(myhor,  obs_el)

	# Calculate the pointing offsets
	dpos   = utils.rewind(mycat[myinds,:2]-refcat[refinds,:2])
	dhor   = utils.rewind(myhor[:,:2]-refhor[:,:2])

	# Accumulate each source in output. Do it in a plain array to make
	# mpi simpler. Will reformat in the end
	for i, (myi, refi) in enumerate(zip(myinds, refinds)):
		output.append([ind, refi, dpos[i,0], dpos[i,1], dhor[i,0], dhor[i,1],
			times[i], mycat[myi,0], mycat[myi,1], mycat[myi,2], mycat[myi,3], myhor[i,0], myhor[i,1], obs_az, obs_waz, obs_el, obs_roll])
		if args.verbose >= 2:
			print(format_row(output[-1], refcat, rhofiles))

if comm.rank == 0:
	print("%sReducing%s" % (colors.lgreen, colors.reset))

output = np.array(output)
# Reshape to handle empty case
output = utils.allgatherv(output, comm).reshape(-1,17)
# Sort by time
output = output[np.argsort(output[:,6])]

if comm.rank == 0:
	print("%sWriting %s%s" % (colors.lgreen, args.ofile, colors.reset))
	with open(args.ofile, "w") as ofile:
		for row in output:
			ofile.write(format_row(row, refcat, rhofiles) + "\n")
