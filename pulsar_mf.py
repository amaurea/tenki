import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imaps", nargs="+")
parser.add_argument("odir")
parser.add_argument("tag")
parser.add_argument("-b", "--beam", type=str,   default=1.4)
parser.add_argument("-f", "--freq", type=float, default=98)
#parser.add_argument("-R", "--rmask",type=float, default=2)
parser.add_argument(     "--noise-block-size", type=int, default=2)
args = parser.parse_args()
import numpy as np
from scipy import ndimage
from pixell import enmap, utils, uharm, analysis

def get_beam(fname_or_fwhm):
	try:
		sigma = float(fname_or_fwhm)*utils.arcmin*utils.fwhm
		l = np.arange(40000)
		return np.exp(-0.5*l**2*sigma**2)
	except ValueError:
		_, bl = np.loadtxt(fname_or_fwhm, usecols=(0,1), ndmin=2).T
		bl /= np.max(bl)
		return bl
def blockavg(imap, bsize):
	return enmap.upgrade(enmap.downgrade(imap, bsize, inclusive=True), bsize, oshape=imap.shape, inclusive=True)
def find_edge(mask):
	edge_filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
	return ndimage.convolve(mask.astype(int), edge_filter)<-1
def gapfill_edgemean(map, mask):
	edge = find_edge(mask)
	fill = np.mean(map[...,edge],-1)
	omap = map.copy()
	omap[...,mask] = fill[...,None]
	return omap
def calibrate_ivar(maps, ivars, rmask=2*utils.arcmin, bsize=2):
	# Correction factor locally across splits
	r    = maps.modrmap([0,0])
	mask = r >= rmask
	rhs  = blockavg(np.mean(maps**2*ivars*mask,0),bsize)
	div  = blockavg(mask,bsize)
	with utils.nowarn():
		correction = utils.without_nan(rhs/div)
	enmap.write_map("corr1.fits", correction)
	correction = gapfill_edgemean(correction, ~mask)
	enmap.write_map("corr2.fits", correction)
	oivars = ivars / np.maximum(correction, np.max(correction)*1e-3)
	return oivars
def measure_white_noise(diffmaps, lknee=5000, alpha=-10, bsize=2):
	# Now that I'm doing phase differences before calculating
	# the noise, this filtering is unneccessary (but not harmful)
	fmap = enmap.fft(diffmaps)
	l    = fmap.modlmap()+0.5
	F    = (1+(l/lknee)**alpha)**-1
	whitemaps = enmap.ifft(F*fmap).real
	var  = blockavg(np.mean(whitemaps**2,(0,1)),bsize)
	# Undo loss of power from filtering
	var /= np.mean(F**2)
	return var

utils.mkdir(args.odir)
prefix = args.odir + "/"
if args.tag: prefix += args.tag + "_"
bsize = args.noise_block_size

# Read in the maps [nt,ncomp,ny,nx]. We read in two data splits
# to build an empirical noise model without worrying about signal
# contamination
split_maps = enmap.enmap([enmap.read_map(ifile) for ifile in args.imaps])
nmap, nbin  = split_maps.shape[:2]
# Output them again for reference. They shouldn't be big anyway
enmap.write_map(prefix + "split_maps.fits", split_maps)
# Output coadd across splits too
enmap.write_map(prefix + "map_coadd.fits", np.mean(split_maps,0))
# From this point we will only work with phase-mean-subtracted maps
phase_diffs = split_maps - np.mean(split_maps,1)[:,None]
# Build the noise model. We don't trust the ivar from the
# mapmaker, as we have strong evidence of multiplicative noise.
# There's therefore not much point in reading in ivar.
# We build the diff the noise model is based on from splits. That way
# the real pulsar signal we're looking for will have zero contribution.
# Sadly this seems to overestimate the total noise.
split_diffs= phase_diffs - np.mean(phase_diffs,0)
enmap.write_map(prefix + "split_diffs.fits", split_diffs)
vemp       = measure_white_noise(split_diffs, bsize=bsize)
enmap.write_map(prefix + "vemp.fits", vemp)
# vemp is now the noise of an individual split
# From now on we won't work with splits any more
map_diffs = np.mean(phase_diffs,0) # [nt,ncomp,ny,nx]
enmap.write_map(prefix + "map_diffs.fits", map_diffs)
with utils.nowarn():
	ivar_coadd = utils.without_nan((vemp/nmap)**-1)
enmap.write_map(prefix + "ivar_coadd.fits", ivar_coadd)
ivar_diffs   = ivar_coadd + map_diffs*0

# Setup our beam, which we need for the matched filter
bl  = get_beam(args.beam)
uht = uharm.UHT(map_diffs.shape, map_diffs.wcs)
B   = enmap.samewcs(utils.interp(uht.l, np.arange(len(bl)), bl), map_diffs)
# Do the matched filter
fconv = utils.dplanck(args.freq*1e9)/1e3
rho, kappa = analysis.matched_filter_white(map_diffs*fconv, B, ivar_diffs/fconv**2, uht)
enmap.write_map(prefix + "rho.fits", rho)
enmap.write_map(prefix + "kappa.fits", kappa)
# Solve
kappa = np.maximum(kappa, np.max(kappa)*1e-3)
flux  = rho/kappa
dflux = kappa**-0.5
enmap.write_map(prefix + "flux.fits", flux)
enmap.write_map(prefix + "kappa.fits", kappa)
enmap.write_map(prefix + "snr.fits", flux/dflux)
# Read off values at center
flux_vals  = flux.at([0,0]).T # [ncomp,nt]
dflux_vals = dflux.at([0,0], order=1).T
# My error bars are a bit big, but an empirical test shows they're right. This must
# just mean that the bins are strongly correlated.
#snr = rho/kappa**0.5
#correction = np.std(snr,(0,-2,-1)) # [ncomp]
#dflux_vals *= correction[:,None]
oarr = np.moveaxis(np.array([flux_vals, dflux_vals]),0,1).reshape(-1, flux_vals.shape[-1])
np.savetxt(prefix + "flux.txt", np.concatenate([
	np.arange(nbin)[None]/nbin, oarr]).T, fmt="%15.7e")
