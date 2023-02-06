import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mapstack")
parser.add_argument("ivar")
parser.add_argument("odir")
parser.add_argument("tag", nargs="?", default=None)
parser.add_argument("-b", "--beam", type=str,   default=1.4)
parser.add_argument("-f", "--freq", type=float, default=98)
parser.add_argument("-R", "--rmask",type=float, default=2)
parser.add_argument(     "--noise-block-size", type=int, default=5)
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

utils.mkdir(args.odir)
prefix = args.odir + "/"
if args.tag: prefix += args.tag + "_"
bsize = args.noise_block_size
rmask = args.rmask*utils.arcmin

# Read in the maps [nt,ncomp,ny,nx]
imaps = enmap.read_map(args.mapstack)
ivars = enmap.read_map(args.ivar)
nmap  = len(imaps)
# Output them again for reference. They shouldn't be big anyway
enmap.write_map(prefix + "map.fits", imaps)
enmap.write_map(prefix + "ivar.fits", imaps)
# Compute the coadded map, which we will subtract
rhs = np.sum(imaps*ivars[:,None],0)
div = np.sum(ivars,0)
with utils.nowarn():
	map_coadd = utils.without_nan(rhs/div[None])
# Subtract this coadd map to get the deviation from mean.
# This is where the pulsar pulses would be
imaps -= map_coadd
# Output cleaned map
enmap.write_map(prefix + "diffmap.fits", imaps)
# We can't really trust ivars as it is because there will be model error
# noise in the region of bright signal. But we can try to fit this empirically.
# For example var_tot = 1/ivar + a*abs(T). But for now, just do a simple block
# measurement.
#with utils.nowarn():
#	correction = blockavg(imaps**2*ivars[:,None], bsize)
#	# Maximum to avoid dividing by too small values, which can happen in very poorly hit pixels
#	ivars_emp  = ivars[:,None] / np.maximum(correction, np.max(correction)*1e-3)
ivars_emp = calibrate_ivar(imaps, ivars[:,None], rmask=rmask)
bl  = get_beam(args.beam)
uht = uharm.UHT(imaps.shape, imaps.wcs)
B   = enmap.samewcs(utils.interp(uht.l, np.arange(len(bl)), bl), imaps)
# Do the matched filter
fconv = utils.dplanck(args.freq*1e9)/1e3
rho, kappa = analysis.matched_filter_white(imaps*fconv, B, ivars_emp/fconv**2, uht)
enmap.write_map(prefix + "rho.fits", rho)
enmap.write_map(prefix + "kappa.fits", kappa)
# Solve
kappa = np.maximum(kappa, np.max(kappa)*1e-3)
flux  = rho/kappa
dflux = kappa**-0.5
enmap.write_map(prefix + "flux.fits", flux)
enmap.write_map(prefix + "kappa.fits", dflux)
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
	np.arange(nmap)[None]/nmap, oarr]).T, fmt="%15.7e")
