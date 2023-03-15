import argparse
parser = argparse.ArgumentParser()
parser.add_argument("maps1")
parser.add_argument("maps2")
parser.add_argument("odir")
parser.add_argument("tag", nargs="?", default=None)
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

utils.mkdir(args.odir)
prefix = args.odir + "/"
if args.tag: prefix += args.tag + "_"
bsize = args.noise_block_size
#rmask = args.rmask*utils.arcmin

# Read in the maps [nt,ncomp,ny,nx]. We read in two data splits
# to build an empirical noise model without worrying about signal
# contamination
imaps1 = enmap.read_map(args.maps1)
imaps2 = enmap.read_map(args.maps2)
nmap   = len(imaps1)
# Output them again for reference. They shouldn't be big anyway
enmap.write_map(prefix + "maps1.fits", imaps1)
enmap.write_map(prefix + "maps2.fits", imaps2)
# Build the noise model. We don't trust the ivar from the
# mapmaker, as we have strong evidence of multiplicative noise.
# There's therefore not much point in reading in ivar. /2 takes us
# from var of diff to var of individual

# These diffs reveal pointing issues for the tau_a tods. Need to build
# new pointing model
enmap.write_map(prefix + "rawdiffs.fits", imaps1-imaps2)


vemp = blockavg(np.mean((imaps1-imaps2)**2,0),bsize) / 2
mask = np.all((imaps1[:,0]!=0)&(imaps2[:,0]!=0),0)
# Coadd, assuming uniform noise between splits and bins. The bin part
# is completely safe. The split thing is almost certainly good enough.
map_coadd  = 0.5*(np.mean(imaps1,0)+np.mean(imaps2,0))
with utils.nowarn():
	ivar_coadd = utils.without_nan(2*nmap / vemp)*mask
enmap.write_map(prefix + "map_coadd.fits",  map_coadd)
enmap.write_map(prefix + "ivar_coadd.fits", ivar_coadd)
# Mean-subtracted split-coadded maps
maps  = 0.5*(imaps1+imaps2) - map_coadd
with utils.nowarn():
	ivars = ivar_coadd / (nmap-1) + maps*0
enmap.write_map(prefix + "diffmap.fits", maps)

# Estimating noise from diff maps results in a factor 5
# overestimate of the noise RMS in the central region!
vemp2 = blockavg(np.mean(maps**2*ivars,0),bsize)
enmap.write_map(prefix + "diffrms.fits", vemp2**0.5)


# Setup our beam, which we need for the matched filter
bl  = get_beam(args.beam)
uht = uharm.UHT(maps.shape, maps.wcs)
B   = enmap.samewcs(utils.interp(uht.l, np.arange(len(bl)), bl), maps)
# Do the matched filter
fconv = utils.dplanck(args.freq*1e9)/1e3
rho, kappa = analysis.matched_filter_white(maps*fconv, B, ivars/fconv**2, uht)
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
