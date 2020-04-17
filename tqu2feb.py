import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("odir")
parser.add_argument("-b", "--beam", type=str,   default="1.4")
parser.add_argument(      "--lknee",type=float, default=3000)
parser.add_argument(      "--alpha",type=float, default=-3)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-a", "--apodize", type=int, default=0)
parser.add_argument("-m", "--mask", type=float, default=None)
args = parser.parse_args()
import numpy as np, os, glob
from pixell import enmap, utils, mpi
from scipy import ndimage

def get_beam(fname):
	try:
		sigma = float(fname)*utils.arcmin*utils.fwhm
		l     = np.arange(40e3)
		beam  = np.exp(-0.5*(l*sigma)**2)
	except ValueError:
		beam = np.loadtxt(fname, usecols=(1,))
	return beam

#def gapfill(map, mask):
#	omap = map.copy()
#	labels, nlabel = ndimage.label(mask)
#	for obj in ndimage.find_objects(labels):
#		submask = mask[obj]
#		for mi, osingle in enumerate(omap.preflat):
#			submap = osingle[obj]
#			submap[submask] = np.sum(submap[~submask])/max(1,submap[~submask].size)
#	return omap
#
#def build_mask(tmap, matched_filter, niter=3, tol1=10, tol2=0.01):
#	mask  = enmap.zeros(imap.shape[-2:], imap.wcs, bool)
#	for i in range(niter):
#		wmap = gapfill(tmap, mask)
#		wmap = enmap.ifft(enmap.fft(wmap)*matched_filter).real**2
#		lim  = max(np.median(wmap)*tol1**2, np.max(wmap)*tol2**2)
#		mask |= wmap > lim
#	return mask

comm   = mpi.COMM_WORLD
beam1d = get_beam(args.beam)
ifiles = sorted(sum([glob.glob(ifile) for ifile in args.ifiles],[]))

for ind in range(comm.rank, len(ifiles), comm.size):
	ifile = ifiles[ind]
	if args.verbose: print(ifile)
	ofile = args.odir + "/" + ifile
	imap  = enmap.read_map(ifile)
	if args.mask is not None: mask = imap == args.mask
	if args.apodize:
		imap = imap.apod(args.apodize)
	# We will apply a semi-matched-filter to T
	l     = np.maximum(1,imap.modlmap())
	beam2d= enmap.samewcs(np.interp(l, np.arange(len(beam1d)), beam1d), imap)
	matched_filter = (1+(l/args.lknee)**args.alpha)**-1 * beam2d
	fmap  = enmap.map2harm(imap, iau=True)
	fmap[0] *= matched_filter
	omap  = enmap.ifft(fmap).real
	if args.mask is not None:
		omap[mask] = 0
		del mask
	utils.mkdir(os.path.dirname(ofile))
	enmap.write_map(ofile, omap)
	del omap
