import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("ivar")
parser.add_argument("omap")
parser.add_argument("-m", "--mask", type=str,   default=None)
parser.add_argument("-y", "--ly",   type=float, default=4000)
parser.add_argument("-x", "--lx",   type=float, default=5)
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils

def build_filter(shape, wcs, lbounds):
	# Intermediate because the filter we're applying is for a systematic that
	# isn't on the curved sky.
	ly, lx  = enmap.laxes(shape, wcs, method="intermediate")
	lbounds = np.asarray(lbounds)
	if lbounds.ndim < 2:
		lbounds = np.broadcast_to(lbounds, (1,2))
	if lbounds.ndim > 2 or lbounds.shape[-1] != 2:
		raise ValueError("lbounds must be [:,{ly,lx}]")
	filter = enmap.ones(shape[-2:], wcs)
	# Apply the filters
	for i , (ycut, xcut) in enumerate(lbounds):
		filter *= 1-(np.exp(-0.5*(ly/ycut)**2)[:,None]*np.exp(-0.5*(lx/xcut)**2)[None,:])
	return filter

def filter_map(imap, ivar, filter, tol=1e-4, ref=0.9):
	"""Filter enmap imap with the given 2d fourier filter while
	weithing spatially with ivar"""
	omap   = enmap.ifft(filter*enmap.fft(imap*ivar)).real
	div    = enmap.ifft(filter*enmap.fft(     ivar)).real
	# Avoid division by very low values
	div    = np.maximum(div, np.percentile(ivar,ref*100)*tol)
	# Solve for the stuff we want to filter away
	omap /= div
	return omap

lbounds = np.array([args.ly, args.lx])
imap = enmap.read_map(args.imap)
ivar = enmap.read_map(args.ivar).preflat[0]
if args.mask:
	ivar *= 1-enmap.read_map(args.mask, geometry=ivar.geometry).preflat[0]

filter = build_filter(imap.shape, imap.wcs, lbounds)
#enmap.write_map("filter.fits", filter)
bad    = filter_map_simple(imap, ivar, 1-filter); del filter, ivar
#enmap.write_map("bad.fits", bad)
omap   = (imap-bad)*(imap!=0); del bad
enmap.write_map(args.omap, omap)
