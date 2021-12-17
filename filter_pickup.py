import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("ivar")
parser.add_argument("omap")
parser.add_argument("-m", "--mask", type=str,   default=None)
parser.add_argument("-y", "--ly",   type=float, default=4000)
parser.add_argument("-x", "--lx",   type=float, default=5)
parser.add_argument("-M", "--mode", type=str,   default="weighted")
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils

def build_filter(shape, wcs, lbounds, dtype=np.float32):
	# Intermediate because the filter we're applying is for a systematic that
	# isn't on the curved sky.
	ly, lx  = enmap.laxes(shape, wcs, method="intermediate")
	ly, lx  = [a.astype(dtype) for a in [ly,lx]]
	lbounds = np.asarray(lbounds).astype(dtype)
	if lbounds.ndim < 2:
		lbounds = np.broadcast_to(lbounds, (1,2))
	if lbounds.ndim > 2 or lbounds.shape[-1] != 2:
		raise ValueError("lbounds must be [:,{ly,lx}]")
	filter = enmap.ones(shape[-2:], wcs, dtype)
	# Apply the filters
	for i , (ycut, xcut) in enumerate(lbounds):
		filter *= 1-(np.exp(-0.5*(ly/ycut)**2)[:,None]*np.exp(-0.5*(lx/xcut)**2)[None,:])
	return filter

def filter_simple(imap, filter):
	return enmap.ifft(enmap.fft(imap)*filter).real

def filter_weighted(imap, ivar, filter, tol=1e-4, ref=0.9):
	"""Filter enmap imap with the given 2d fourier filter while
	weithing spatially with ivar"""
	filter = 1-filter
	omap   = enmap.ifft(filter*enmap.fft(imap*ivar)).real
	div    = enmap.ifft(filter*enmap.fft(     ivar)).real
	del filter
	# Avoid division by very low values
	div    = np.maximum(div, np.percentile(ivar[::10,::10],ref*100)*tol)
	# omap = imap - rhs/div
	omap /= div
	del div
	omap *= -1
	omap += imap
	omap *= imap != 0
	return omap

def filter_riseset(imap, ivar, riseset, filter, tol=1e-4, ref=0.9):
	"""Filter enmap imap with the given 2d fourier filter while
	weighting spatially with ivar and allowing for rise vs set-dependent
	pickup. riseset should be a map with values between -1 and 1 that determines
	the balance between rising and setting scans, with 0 meaning equal weight from both.
	Overall this is only a marginal improvement over filter_weighted depsite being
	quite a bit heaver, and it only helps a bit for the pa7 residual stripe issue it
	was invented to deal with, so it probably isn't worth using.
	"""
	# We model the map as m = Pa+n, where a is [2] is the rising and setting pickup in
	# a single pixel, and P is the map's response to this pickup, which is
	# P = [x,1-x]*filter, where x=(1-riseset)/2. Given this we can solve for a as:
	# a = (P'N"P)"P'N"m, where N" is ivar. Written out this becomes, for a single pixel.
	# rhs = ifft(filter*fft([x,1-x]*ivar*imap))
	# div = ifft(filter*fft([x,1-x]*ivar*[x,1-x]*ifft(filter)
	# Hm.... The problem here is that we're assuming that there's nothing in the other pixels.
	# That's not what we did in filter_weighted. Let's just do something simple for now.
	x      = (1-riseset)/2
	x1x    = np.array([x,1-x]); del x
	# Broadcast to imap shape
	x1x    = x1x[(slice(None),)+(None,)*(imap.ndim-2)]
	print(imap.shape, x1x.shape)
	filter = 1-filter
	rhs    = enmap.ifft(filter*enmap.fft(x1x*ivar*imap)).real
	div    = enmap.ifft(filter*enmap.fft(x1x[:,None]*x1x[None,:]*ivar)).real
	del filter
	# Avoid division by very low values
	ref    = np.percentile(ivar[::10,::10],ref*100)*tol
	for i in range(2):
		div[i,i] = np.maximum(div[i,i],ref)
	# Solve the system
	rhs = np.moveaxis(rhs,0,-1)
	div = np.moveaxis(div,(0,1),(-2,-1))
	omap= np.linalg.solve(div,rhs)
	del rhs, div
	omap = np.sum(x1x*np.moveaxis(omap,-1,0),0)
	del x1x
	omap = enmap.ndmap(omap,imap.wcs)
	omap *= -1
	omap += imap
	omap *= imap != 0
	return omap

lbounds = np.array([args.ly, args.lx])
imap    = enmap.read_map(args.imap)
dtype   = imap.dtype
filter  = build_filter(imap.shape, imap.wcs, lbounds, dtype=dtype)

if   args.mode == "simple":
	omap = filter_simple(imap, filter)
elif args.mode == "weighted":
	ivar = enmap.read_map(args.ivar).preflat[0].astype(dtype, copy=False)
	if args.mask:
		ivar *= 1-enmap.read_map(args.mask, geometry=ivar.geometry).preflat[0]
	omap = filter_weighted(imap, ivar, filter); del ivar
elif args.mode == "riseset":
	ivar, riseset = enmap.read_map(args.ivar).astype(dtype, copy=False)[[0,2]]
	riseset /= np.maximum(ivar, np.max(ivar)*1e-6)
	if args.mask:
		ivar *= 1-enmap.read_map(args.mask, geometry=ivar.geometry).preflat[0]
	omap = filter_riseset(imap, ivar, riseset, filter); del ivar, riseset

del filter, imap
enmap.write_map(args.omap, omap)
