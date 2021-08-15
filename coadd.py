from __future__ import division, print_function
import numpy as np, argparse, os
from enlib import enmap, array_ops, utils, mpi
from scipy import ndimage
parser = argparse.ArgumentParser()
parser.add_argument("imaps_and_hits", nargs="+", help="map map map ... hits hits hits ... unless --transpose, in which case it's map hits map hits map hits ...")
parser.add_argument("omap")
parser.add_argument("ohit")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-a", "--apod",    type=str, default=None)
parser.add_argument("-e", "--edge",    type=int, default=0)
parser.add_argument("-t", "--trim",    type=int, default=0, help="Amount to trim maps that need to be interplated by, in pixels on each side.")
parser.add_argument("--fslice",        type=str, default="")
parser.add_argument("-c", "--cont",          action="store_true")
parser.add_argument("-M", "--allow-missing", action="store_true")
parser.add_argument("-T", "--transpose",     action="store_true")
parser.add_argument("-W", "--warn",          action="store_true")
parser.add_argument("-N", "--ncomp",   type=int, default=-1)
args = parser.parse_args()

comm = mpi.COMM_WORLD

n = len(args.imaps_and_hits)//2
if not args.transpose:
	imaps = args.imaps_and_hits[:n]
	ihits = args.imaps_and_hits[n:]
else:
	imaps = args.imaps_and_hits[0::2]
	ihits = args.imaps_and_hits[1::2]

imaps = eval("imaps" + args.fslice)
ihits = eval("ihits" + args.fslice)

apod_params = [float(w) for w in args.apod.split(":")] if args.apod else None

def read_helper(fname, shape=None, wcs=None):
	if shape is None: return enmap.read_map(fname)
	mshape, mwcs = enmap.read_map_geometry(fname)
	pixbox = enmap.pixbox_of(mwcs, shape, wcs)
	return enmap.read_map(fname, pixbox=pixbox)

def read_map(fname, shape=None, wcs=None, ncomp=3):
	m = nonan(read_helper(fname, shape, wcs))
	#return m.preflat[:1]
	m = m.reshape(-1, m.shape[-2], m.shape[-1])
	if ncomp == 0: return m[0]
	if len(m) == 1:
		res = enmap.zeros((ncomp,)+m.shape[1:],m.wcs,m.dtype)
		res[0] = m
		return res
	else: return m
def read_div(fname, shape=None, wcs=None, ncomp=3):
	m = nonan(read_helper(fname, shape, wcs))*1.0
	if ncomp == 0: return m.preflat[0]
	#return m.preflat[:1][None]
	if m.ndim == 2:
		res = enmap.zeros((ncomp,ncomp)+m.shape[-2:], m.wcs, m.dtype)
		for i in range(ncomp):
			res[i,i] = m
		return res
	elif m.ndim == 3:
		res = enmap.zeros((ncomp,ncomp)+m.shape[-2:], m.wcs, m.dtype)
		for i in range(ncomp):
			res[i,i] = m[i]
		return res
	elif m.ndim == 4: return m
	else: raise ValueError("Wrong number of dimensions in div %s" % fname)
def get_tilenames(dir):
	return sorted([name for name in os.listdir(dir) if name.endswith(".fits") or name.endswith(".hdf")])

def mul(w,m):
	if w.ndim < 4: return m*w
	elif w.ndim == 4: return enmap.samewcs(array_ops.matmul(w,m, axes=[0,1]),m)
	else: raise NotImplementedError("Only 2d, 3d or 4d weight maps understood")
def add(m1,m2):
	ndim = min(m1.ndim,m2.ndim)
	if m1.ndim == m2.ndim or ndim < 4: return m1+m2
	elif ndim == 4:
		if m1.ndim < m2.ndim: m1,m2 = m2,m1
		res = m1.copy()
		for i in range(len(m1)):
			res[i,i] += (m2 if m2.ndim == 2 else m2[i])
		return res
	else: raise NotImplementedError("Only 2d, 3d or 4d maps understood")
def solve(w,m):
	if w.ndim < 4: return m/w
	elif w.ndim == 4:
		# This is slower, but handles low-hit areas near the edge better
		iw = array_ops.eigpow(w,-1,axes=[0,1])
		return enmap.samewcs(array_ops.matmul(iw,m,axes=[0,1]), m)
		#return array_ops.solve_masked(w,m,axes=[0,1])
	else: raise NotImplementedError("Only 2d, 3d or 4d weight maps understood")
def nonan(a):
	res = a.copy()
	res[~np.isfinite(res)] = 0
	return res
def apply_apod(div):
	if apod_params is None: return div
	weight = div.preflat[0]
	moo = enmap.downgrade(weight,50)
	maxval = np.max(enmap.downgrade(weight,50))
	apod   = np.minimum(1,weight/maxval/apod_params[0])**apod_params[1]
	return div*apod
def apply_trim(div):
	t = args.trim
	if t <= 0: return div
	div[...,range(t)+range(-t,0),:] = 0
	div[...,:,range(t)+range(-t)] = 0
	return div
	#fdiv = div.reshape((-1,)+div.shape[-2:])
	#dists= ndimage.distance_transform_edt(np.any(fdiv!=0,0))
	#mask = (dists>0)&(dists<args.trim)
	#apod = dists[mask]*float(args.trim)**-1
	#for cdiv in fdiv:
	#	print "A"
	#	cdiv[mask] *= apod
	return div
def apply_edge(div):
	if args.edge == 0: return div
	w = div.preflat[0]*0+1
	w[[0,-1],:] = 0
	w[:,[0,-1]] = 0
	dists = ndimage.distance_transform_edt(w)
	apod = np.minimum(1,dists/float(args.edge))
	return div*apod

def coadd_maps(imaps, ihits, omap, ohit, cont=False, ncomp=-1):
	# The first map will be used as a reference. All subsequent maps
	# must fit in its boundaries.
	if cont and os.path.exists(omap): return
	if args.verbose: print("Reading %s" % imaps[0])
	if ncomp < 0:
		shape, wcs = enmap.read_map_geometry(imaps[0])
		ncomp = 0 if len(shape) == 2 else shape[0]
	m = read_map(imaps[0], ncomp=ncomp)
	if args.verbose: print("Reading %s" % ihits[0])
	w = apply_edge(apply_apod(apply_trim(read_div(ihits[0], ncomp=ncomp))))
	if args.warn and np.any(w.preflat[0]<0):
		print("Negative weight in %s" % ihits[0])
	wm = mul(w,m)

	for i, (mif,wif) in enumerate(zip(imaps[1:],ihits[1:])):
		if args.verbose: print("Reading %s" % mif)
		try:
			mi = read_map(mif, m.shape, m.wcs, ncomp=ncomp)
		except (IOError, OSError):
			if args.allow_missing:
				print("Can't read %s. Skipping" % mif)
				continue
			else: raise
		if args.verbose: print("Reading %s" % wif)

		wi = apply_edge(apply_apod(apply_trim(read_div(wif, m.shape, m.wcs, ncomp=ncomp))))
		if args.warn and np.any(wi.preflat[0]<0):
			print("Negative weight in %s" % ihits[i+1])
		## We may need to reproject maps
		#if mi.shape != m.shape or str(mi.wcs.to_header()) != str(m.wcs.to_header()):
		#	mi = enmap.extract(mi, m.shape, m.wcs)
		#	wi = enmap.extract(wi, w.shape, w.wcs)
		w  = add(w,wi)
		wm = add(wm,mul(wi,mi))

	if args.verbose: print("Solving")
	m = solve(w,wm)
	if args.verbose: print("Writing %s" % omap)
	enmap.write_map(omap, m)
	if args.verbose: print("Writing %s" % ohit)
	enmap.write_map(ohit, w)

# Two cases: Normal enmaps or dmaps
if not os.path.isdir(imaps[0]):
	# Normal monotlithic map
	coadd_maps(imaps, ihits, args.omap, args.ohit, cont=args.cont, ncomp=args.ncomp)
else:
	# Dmap. Each name is actually a directory, but they
	# all have compatible tile names.
	tilenames = get_tilenames(imaps[0])
	utils.mkdir(args.omap)
	utils.mkdir(args.ohit)
	for tilename in tilenames[comm.rank::comm.size]:
		timaps = ["%s/%s" % (imap,tilename) for imap in imaps]
		tihits = ["%s/%s" % (ihit,tilename) for ihit in ihits]
		print("%3d %s" % (comm.rank, tilename))
		coadd_maps(timaps, tihits, args.omap + "/" + tilename, args.ohit + "/" + tilename, cont=args.cont, ncomp=args.ncomp)
	if args.verbose: print("Done")
