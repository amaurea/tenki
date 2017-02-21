import numpy as np, argparse, os
from enlib import enmap, array_ops, utils, mpi
from scipy import ndimage
parser = argparse.ArgumentParser()
parser.add_argument("imaps_and_hits", nargs="+")
parser.add_argument("omap")
parser.add_argument("ohit")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-a", "--apod",    type=str, default=None)
parser.add_argument("-e", "--edge",    type=int, default=0)
parser.add_argument("-t", "--trim",    type=int, default=0, help="Amount to trim maps that need to be interplated by, in pixels on each side.")
parser.add_argument("-c", "--cont",    action="store_true")
parser.add_argument("--fslice", type=str, default="")
args = parser.parse_args()

comm = mpi.COMM_WORLD

n = len(args.imaps_and_hits)
imaps = args.imaps_and_hits[:n/2]
ihits = args.imaps_and_hits[n/2:]
imaps = eval("imaps" + args.fslice)
ihits = eval("ihits" + args.fslice)

apod_params = [float(w) for w in args.apod.split(":")] if args.apod else None

def read_map(fname):
	m = nonan(enmap.read_map(fname))
	#return m.preflat[:1]
	return m.reshape(-1, m.shape[-2], m.shape[-1])
def read_div(fname, padlen):
	m = nonan(enmap.read_map(fname))*1.0
	#return m.preflat[:1][None]
	if m.ndim == 2:
		res = enmap.zeros((padlen,padlen)+m.shape[-2:], m.wcs, m.dtype)
		for i in range(padlen):
			res[i,i] = m
		return res
	elif m.ndim == 4: return m
	else: raise ValueError("Wrong number of dimensions in div %s" % fname)
def get_tilenames(dir):
	return sorted([name for name in os.listdir(dir) if name.endswith(".fits") or name.endswith(".hdf")])

def mul(w,m):
	if w.ndim == 2: return m*w[None]
	elif w.ndim == 3: return m*w
	elif w.ndim == 4: return enmap.samewcs(array_ops.matmul(w,m, axes=[0,1]),m)
	else: raise NotImplementedError("Only 2d, 3d or 4d weight maps understood")
def solve(w,m):
	if w.ndim == 2: return m/w[None]
	elif w.ndim == 3: return m/w
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
	weight = div[0,0]
	moo = enmap.downgrade(weight,50)
	print np.mean(moo), np.median(moo), np.max(moo)
	maxval = np.max(enmap.downgrade(weight,50))
	apod   = np.minimum(1,weight/maxval/apod_params[0])**apod_params[1]
	return div*apod[None,None]
def apply_trim(div):
	t = args.trim
	if t <= 0: return div
	div[:,:,range(t)+range(-t,0),:] = 0
	div[:,:,:,range(t)+range(-t)] = 0
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
	w = div[0,0]*0+1
	w[[0,-1],:] = 0
	w[:,[0,-1]] = 0
	dists = ndimage.distance_transform_edt(w)
	apod = np.minimum(1,dists/float(args.edge))
	return div*apod[None,None]

def coadd_maps(imaps, ihits, omap, ohit, cont=False):
	# The first map will be used as a reference. All subsequent maps
	# must fit in its boundaries.
	if cont and os.path.exists(omap): return
	if args.verbose: print "Reading %s" % imaps[0]
	m = read_map(imaps[0])
	if args.verbose: print"Reading %s" % ihits[0]
	w = apply_edge(apply_apod(apply_trim(read_div(ihits[0], len(m)))))
	wm = mul(w,m)

	for mif,wif in zip(imaps[1:],ihits[1:]):
		if args.verbose: print"Reading %s" % mif
		mi = read_map(mif)
		if args.verbose: print"Reading %s" % wif
		wi = apply_edge(apply_apod(apply_trim(read_div(wif, len(mi)))))
		# We may need to reproject maps
		if mi.shape != m.shape or str(mi.wcs.to_header()) != str(m.wcs.to_header()):
			mi = enmap.project(mi, m.shape, m.wcs, mode="constant")
			wi = enmap.project(wi, w.shape, w.wcs, mode="constant")
		w[:len(wi),:len(wi)] += wi
		wm[:len(wi)] += mul(wi,mi)

	if args.verbose: print"Solving"
	m = solve(w,wm)
	if args.verbose: print"Writing %s" % omap
	enmap.write_map(omap, m)
	if args.verbose: print"Writing %s" % ohit
	enmap.write_map(ohit, w)

# Two cases: Normal enmaps or dmaps
if not os.path.isdir(imaps[0]):
	# Normal monotlithic map
	coadd_maps(imaps, ihits, args.omap, args.ohit, cont=args.cont)
else:
	# Dmap. Each name is actually a directory, but they
	# all have compatible tile names.
	tilenames = get_tilenames(imaps[0])
	utils.mkdir(args.omap)
	utils.mkdir(args.ohit)
	for tilename in tilenames[comm.rank::comm.size]:
		timaps = ["%s/%s" % (imap,tilename) for imap in imaps]
		tihits = ["%s/%s" % (ihit,tilename) for ihit in ihits]
		print "%3d %s" % (comm.rank, tilename)
		coadd_maps(timaps, tihits, args.omap + "/" + tilename, args.ohit + "/" + tilename, cont=args.cont)
	if args.verbose: print"Done"
