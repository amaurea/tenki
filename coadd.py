import numpy as np, argparse
from enlib import enmap, log, array_ops
from scipy import ndimage
parser = argparse.ArgumentParser()
parser.add_argument("imaps_and_hits", nargs="+")
parser.add_argument("omap")
parser.add_argument("ohit")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-a", "--apod",    type=str, default=None)
parser.add_argument("-t", "--trim",    type=int, default=1, help="Amount to trim maps that need to be interplated by, in pixels on each side.")
args = parser.parse_args()

L = log.init(level=log.DEBUG if args.verbose else log.ERROR)

n = len(args.imaps_and_hits)
imaps = args.imaps_and_hits[:n/2]
ihits = args.imaps_and_hits[n/2:]

apod_params = [float(w) for w in args.apod.split(":")] if args.apod else None

def read_map(fname):
	m = nonan(enmap.read_map(fname))
	return m.reshape(-1, m.shape[-2], m.shape[-1])
def read_div(fname, padlen):
	m = nonan(enmap.read_map(fname))
	if m.ndim == 2:
		res = enmap.zeros((padlen,padlen)+m.shape[-2:], m.wcs, m.dtype)
		for i in range(padlen):
			res[i,i] = m
		return res
	elif m.ndim == 4: return m
	else: raise ValueError("Wrong number of dimensions in div %s" % fname)

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
	maxval = np.max(enmap.downgrade(weight,100))
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

# The first map will be used as a reference. All subsequent maps
# must fit in its boundaries.
L.info("Reading %s" % imaps[0])
m = read_map(imaps[0])
L.info("Reading %s" % ihits[0])
w = apply_trim(apply_apod(read_div(ihits[0], len(m))))
wm = mul(w,m)

for mif,wif in zip(imaps[1:],ihits[1:]):
	L.info("Reading %s" % mif)
	mi = read_map(mif)
	L.info("Reading %s" % wif)
	wi = apply_trim(apply_apod(read_div(wif, len(mi))))
	# We may need to reproject maps
	if mi.shape != m.shape or str(mi.wcs.to_header()) != str(m.wcs.to_header()):
		mi = enmap.project(mi, m.shape, m.wcs, mode="constant")
		wi = enmap.project(wi, w.shape, w.wcs, mode="constant")
	w[:len(wi),:len(wi)] += wi
	wm[:len(wi)] += mul(wi,mi)

L.info("Solving")
m = solve(w,wm)
L.info("Writing %s" % args.omap)
enmap.write_map(args.omap, m)
L.info("Writing %s" % args.ohit)
enmap.write_map(args.ohit, w)
