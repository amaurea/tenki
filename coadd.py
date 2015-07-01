import numpy as np, argparse
from enlib import enmap, log, array_ops
parser = argparse.ArgumentParser()
parser.add_argument("imaps_and_hits", nargs="+")
parser.add_argument("omap")
parser.add_argument("ohit")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

L = log.init(level=log.DEBUG if args.verbose else log.ERROR)

n = len(args.imaps_and_hits)
imaps = args.imaps_and_hits[:n/2]
ihits = args.imaps_and_hits[n/2:]

def mul(w,m):
	if w.ndim == 2: return m*w[None]
	elif w.ndim == 3: return m*w
	elif w.ndim == 4: return enmap.map_mul(w,m)
	else: raise NotImplementedError("Only 2d, 3d or 4d weight maps understood")
def solve(w,m):
	if w.ndim == 2: return m/w[None]
	elif w.ndim == 3: return m/w
	#elif w.ndim == 4: return enmap.map_mul(enmap.multi_pow(w,-1),m)
	elif w.ndim == 4: return array_ops.solve_masked(w,m,axes=[0,1])#  enmap.map_mul(enmap.multi_pow(w,-1),m)
	else: raise NotImplementedError("Only 2d, 3d or 4d weight maps understood")
def nonan(a):
	res = a.copy()
	res[np.isnan(res)] = 0
	return res

# The first map will be used as a reference. All subsequent maps
# must fit in its boundaries.
L.info("Reading %s" % imaps[0])
m = nonan(enmap.read_map(imaps[0]))
L.info("Reading %s" % ihits[0])
w = nonan(enmap.read_map(ihits[0]))
wm = mul(w,m)

for mif,wif in zip(imaps[1:],ihits[1:]):
	L.info("Reading %s" % mif)
	mi = nonan(enmap.read_map(mif))
	L.info("Reading %s" % wif)
	wi = nonan(enmap.read_map(wif))
	# We may need to reproject maps
	if mi.shape != m.shape or str(mi.wcs.to_header()) != str(m.wcs.to_header()):
		mi = enmap.project(mi, m.shape, m.wcs, mode="constant")
		wi = enmap.project(wi, w.shape, w.wcs, mode="constant")
	w  += wi
	wm += mul(wi,mi)

L.info("Solving")
m = solve(w,wm)
L.info("Writing %s" % args.omap)
enmap.write_map(args.omap, m)
L.info("Writing %s" % args.ohit)
enmap.write_map(args.ohit, w)
