# Program for benchmarking pointing matrix implementations.
# Results on my laptop Intel(R) Core(TM) i5-3210M CPU @ 2.50GHz (2(4) cores)
# Accuracy is in units of 0.01 pixels of size 0.5' each.
# Size is in millions, but memory requirements for current grad implementation
# is much higher, and by sizes of about 100M we use 10 GB or so.
# Build time is proportional to size, with a factor of about 7s/M
# Tod used is (100,100000)
# Methods:
#  ograd    original grad interpol, with precomputed ys({val,dval1,dval2,dval3},...)
#  grad     like ograd, but ys is precomputed internally before the main loop
#  igrad    like grad, but nothing is precomputed
#  bilin    bilinear interpolation with no precomputation
# For grad methods, time is ograd 64, grad 64, igrad 64, ograd 32, grad 32, igrad 32
# For bilin method, time is bilin 64, bilin 32
#
# method msize    acc     max    std  size time
#  grad    0.1 100.00  71.148 14.987  0.00 0.339 0.343 0.339 0.272 0.277 0.287
#  grad    0.1  10.00   6.954  1.642  0.03 0.506 0.526 0.369 0.417 0.439 0.326
#  grad    0.1   1.00   1.625  0.385  0.24 0.587 0.596 0.598 0.510 0.562 0.561
#  grad    1.0   1.00   0.775  0.164  0.98 0.610 0.640 0.630 0.552 0.581 0.586
#  grad   10.0   0.50   0.395  0.093  2.00 0.633 0.663 0.694 0.600 0.603 0.591
#  grad   10.0   0.25   0.188  0.040  8.13 0.834 0.912 0.631 0.695 0.734 0.674
#  bilin   0.1 100.00 100.839 27.025  0.00 0.449 0.427
#  bilin   0.1  10.00   7.421  1.501  0.00 0.443 0.433
#  bilin   0.1   1.00   1.300  0.279  0.00 0.447 0.429
#  bilin   0.1   0.50   0.617  0.135  0.00 0.453 0.434
#  bilin   0.1   0.25   0.291  0.072  0.01 0.462 0.435
#  bilin   0.1   0.10   0.127  0.026  0.03 0.492 0.461
#  bilin   1.0   0.01   0.007  0.002  0.97 0.894 0.820
#
# So for the same grid size, bilinear is about 50% slower than gradient,
# but is 100 times more accurate.
# bilinear is memory-limited from grid-sizes of about 0.05 or so.
# gradient is memory-limited from about 0.02
# So for reasonable accuracies, gradient is so memory-limited that
# the quite significant difference in flops is drowned in memory
# overhead.

import numpy as np, argparse, bunch, os, time, sys
from enlib import pmat, config, utils, interpol, coordinates, bench, enmap
config.default("map_bits", 64, "Bits to use for maps")
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("--t",   type=float, default=56935, help="mjd")
parser.add_argument("--wt",  type=float, default=15,    help="minutes")
parser.add_argument("--az",  type=float, default=55,    help="degrees")
parser.add_argument("--waz", type=float, default=20,    help="degrees")
parser.add_argument("--el",  type=float, default=60,    help="degrees")
parser.add_argument("--wel", type=float, default=2,     help="degrees")
parser.add_argument("--res", type=float, default=0.5,   help="arcmin")
parser.add_argument("--nsamp", type=int, default=100000)
parser.add_argument("--ndet",  type=int, default=100)
parser.add_argument("--ntime", type=int, default=3)
parser.add_argument("-T", action="store_true")
parser.add_argument("-i", "--interpolator", type=str, default="grad")
parser.add_argument("-s", "--seed", type=int, default=0)
args = parser.parse_args()

# Hardcode an arbitrary site
site = bunch.Bunch(
	lat  = -22.9585,
	lon  = -67.7876,
	alt  = 5188.,
	T    = 273.15,
	P    = 550.,
	hum  = 0.2,
	freq = 150.,
	lapse= 0.0065)

ncomp    = 3
nsamp    = args.nsamp
ndet     = args.ndet
bits     = config.get("map_bits")
acc      = config.get("pmat_accuracy")
max_size = config.get("pmat_interpol_max_size")
max_time = config.get("pmat_interpol_max_time")
dtype    = np.float64 if bits > 32 else np.float32
ptype    = np.float64
core     = pmat.get_core(dtype)
np.random.seed(args.seed)

def hor2cel(hor, toff):
	shape = hor.shape[1:]
	hor = hor.reshape(hor.shape[0],-1).astype(float)
	tmp = coordinates.transform("hor", "cel", hor[1:], time=hor[0]+toff, site=site, pol=True)
	res = np.zeros((4,)+tmp.shape[1:])
	res[0] = utils.rewind(tmp[0], tmp[0,0])
	res[1] = tmp[1]
	res[2] = np.cos(2*tmp[2])
	res[3] = np.sin(2*tmp[2])
	res = res.reshape(res.shape[:1]+shape)
	return res

class hor2pix:
	def __init__(self, shape, wcs, toff):
		self.shape, self.wcs, self.toff = shape, wcs, toff
	def __call__(self, hor):
		res = hor2cel(hor, self.toff)
		res[:2] = enmap.sky2pix(self.shape, self.wcs, res[1::-1])
		return res

# input box
t0 = args.t
ibox = np.array([
		[-args.wt/2., args.wt/2.],
		[args.az-args.waz/2., args.az+args.waz/2.],
		[args.el-args.wel/2., args.el+args.wel/2.],
	]).T
# units
ibox[:,0]  /= 24*60
ibox[:,1:] *= utils.degree
wibox = ibox.copy()
wibox[:,1:] = utils.widen_box(ibox[:,1:])

# output box
icorners = utils.box2corners(ibox)
ocorners = hor2cel(icorners.T, t0)
obox     = utils.minmax(ocorners, -1)[:,:2]
wobox    = utils.widen_box(obox)

# define a pixelization
shape, wcs = enmap.geometry(pos=wobox[:,::-1], res=args.res*utils.arcmin, proj="cea")
map = enmap.rand_gauss((ncomp,)+shape, wcs).astype(dtype)
pbox = np.array([[0,0],shape],dtype=int)
print "shape", shape
# define a test tod
bore = (ibox[None,0] + np.random.uniform(0,1,size=(nsamp,3))*(ibox[1]-ibox[0])[None,:]).astype(ptype)
det_pos = np.zeros((ndet,3),dtype=ptype)
det_comps = np.full((ndet,3),1,dtype=dtype)
tod = np.zeros((ndet,nsamp),dtype=dtype)

transfun = hor2pix(shape, wcs, t0)
errlim   = np.array([0.01, 0.01, utils.arcmin, utils.arcmin])*acc
# build our interpolator
ipfun = {
		"grad":  interpol.ip_grad,
		"igrad": interpol.ip_grad,
		"ograd": interpol.ip_grad,
		"bilin": interpol.ip_ndimage,
	}[args.interpolator]
t1 = time.time()
ipol, obox, ok, err = interpol.build(transfun, ipfun, wibox, errlim,
		maxsize=max_size, maxtime=max_time, return_obox=True, return_status=True, order=1)
t2 = time.time()
tbuild = t2-t1
# evaluate accuracy
pos_exact = transfun(bore.T)
pos_inter = ipol(bore.T)
err  = np.max(np.abs(pos_exact-pos_inter)/errlim[:,None])*acc
err2 = np.max(np.std(pos_exact-pos_inter,1)/errlim)*acc
# evaluate speed
if args.interpolator == "ograd":
	rbox, nbox, yvals = pmat.extract_interpol_params_old(ipol, ptype)
else:
	rbox, nbox, yvals = pmat.extract_interpol_params(ipol, ptype)
t1 = time.time()

dir = -1 if args.T else 1
for i in range(args.ntime):
	if args.interpolator == "grad":
		core.pmat_nearest_grad_precomp_internal(dir, 0, 1, tod.T, map.T, bore.T, det_pos.T, det_comps.T,
			np.zeros(1), rbox.T, nbox, yvals.T, pbox.T)
	elif args.interpolator == "igrad":
		core.pmat_nearest_grad_implicit(dir, 0, 1, tod.T, map.T, bore.T, det_pos.T, det_comps.T,
			np.zeros(1), rbox.T, nbox, yvals.T, pbox.T)
	elif args.interpolator == "ograd":
		core.pmat_nearest_grad_precomp(dir, 0, 1, tod.T, map.T, bore.T, det_pos.T, det_comps.T,
			np.zeros(1), rbox.T, nbox, yvals.T, pbox.T)
	elif args.interpolator == "bilin":
		core.pmat_nearest_bilinear(dir, 0, 1, tod.T, map.T, bore.T, det_pos.T, det_comps.T,
			np.zeros(1), rbox.T, nbox, yvals.T, pbox.T)
t2 = time.time()
tuse = (t2-t1)/args.ntime
print "ip: %s b: %5.3f o: %d s: %5.3f M a: %5.3f %5.3f u: %5.3f v: %10.4e" % (
		args.interpolator, tbuild, ok, np.product(nbox)*1e-6, err, err2, tuse, np.sum(tod**2))
