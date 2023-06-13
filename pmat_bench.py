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

# On scinet:
#  grad    0.1   1.00 0.373 0.391 0.321 0.305 0.312 0.204
#  grad   10.0   0.25
#  bilin   0.1   1.00 0.208 0.176
#  bilin   1.0   0.01 0.444 0.387

#
# So for the same grid size, bilinear is about 50% slower than gradient,
# but is 100 times more accurate.
# bilinear is memory-limited from grid-sizes of about 0.05 or so.
# gradient is memory-limited from about 0.02
# So for reasonable accuracies, gradient is so memory-limited that
# the quite significant difference in flops is drowned in memory
# overhead.

import numpy as np, argparse, os, time, sys
from enlib import pmat, config, utils, interpol, coordinates, bench, enmap, bunch
config.default("map_bits", 32, "Bits to use for maps")
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("--t",   type=float, default=56935, help="mjd")
parser.add_argument("--wt",  type=float, default=15,    help="minutes")
parser.add_argument("--az",  type=float, default=90,    help="degrees")
parser.add_argument("--waz", type=float, default=80,    help="degrees")
parser.add_argument("--el",  type=float, default=50,    help="degrees")
parser.add_argument("--wel", type=float, default=0,     help="degrees")
parser.add_argument("--res", type=float, default=0.5,   help="arcmin")
parser.add_argument("--dir", type=str,   default="1",   help="1 (forward) or -1 (backward), or list")
parser.add_argument("--nsamp", type=int, default=250000)
parser.add_argument("--ndet",  type=int, default=1000)
parser.add_argument("--ntime", type=int, default=3)
#parser.add_argument("-T", action="store_true")
parser.add_argument("-H", "--hwp", action="store_true")
parser.add_argument("-i", "--interpolator", type=str, default="all")
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
dirs     = [int(w) for w in args.dir.split(",")]
np.random.seed(args.seed)

def hor2cel(hor, toff):
	"""Transform from [{tsec,az,el},nsamp] to [{ra,dec,c,s},nsamp],
	where tsec is the offset from toff. Toff is given in mjd."""
	shape = hor.shape[1:]
	hor = hor.reshape(hor.shape[0],-1).astype(float)
	tmp = coordinates.transform("hor", "cel", hor[1:], time=hor[0]/24/60/60+toff, site=site, pol=True)
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
		"""Transform from [{tsec,az,el},nsamp] to [{y,x,c,s},nsamp]"""
		res = hor2cel(hor, self.toff)
		res[:2] = enmap.sky2pix(self.shape, self.wcs, res[1::-1])
		return res

det_pos = (np.random.standard_normal((ndet,3))*0.2*utils.degree).astype(ptype)
det_pos[:,0] = 0
det_box = np.array([np.min(det_pos,0),np.max(det_pos,0)])
det_comps = np.full((ndet,3),1,dtype=dtype)
wt = args.wt * 60.0

# input box
t0 = args.t # In mjd
ibox = np.array([
		[0, wt],
		[args.az-args.waz/2., args.az+args.waz/2.],
		[args.el-args.wel/2., args.el+args.wel/2.],
	]).T + det_box/utils.degree
# units
ibox[:,1:] *= utils.degree
wibox = ibox.copy()
wibox[:,:] = utils.widen_box(ibox[:,:])
srate = nsamp/wt

# output box
icorners = utils.box2corners(ibox)
ocorners = hor2cel(icorners.T, t0)
obox     = utils.minmax(ocorners, -1)[:,:2]
wobox    = utils.widen_box(obox)

# define a pixelization
shape, wcs = enmap.geometry(pos=wobox[:,::-1], res=args.res*utils.arcmin, proj="cea")
nphi = int(2*np.pi/(args.res*utils.arcmin))
map_orig = enmap.rand_gauss((ncomp,)+shape, wcs).astype(dtype)
print "map shape %s" % str(map_orig.shape)

pbox = np.array([[0,0],shape],dtype=int)
# define a test tod
bore = np.zeros([nsamp,3],dtype=ptype)
bore[:,0] = (wt*np.linspace(0,1,nsamp,endpoint=False))
bore[:,1] = (args.az + args.waz/2*utils.triangle_wave(np.linspace(0,1,nsamp,endpoint=False)*20))*utils.degree
bore[:,2] = (args.el + args.wel/2*utils.triangle_wave(np.linspace(0,1,nsamp,endpoint=False)))*utils.degree
#bore = (ibox[None,0] + np.random.uniform(0,1,size=(nsamp,3))*(ibox[1]-ibox[0])[None,:]).astype(ptype)
tod = np.zeros((ndet,nsamp),dtype=dtype)
psi = np.arange(nsamp)*2*np.pi/100
hwp = np.zeros([nsamp,2])
hwp[:,0] = np.cos(psi)
hwp[:,1] = np.sin(psi)
if not args.hwp: hwp[0] = 0

transfun = hor2pix(shape, wcs, t0)

#moo = transfun(bore.T)
#sys.stdout.flush()
#np.savetxt("/dev/stdout", moo[:2,::100].T, "%9.2f")
#sys.stdout.flush()

errlim   = np.array([0.01, 0.01, utils.arcmin, utils.arcmin])*acc
ipfun    = interpol.ip_ndimage
if args.interpolator == "all":
	ipnames = [
			#"fast",
			"std_bi_0","std_bi_1","std_bi_3"]
else:
	ipnames = args.interpolator.split(",")
pix = None

for dir in dirs:
	for ipname in ipnames:
		# Precompute pointing
		if ipname.split("_")[0] in ["std","cf"]:
			t1 = time.time()
			corners = utils.box2corners(wibox).T
			ocorners = transfun(corners)
			ipol, obox, ok, err = interpol.build(transfun, ipfun, wibox, errlim,
					maxsize=max_size, maxtime=max_time, return_obox=True, return_status=True, order=1)
			t2 = time.time()
			tbuild = t2-t1
			# evaluate accuracy
			pos_exact = transfun(bore.T)
			pos_inter = ipol(bore.T)
			rbox, nbox, yvals = pmat.extract_interpol_params(ipol, ptype)
		else:
			t1 = time.time()
			poly = pmat.PolyInterpol(transfun, bore, det_pos)
			tbuild = time.time()-t1
			pos_exact = transfun((bore + det_pos[0]).T)
			pos_inter = poly(bore, [0])[0]
			ok   = True
			nbox = [ndet, 11]
		err  = np.max(np.abs(pos_exact-pos_inter)/errlim[:,None])*acc
		err2 = np.max(np.std(pos_exact-pos_inter,1)/errlim)*acc

		# Set up arrays
		map = map_orig.copy()
		tod = np.arange(tod.size,dtype=dtype).reshape(tod.shape)*1e-8

		# Precompute pixels if necessary
		tpre = 0
		if ipname == "fast":
			pix    = np.zeros([ndet,nsamp],np.int32)
			phase  = np.zeros([ndet,nsamp,2],dtype)
			sdir   = pmat.get_scan_dir(bore[:,1])
			period = pmat.get_scan_period(bore[:,1], srate)
			wbox, wshift = pmat.build_work_shift(transfun, wibox, period)
			t1 = time.time()
			core.pmat_map_get_pix_poly_shift(pix.T, phase.T, bore.T, hwp.T, det_comps.T,
				poly.coeffs.T, sdir, wbox.T, wshift.T)
			t2 = time.time()
			tpre = t2-t1

		t1 = time.time()
		times = np.zeros(5, dtype=ptype)
		iptoks = ipname.split("_")
		for i in range(args.ntime):
			if ipname == "fast":
				core.pmat_map_use_pix_shift(dir, tod.T, 1, map.T, 1, pix.T, phase.T, wbox.T, wshift.T, nphi, times)
			elif iptoks[0] == "std":
				pmet = {"bi":1,"gr":2}[iptoks[1]]
				mmet = int(iptoks[2])
				core.pmat_map_direct_grid(dir, tod.T, 1, map.T, 1, pmet, mmet, bore.T, hwp.T, det_pos.T, det_comps.T,
					rbox.T, nbox, yvals.T, pbox.T, nphi, times)
			elif iptoks[0] == "cf":
				pmet = {"bi":1,"gr":2}[iptoks[1]]
				mmet = int(iptoks[2])
				core.pmat_map_direct_grid_cf(dir, tod.T, 1, map.T, 1, pmet, mmet, bore.T, hwp.T, det_pos.T, det_comps.T,
					rbox.T, nbox, yvals.T, pbox.T, nphi, times)
		t2 = time.time()
		tuse = (t2-t1)/args.ntime
		times /= args.ntime
		if dir > 0:
			val = np.sum(tod**2)
		else:
			val = np.sum(map**2)
		print "ip %-14s dir %2d tb %6.4f ok %d size %5.3f M acc %5.2f %5.2f t %5.3f: %5.3f %5.3f %5.3f %5.3f v %13.7e %5.3f" % ((
				ipname, dir, tbuild, ok, np.product(nbox)*1e-6, err, err2, tuse) + tuple(times[1:]) + (val,tpre))
