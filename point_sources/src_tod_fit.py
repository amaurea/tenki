# TOD-level point source fitting. This works, and seems to be quite a bit better than
# my filtered map-level approach. For example, see this comparison for a high-S/N source
# in 1378883069.1378883122.ar1:
#  1.4179  0.0209 -3.8517  0.0303  9.7507 26.7
#  1.435          -3.864          13.217  38.4
# So that's a 44% higher S/N, corresponding to 2x more data. Not completely comparable, though,
# since the TOD-one didn't marginalize over position. The position also differs by almost a sigma,
# which it shouldn't considering that they share data. And I trust the tod-level one more.
# However, this takes 1-5 s per likelihood evaluation. A robust fit requires ~500 evaluations,
# which would be 8-42 minutes. And that's using 16 cores! That's too slow. So this one is
# useful for comparing with a faster methods for a few reference tods, but not in general.
# Currently N and P take similar time. Can optimize P more with some effort, but N is dominated
# by ffts, and can't improve much.
from __future__ import division, print_function
import numpy as np, time, astropy.io.fits, os, sys
from scipy import optimize, integrate
from enlib import utils
with utils.nowarn(): import h5py
from enlib import mpi, errors, fft, mapmaking, config, jointmap, pointsrcs
from enlib import pmat, coordinates, enmap, bench, bunch, nmat, sampcut, gapfill, wcsutils, array_ops
from enact import filedb, actdata, actscan, nmat_measure

config.set("downsample", 1, "Amount to downsample tod by")
config.set("gapfill", "linear", "Gapfiller to use. Can be 'linear' or 'joneig'")
config.default("pmat_interpol_pad", 10.0, "Number of arcminutes to pad the interpolation coordinate system by")
config.default("pmat_interpol_max_size", 4000000, "Maximum mesh size in pointing interpolation. Worst-case time and memory scale at most proportionally with this.")

parser = config.ArgumentParser(os.environ["HOME"] + "./enkirc")
parser.add_argument("mode", help="Mode to use. Can be srcs or planet. This sets up useful defaults for other arguments")
parser.add_argument("srcdb_or_planet")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("-R", "--radius",    type=float, default=12)
parser.add_argument("-r", "--res",       type=float, default=0.1)
parser.add_argument("-m", "--method",    type=str,   default="fixamp")
parser.add_argument("-M", "--minimaps",  action="store_true")
parser.add_argument("-c", "--cont",      action="store_true")
parser.add_argument("-s", "--srcs",      type=str,   default=None)
parser.add_argument("-A", "--minamp",    type=float, default=None)
parser.add_argument("-v", "--verbose",   action="count", default=0)
parser.add_argument("-q", "--quiet",     action="count", default=0)
parser.add_argument("-p", "--perdet",    type=int,   default=None)
parser.add_argument("-g", "--group",     type=str,   default="tod")
parser.add_argument("-H", "--highpass",  type=float, default=None)
parser.add_argument(      "--minsn",     type=float, default=1)
parser.add_argument(      "--dump-tod",  action="store_true")
parser.add_argument(      "--dump-tod-ndet", type=int, default=8)
parser.add_argument("-G", "--nogrid",    action="store_true")
parser.add_argument(      "--ncomp",     type=int, default=3)
parser.add_argument(      "--force",     type=float, default=None)
args = parser.parse_args()

#config.default("pmat_accuracy", 10.0, "Factor by which to lower accuracy requirement in pointing interpolation. 1.0 corresponds to 1e-3 pixels and 0.1 arc minute in polangle")

def getdef(val, default): return val if val is not None else default

def read_srcs(fname):
	data = pointsrcs.read(fname)
	return np.array([data.ra*utils.degree, data.dec*utils.degree,data.I])

filedb.init()
db      = filedb.scans.select(args.sel)
ids     = db.ids
comm    = mpi.COMM_WORLD
dtype   = np.float64
ndir    = 1
verbose = args.verbose - args.quiet
R       = args.radius*utils.arcmin
res     = args.res*utils.arcmin
poly_pad= 3*utils.degree
grid_bounds = np.array([[-2,-3],[2,5]])*utils.arcmin
grid_res    = 0.75*utils.arcmin
#grid_bounds = np.array([[-3,-3],[3,2]])*utils.arcmin
#grid_res    = 0.25*utils.arcmin
utils.mkdir(args.odir)

if args.highpass: highpass = [args.highpass, 8]
else:             highpass = None

# Set up our mode-dependent arguments
if args.mode == "srcs":
	perdet = getdef(args.perdet, 0)>0
	minamp = getdef(args.minamp, 500)*1.0
	planet = None
	srcdata= read_srcs(args.srcdb_or_planet)
	src_sys= "cel"
	bounds = db.data["bounds"]
	bounds = db.data["bounds"]
	prune_unreliable_srcs = True
elif args.mode == "planet":
	perdet = getdef(args.perdet, 1)>0
	minamp = 0.0
	planet = args.srcdb_or_planet.capitalize()
	# Coordinates are in relative to the planet itself, so it's fiducially at 0,0
	srcdata= np.array([0,0,5000e3])[:,None]
	src_sys= "hor:%s/0_0" % planet
	bounds = None
	prune_unreliable_srcs = False
else:
	print("Unknown mode '%s'" % args.mode)
	sys.exit(1)

### How to handle cuts.
#
# 1. Solve for cuts like we solve for amplitudes. For each
#    solver step, we linearly solve for the cut values.
#    d = [Pamp Pcut] [amp, junk] + n, which has the solution
#    [amp, junk] = (P'N"P)"P'N"d. amp and junk have no sample
#    overlap, but they are still coupled via N". Solving this
#    would be slow.
# 2. Don't solve for the cuts, just simulate the effect they
#    have. If we did linear gapfilling, then we can model
#    our data as d = gapfill(Pa + n). Gapfilling is a linear
#    operation, so this is d = G(Pa + n). The solution for
#    amp is  a = (P'G'(GNG')"GP)"P'G'(GNG')"d. GP is simple,
#    P'G' should also be doable, but (GNG')" is hard.
#    GNG represents the noise properties of the gapfilled TOD.
#    This will be nasty and nonstationary, but we usually just
#    approximate it as stationary and gaussian, and that is actually
#    what we usually mean by N. So we rename GNG' -> N. This leaves
#    us with a = [P'G'N"GP]"P'G'N"d.
#    The only new thing to implement here is G', e.g. the tranpose of
#    gapfilling. With zero gapfilling this is simple (G'=G). But
#    zero gapfilling messes up the GNG' -> N approximation too much.
#    For linear gapfilling G' = 0 for cut samples, ncut/w for context
#    samples and 1 for other samples. Should implement this.

# Amplitude parameters are stored as params[nsrc,ndir,ndet or 1,nparam].
# The amps we fit for will be amps[nsrc,ndir,ndet or 1,npol].
# Easiest to always keep all these axes, and let them be length
# 1 when they aren't relevant. This results in amps always being 4d

class PmatTot:
	def __init__(self, data, srcpos, ndir=1, perdet=False):
		# Build source parameter struct for PmatPtsrc
		self.params = np.zeros([srcpos.shape[-1],ndir,data.ndet if perdet else 1,8],np.float)
		self.params[:,:,:,:2] = srcpos[::-1,None,None,:].T
		self.params[:,:,:,5:7] = 1
		scan = actscan.ACTScan(data.entry, d=data)
		self.psrc = pmat.PmatPtsrc(scan, self.params, sys=src_sys)
		self.pcut = pmat.PmatCut(scan)
		# Extract basic offset
		self.off0 = data.point_correction
		self.off  = self.off0*0
		self.el   = np.mean(data.boresight[2,::100])
		self.point_template = data.point_template
		self.cut = data.cut
	def set_offset(self, off):
		self.off = off*1
		self.psrc.scan.offsets[:,1:] = actdata.offset_to_dazel(self.point_template + off + self.off0, [0,self.el])
	def forward(self, tod, amps, pmul=1):
		# Amps should be [nsrc,ndir,ndet|1,npol]
		params = self.params.copy()
		params[...,2:2+amps.shape[-1]]   = amps
		self.psrc.forward(tod, params, pmul=pmul)
		sampcut.gapfill_linear(self.cut, tod, inplace=True)
	def backward(self, tod, amps=None, pmul=1, ncomp=3):
		params = self.params.copy()
		tod = sampcut.gapfill_linear(self.cut, tod, inplace=False, transpose=True)
		self.psrc.backward(tod, params, pmul=pmul)
		if amps is None: amps = params[...,2:2+ncomp]
		else: amps[:] = params[...,2:2+amps.shape[-1]]
		return amps

class NmatTot:
	def __init__(self, data, model=None, window=None, filter=None):
		model  = config.get("noise_model", model)
		window = config.get("tod_window", window)*data.srate
		nmat.apply_window(data.tod, window)
		self.nmat = nmat_measure.NmatBuildDelayed(model, cut=data.cut_noiseest, spikes=data.spikes[:2].T)
		self.nmat = self.nmat.update(data.tod, data.srate)
		nmat.apply_window(data.tod, window, inverse=True)
		self.model, self.window = model, window
		self.ivar = self.nmat.ivar
		self.cut  = data.cut
		# Optional extra filter
		if filter:
			freq = fft.rfftfreq(data.nsamp, 1/data.srate)
			fknee, alpha = filter
			with utils.nowarn():
				self.filter = (1 + (freq/fknee)**-alpha)**-1
		else: self.filter = None
	def apply(self, tod):
		nmat.apply_window(tod, self.window)
		ft = fft.rfft(tod)
		self.nmat.apply_ft(ft, tod.shape[-1], tod.dtype)
		if self.filter is not None: ft *= self.filter
		fft.irfft(ft, tod, flags=['FFTW_ESTIMATE','FFTW_DESTROY_INPUT'])
		nmat.apply_window(tod, self.window)
		return tod
	def white(self, tod):
		nmat.apply_window(tod, self.window)
		self.nmat.white(tod)
		nmat.apply_window(tod, self.window)

class PmatThumbs:
	def __init__(self, data, srcpos, res=0.25*utils.arcmin, rad=20*utils.arcmin, perdet=False, detoff=10*utils.arcmin):
		scan = actscan.ACTScan(data.entry, d=data)
		if perdet:
			# Offset each detector's pointing so that we produce a grid of images, one per detector.
			gside  = int(np.ceil(data.ndet**0.5))
			goffs  = np.mgrid[:gside,:gside] - (gside-1)/2.0
			goffs  = goffs.reshape(2,-1).T[:data.ndet]*detoff
			scan.offsets = scan.offsets.copy()
			scan.offsets[:,1:] += goffs
			rad    = rad + np.max(np.abs(goffs))
		# Build geometry for each source
		shape, wcs = enmap.geometry(pos=[[-rad,-rad],[rad,rad]], res=res, proj="car")
		area = enmap.zeros((3,)+shape, wcs, dtype=data.tod.dtype)
		self.pmats = []
		for i, pos in enumerate(srcpos.T):
			if planet: sys = src_sys
			else:      sys = ["icrs",[np.array([[pos[0]],[pos[1]],[0],[0]]),False]]
			with config.override("pmat_accuracy", 10):
				self.pmats.append(pmat.PmatMap(scan, area, sys=sys))
		self.shape = (len(srcpos.T),3)+shape
		self.wcs   = wcs
	def forward(self, tod, map):
		for i, p in enumerate(self.pmats):
			p.forward(tod, map[i])
	def backward(self, tod, map):
		for i, p in enumerate(self.pmats):
			p.backward(tod, map[i])

class ThumbMapper:
	def __init__(self, data, srcpos, pcut, nmat, perdet=False):
		pthumb = PmatThumbs(data, srcpos, perdet=perdet)
		twork  = np.full(data.tod.shape, 1.0, data.tod.dtype)
		nmat.white(twork)
		div   = enmap.zeros(pthumb.shape, pthumb.wcs, data.tod.dtype)
		junk  = np.zeros(pcut.njunk,data.tod.dtype)
		pcut.backward(twork, junk)
		pthumb.backward(twork, div)
		div = div[:,0]
		self.pthumb, self.pcut, self.nmat = pthumb, pcut, nmat
		self.div = div
		with utils.nowarn():
			self.idiv = 1/self.div
			self.idiv[~np.isfinite(self.idiv)] = 0
	def map(self, tod):
		junk = np.zeros(self.pcut.njunk,tod.dtype)
		rhs  = enmap.zeros(self.pthumb.shape, self.pthumb.wcs, tod.dtype)
		#self.nmat.white(tod)
		self.nmat.apply(tod)
		self.pcut.backward(tod, junk)
		self.pthumb.backward(tod, rhs)
		rhs *= self.idiv[:,None]
		return rhs

# For fixed amplitude, our chisquare is
# chisq = (d-Pa)'N"(d-Pa) = d'N"d + (Pa)'N"(Pa) - 2d'N"Pa
#       = d'N"d - 2*(Pa)'N"(d-Pa/2)
# The derivative of this with respect to position is
# dchisq = -2*d(Pa)'N"(d-Pa/2) + (Pa')N"dPa = 

class Likelihood:
	def __init__(self, data, srcpos, srcamp, perdet=False, ncomp=1, thumbs=False, N=None, method="fixamp", filter=None):
		# Set up fiducial source model. These source parameters
		# are not the same as those we will be optimizing.
		with bench.show("PmatTot"):
			self.P = PmatTot(data, srcpos, perdet=perdet)
		with bench.show("NmatTot"):
			self.N = N if N else NmatTot(data, filter=filter)
		self.tod  = data.tod # might only need the one below
		with bench.show("Nmat apply"):
			self.Nd   = self.N.apply(self.tod.copy())
		self.i    = 0
		# Initial values. We have room for 3 stokes parameters, but for now the input
		# amplitudes are always T-only
		self.amp0   = np.zeros(self.P.params.shape[:-1]+(ncomp,))
		self.amp0[...,0] = srcamp[:,None,None]
		self.off0   = self.P.off0
		self.chisq0 = None
		# These are for internal mapmaking
		self.thumb_mapper = None
		if thumbs:
			with bench.show("ThumbMapper"):
				self.thumb_mapper = ThumbMapper(data, srcpos, self.P.pcut, self.N, perdet=perdet)
		self.amp_unit, self.off_unit = 1e3, utils.arcmin
		# Save samples from the wrapper, so we can use them to estimate uncertainty
		self.samples = bunch.Bunch(offs=[], amps=[], aicovs=[], chisqs=[])
		self.method  = method
		self.ncomp   = ncomp
	#def zip(self, off, amps): return np.concatenate([off/self.off_unit, amps[:,0]/self.amp_unit],0)
	#def unzip(self, x): return x[:2]*self.off_unit, x[2:,None]*self.amp_unit
	def zip(self, off): return off/self.off_unit
	def unzip(self, x): return x*self.off_unit
	def fit_amp(self, off=None, ncomp=None):
		"""Compute the ML amplitude for each point source, along with their covariance.
		This assumes independent source amplitudes. For perdet mapping, this means we may
		need to use detector-diagonal noise. The ncomp argument lets us override the
		ncomp used for the rest of the operations in this class. This can be useful for
		getting polarized amplitudes at the end of a long unpolarized fit."""
		if ncomp is None:   ncomp = self.ncomp
		if off is not None: self.P.set_offset(off)
		rhs = self.P.backward(self.Nd, ncomp=ncomp)
		div = np.zeros(rhs.shape + rhs.shape[-1:], rhs.dtype)
		for comp in range(ncomp):
			work = np.zeros(self.tod.shape, self.tod.dtype)
			amps = rhs*0; amps[...,comp] = 1
			self.P.forward(work, amps)
			self.N.apply(work)
			self.P.backward(work, div[...,comp,:])
		idiv = safe_invert(div)
		amps = amatmul(idiv, rhs)
		return amps, div
	def calc_chisq_fixamp(self, off):
		# Compute (the nonstatic part of) the chisquare for the given offset, while
		# keeping the amplitudes fixed to their fiducial values.
		# We want (d-Pa)'N"(d-Pa), but this suffers from poor accuracy because
		# so many noisy numbers are summed. We don't need d'N"d, so the terms
		# we care about are chisq = a'P'N"Pa - 2d'N"Pa = -2(Pa)'N"(d-Pa/2)
		self.P.set_offset(off)
		Nr = self.tod.copy()
		self.P.forward(Nr, -self.amp0/2, pmul=1)
		self.N.apply(Nr)
		PNr = self.P.backward(Nr, ncomp=self.ncomp)
		chisq = -2 * np.sum(self.amp0*PNr)
		aicov = np.zeros(self.amp0.shape+self.amp0.shape[-1:])
		return chisq, self.amp0, aicov
	def calc_chisq_fixamp_simple(self, off):
		# Computes r'N"r directly in time domain. Numerically lossy
		self.P.set_offset(off)
		r = self.tod.copy()
		self.P.forward(r, -self.amp0)
		Nr = r.copy()
		self.N.apply(Nr)
		chisq = np.sum(r*Nr)
		self.r = r
		self.Nr = Nr
		aicov = np.zeros(self.amp0.shape+self.amp0.shape[-1:])
		return chisq, self.amp0, aicov
	def calc_chisq_fitamp(self, off):
		self.P.set_offset(off)
		ahat, aicov = self.fit_amp()
		chisq = -np.sum(amatmul(aicov,ahat)*ahat)
		return chisq, ahat, aicov
	def calc_chisq_posamp(self, off):
		chisq, ahat, aicov = self.calc_chisq_fitamp(off)
		prior = 0
		positivity = (ahat[...,0]*aicov[...,0,0]**0.5).reshape(-1)
		for p in positivity:
			prior -= 2*jointmap.log_prob_gauss_positive_single(p)
		chisq += prior
		return chisq, ahat, aicov
	def get_chisq_fun(self, method=None):
		if method is None: method = self.method
		if   method == "fitamp": fun = self.calc_chisq_fitamp
		elif method == "posamp": fun = self.calc_chisq_posamp
		else:                    fun = self.calc_chisq_fixamp
		return fun
	def make_thumbs(self, off, amps):
		self.P.set_offset(off)
		tod2 = self.tod*0
		self.P.forward(tod2, amps, pmul=1)
		data   = self.thumb_mapper.map(self.tod.copy())
		model  = self.thumb_mapper.map(tod2)
		resid  = data-model
		thumbs = enmap.samewcs([data,model,resid],data)
		return thumbs
	def chisq_wrapper(self, method=None, thumb_path=None, thumb_interval=0, verbose=True):
		fun = self.get_chisq_fun(method)
		def wrapper(off, full=False):
			t1 = time.time()
			off = self.unzip(off)
			chisq, amps, aicov = fun(off)
			t2 = time.time()
			if self.thumb_mapper and thumb_path and thumb_interval and self.i % thumb_interval == 0:
				thumbs = self.make_thumbs(off, amps)
				enmap.write_map(thumb_path % self.i, thumbs)
			if self.chisq0 is None: self.chisq0 = chisq
			if verbose:
				doff = off/utils.arcmin
				msg = "%4d %8.5f %8.5f" % (self.i,doff[0],doff[1])
				msg += " %12.5e %7.3f" % (self.chisq0-chisq, t2-t1)
				famps  = amps.reshape(-1,amps.shape[-1])
				faicov = aicov.reshape((-1,)+aicov.shape[-2:])
				for i in range(len(famps)):
					nsigma = np.sum(amatmul(faicov[i],famps[i])*famps[i])**0.5
					msg += " %10.3f %8.1f" % (famps[i,0]/self.amp_unit, nsigma)
				print(msg)
				self.samples.offs.append(off)
				self.samples.amps.append(amps)
				self.samples.aicovs.append(aicov)
				self.samples.chisqs.append(chisq)
			self.i += 1
			if not full:
				return chisq
			else:
				return chisq, amps, aicov
		return wrapper
	def estimate_error(self, off, method=None, step=0.1):
		# Simple numerical second derivative in each dimension.
		# chisq = x**2/sigma**2 => ddchisq = 2/sigma**2 => sigma = (2/ddchisq)**0.5
		fun = self.get_chisq_fun(method)
		off  = np.array(off)
		doff = self.off_unit*step
		vmid = fun(off)[0]
		derivs = off*0
		for dim in range(len(off)):
			u = off*0; u[dim] = 1
			v1 = fun(off + u*doff)[0]
			v2 = fun(off - u*doff)[0]
			derivs[dim] = (v2+v1-2*vmid)/doff**2
		sigma = (2/derivs)**0.5
		return sigma
	def get_model(self, off, amps):
		self.P.set_offset(off)
		tod2 = self.tod*0
		self.P.forward(tod2, amps, pmul=1)
		return tod2

def safe_invert(div):
	if div.shape[-1] == 1:
		idiv = div.copy()
		idiv[div==0] = 1
		idiv[div!=0] = 1/idiv[div!=0]
	else:
		idiv = array_ops.eigpow(div, -1, axes=[-2,-1], lim=1e-3, fallback="scalar")
	return idiv

def amatmul(M,v): return np.einsum("...ab,...b->...a",M,v)

def build_grid_geometry(bounds, res):
	"""bounds: [[y1,x1],[y2,x2]] in rads
	res: resolution in rads
	Returns shape, wcs"""
	bounds = np.array(bounds)
	ngrid  = np.round((bounds[1]-bounds[0])/res).astype(int)+1
	wcs = wcsutils.WCS(naxis=2)
	wcs.wcs.cdelt[:] = res/utils.degree
	wcs.wcs.crpix[:] = 1
	wcs.wcs.crval = bounds[0,::-1]/utils.degree
	wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
	return tuple(ngrid), wcs

def eval_offs(L, likwrap, offs):
	res = offs.preflat[0]*0
	oflat = offs.reshape(2,-1)
	rflat = res.reshape(-1)
	for i in range(rflat.size):
		rflat[i] = likwrap(L.zip(oflat[:,i]))
	return res

def rhand_polygon(poly):
	"""Returns True if the polygon is ordered in the right-handed convention,
	where the sum of the turn angles is positive"""
	poly = np.concatenate([poly,poly[:1]],0)
	vecs = poly[1:]-poly[:-1]
	vecs /= np.sum(vecs**2,1)[:,None]**0.5
	vecs = np.concatenate([vecs,vecs[:1]],0)
	cosa, sina = vecs[:-1].T
	cosb, sinb = vecs[1:].T
	sins = sinb*cosa - cosb*sina
	coss = sinb*sina + cosb*cosa
	angs = np.arctan2(sins,coss)
	tot_ang = np.sum(angs)
	return tot_ang > 0

def pad_polygon(poly, pad):
	"""Given poly[nvertex,2], return a new polygon where each vertex has been moved
	pad outwards."""
	sign  = -1 if rhand_polygon(poly) else 1
	pwrap = np.concatenate([poly[-1:],poly,poly[:1]],0)
	vecs  = pwrap[2:]-pwrap[:-2]
	vecs /= np.sum(vecs**2,1)[:,None]**0.5
	vort  = np.array([-vecs[:,1],vecs[:,0]]).T
	return poly + vort * sign * pad

def get_groups(ids, mode):
	if   mode == "tod":
		return [[i] for i, id in enumerate(ids)]
	elif mode == "array":
		# Group tods across arrays
		ids = np.asarray(ids)
		tids,  arr = np.char.partition(ids, ":").T[[0,2]]
		groups = utils.find_equal_groups(tids)
		return groups
	else:
		raise ValueError("Unknown group mode '%s'" % mode)

def get_sids_in_tod(id, src_pos, bounds, ind, isids=None, src_sys="cel"):
	if isids is None: isids = list(range(src_pos.shape[-1]))
	if bounds is not None:
		poly      = bounds[:,:,ind]*utils.degree
		poly[0]   = utils.rewind(poly[0],poly[0,0])
		# bounds are defined in celestial coordinates. Must convert srcpos for comparison
		mjd       = utils.ctime2mjd(float(id.split(".")[0]))
		srccel    = coordinates.transform(src_sys, "cel", src_pos, time=mjd)
		srccel[0] = utils.rewind(srccel[0], poly[0,0])
		poly      = pad_polygon(poly.T, poly_pad).T
		accepted  = np.where(utils.point_in_polygon(srccel.T, poly.T))[0]
		sids      = [isids[i] for i in accepted]
	else:
		sids = isids
	return sids

def get_beam_area(beam):
	r, b = beam
	return integrate.simps(2*np.pi*r*b,r)

def padto(a, n):
	res = np.zeros(a.shape[:-1]+(n,))
	res[...,:a.shape[-1]] = a
	return res

# (d-Pa)'N"(d-Pa) = d'N"d + (Pa)'N"(Pa) - 2*d'N"Pa

# If the point sources are far enough away from each other, then they will
# be indepdendent from each other, and all their amplitudes can be fit in
# a single evaluation. Normally you would do:
#  amps = (P'N"P)" P'N"d
# where you need to evaluate P'N"P via unit vector bashing. But if you know
# that it's diagonal, then you can use a single non-unit vector instead:
# diag(P'N"P ones(nsrc)). But should check how good an approximation this is.
# Intuitively, it's a good approximation if the shadow from one souce doesn't
# touch that from another.
#
# P(pos) = int_amp P(pos,amp|d) damp
#        = K int_amp exp(-0.5*(d-Pa)'N"(d-Pa))
#        = K int_amp exp(-0.5*[
#             a'P'N"P(a-(P'N"P)"P'N"d

# (d-Pa)'N"(d-Pa) = d'N"d - 2d'N"Pa + (Pa)'N"Pa

# Set up our tod groups
groups = get_groups(ids, args.group)

# Load source database
srcpos, amps = srcdata[:2], srcdata[2]
# Which sources pass our requirements?
base_sids  = set(range(amps.size))
base_sids &= set(np.where(amps > args.minamp)[0])
if args.srcs is not None:
	selected = [int(w) for w in args.srcs.split(",")]
	base_sids &= set(selected)
base_sids = list(base_sids)

f = open(args.odir + "/fits_%03d.txt" % comm.rank, "w" if not args.cont else "a")

# Iterate over groups
for gid in range(comm.rank, len(groups), comm.size):
	group = groups[gid]

	if args.cont:
		done = [os.path.exists(args.odir + "/fit_%s.hdf" % ids[ind].replace(":","_")) for ind in group]
		print("done", done)
		for ind in group:
			print(args.odir + "/fit_%s.hdf" % ids[ind].replace(":","_"))
		if all(done): continue

	group_data = []
	# Handle each tod in the group
	for ind in group:
		id   = ids[ind]
		oid  = id.replace(":","_")
		sids = get_sids_in_tod(id, srcpos[:,base_sids], bounds, ind, base_sids, src_sys)
		if len(sids) == 0:
			print("%s has 0 srcs: skipping" % id)
			continue
		try:
			nsrc = len(sids)
			print("%s has %d srcs: %s" % (id,nsrc,", ".join(["%d (%.1f)" % (i,a) for i,a in zip(sids,amps[sids])])))
		except TypeError as e:
			print("Weird: %s" % e)
			print(sids)
			print(amps)
			continue

		# Read the data
		entry = filedb.data[id]
		try:
			data = actdata.read(entry, exclude=["tod"], verbose=verbose)
			data+= actdata.read_tod(entry)
			data = actdata.calibrate(data, verbose=verbose)
			#print("fixme") # FIXME
			#data.restrict(dets=data.dets[100:150])
			# Avoid planets while building noise model
			if planet is not None:
				data.cut_noiseest *= actdata.cuts.avoidance_cut(data.boresight, data.point_offset, data.site, planet, R)
			if data.ndet < 2 or data.nsamp < 1: raise errors.DataMissing("no data in tod")
		except errors.DataMissing as e:
			print("%s skipped: %s" % (id, e))
			continue
		# Prepeare our samples
		#data.tod -= np.mean(data.tod,1)[:,None]
		data.tod -= data.tod[:,None,0].copy()
		data.tod  = data.tod.astype(dtype)
		# Set up our likelihood
		L = Likelihood(data, srcpos[:,sids], amps[sids], filter=highpass)
		# Find out which sources are reliable, so we don't waste time on bad ones
		if prune_unreliable_srcs:
			_, aicov = L.fit_amp()
			good = amps[sids]**2*aicov[:,0,0,0,0] > args.minsn**2
			sids = [sid for sid,g in zip(sids,good) if g]
			nsrc = len(sids)
			print("Restricted to %d srcs: %s" % (nsrc,", ".join(["%d (%.1f)" % (i,a) for i,a in zip(sids,amps[sids])])))
		if nsrc == 0: continue
		L    = Likelihood(data, srcpos[:,sids], amps[sids], perdet=perdet, thumbs=True, N=L.N, method=args.method, filter=highpass)
		beam_area = get_beam_area(data.beam)
		_, uids   = actdata.split_detname(data.dets) # Argh, stupid detnames
		freq      = data.array_info.info.nom_freq[uids[0]]
		fluxconv  = utils.flux_factor(beam_area, freq*1e9)
		group_data.append(bunch.Bunch(data=data, sids=sids, lik=L, id=id, oid=oid, ind=ind, beam_area=beam_area, freq=freq, fluxconv=fluxconv))

	if len(group_data) == 0:
		print("No usable tods in group %s. Skipping" % ",".join([ids[i] for i in group]))
		continue

	# Set up the full likelihood
	progress_thumbs = args.minimaps and verbose >= 3
	chisq_wrappers  = [data.lik.chisq_wrapper(thumb_path=args.odir + "/" + data.oid + "_thumb%03d.fits", thumb_interval=progress_thumbs) for data in group_data]
	def likfun(off): return sum([chisq_wrapper(off) for chisq_wrapper in chisq_wrappers])
	# Representaive individual lik for stuff that's common to them. This is a bit hacky.
	# A single joint lik class would have been cleaner.
	L = group_data[0].lik

	if not args.nogrid:
		# Do a grid search first to avoid false minima
		grid_shape, grid_wcs = build_grid_geometry(grid_bounds, grid_res)
		grid_pos = enmap.posmap(grid_shape, grid_wcs)
		grid_lik = eval_offs(L, likfun, grid_pos)
		enmap.write_map(args.odir + "/grid_%s.fits" % oid, grid_lik)
		x0  = enmap.argmin(grid_lik)
	else:
		x0  = np.zeros(2)

	# Then do the actual likelihood search
	off  = L.unzip(optimize.fmin_powell(likfun,L.zip(x0)))
	# and estimate the position error using a few extra samples
	# Estimate position errors using a few extra samples
	doff = np.sum([data.lik.estimate_error(off)**-2 for data in group_data],0)**-0.5

	# Now that we have the best fit position we can get the amplitudes for
	# each individual tod in the group
	tot_dchisq = 0
	for di, data in enumerate(group_data):
		L = data.lik
		# Evaluate the ampitude at the ML point
		data.oamps, data.oaicov = L.fit_amp(off, ncomp=args.ncomp)
		if args.force:
			data.oamps[0,...,0] = args.force
		#chisq, oamps, oaicov = likfun(off, full=True)
		if args.minimaps:
			thumbs = L.make_thumbs(off, data.oamps)
			enmap.write_map(args.odir + "/" + data.oid + "_thumb.fits", thumbs)
	
		# Estimate our total S/N. We know that our amplitudes should be positive, so
		# degrade the S/N 
		zs = (data.oamps[...,0] * data.oaicov[...,0,0]**0.5).reshape(-1)
		chisqs = np.array([z**2 + 2*jointmap.log_prob_gauss_positive_single(z) for z in zs])
		tot_dchisq += np.sum(chisqs)

	tot_sn  = max(0, tot_dchisq)**0.5

	for di, data in enumerate(group_data):
		L = data.lik

		# Measure a detector's signal-weighted signal strength.
		# This tells us at what strength the detector saw the source
		# in the periods that contributed most to the measurement
		# This is effectively just np.sum(model**2)/np.sum(model)
		adummy = data.oamps*0; adummy[...,0] = 1
		model  = np.zeros(L.tod.shape, L.tod.dtype)
		L.P.forward(model, adummy)
		with utils.nowarn():
			mean_bval = np.sum(model**2,1)/np.sum(model,1)
		mean_bval[~np.isfinite(mean_bval)] = 0
		del model

		# Output our fit to hdf file
		_, uid = actdata.split_detname(data.data.dets) # :(
		row = data.data.array_info.info.row[uid]
		col = data.data.array_info.info.col[uid]
		with h5py.File(args.odir + "/fit_%s.hdf" % data.oid, "w") as ofile:
			ofile["off"]  = off
			ofile["doff"] = doff
			ofile["0fidoff"] = L.P.off0
			ofile["amps"]  = data.oamps
			ofile["aicov"] = data.oaicov
			ofile["bval"]  = mean_bval
			ofile["ivar"] = L.N.ivar
			ofile["dets"] = uid
			ofile["row"]  = row
			ofile["col"]  = col
			ofile["srcs"] = data.sids
			ofile["srcpos"] = srcpos[:,data.sids]
			ofile["fidamp"] = amps[data.sids]
			ofile["beam_area"] = data.beam_area
			ofile["freq"] = data.freq
			ofile["fluxconv"] = data.fluxconv
			ofile["detoff"] = data.data.point_template

		meanamps  = padto(np.mean(data.oamps, (1,2)),3)
		# mean here because we want to show the typical error bars, not the error on the mean
		meandamps = padto(np.einsum("...aa->...a",safe_invert(np.mean(data.oaicov,(1,2))))**0.5,3)
		# Format fit result in standard format
		msg = ""
		for si, sid in enumerate(data.sids):
			sn_T = meanamps[si,0]/meandamps[si,0]
			sn_P = np.sum((meanamps[si,1:]/meandamps[si,1:])**2)**0.5 - (meanamps.shape[-1]-1)
			#       id sid | offx  doffx offy  doffy | fidT  T     dT    Q     dQ    U     dU    snT   snP   sntot | ra    dec   | hour  baz   bel   waz   | fidx   fidy | beam  freq  fconv
			msg += "%s %4d | %8.4f %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f %7.4f %7.4f %7.4f %7.4f %6.2f %6.2f %6.2f | %7.2f %7.2f | %5.2f %7.2f %7.2f %7.2f | %8.4f %8.4f | %8.4f %8.4f %8.4f\n" % (
					data.id, sid,
					off[0]/utils.arcmin, off[1]/utils.arcmin, doff[0]/utils.arcmin, doff[1]/utils.arcmin,
					amps[sid]/1e3, meanamps[si,0]/1e3, meandamps[si,0]/1e3, meanamps[si,1]/1e3, meandamps[si,1]/1e3, meanamps[si,2]/1e3, meandamps[si,2]/1e3, sn_T, sn_P, tot_sn,
					srcpos[0,sid]/utils.degree, srcpos[1,sid]/utils.degree,
					db.data["hour"][ind], db.data["baz"][ind], db.data["bel"][ind], db.data["waz"][ind],
					L.P.off0[0]/utils.arcmin, L.P.off0[1]/utils.arcmin,
					data.beam_area*1e9, data.freq, data.fluxconv)
		print(msg)
		f.write(msg + "\n")
		f.flush()

		if args.dump_tod:
			data_dump  = data.data.tod
			model_dump = L.get_model(off, data.oamps)
			resid_dump = data_dump - model_dump
			best = np.argsort(np.max(model_dump,1))[::-1]
			with h5py.File(args.odir + "/dump_%s.hdf" % data.oid, "w") as ofile:
				ofile["data"]  = data_dump[best[:args.dump_tod_ndet]]
				ofile["model"] = model_dump[best[:args.dump_tod_ndet]]
				ofile["resid"] = resid_dump[best[:args.dump_tod_ndet]]

f.close()
print("%4d done, waiting for others" % comm.rank)
comm.Barrier()
if comm.rank == 0: print("All done")
