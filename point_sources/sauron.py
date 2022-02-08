import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("odir")
parser.add_argument("--snr1",  type=float, default=5)
parser.add_argument("--snr2",  type=float, default=4)
parser.add_argument("--nmat1", type=str, default="constcorr")
parser.add_argument("--nmat2", type=str, default="constcorr")
parser.add_argument("--nmat-smooth", type=str, default="angular")
parser.add_argument("--cmb",   type=str, default=None)
parser.add_argument("--slice", type=str, default=None)
parser.add_argument("--box",   type=str, default=None)
parser.add_argument("-c", "--cont",   action="store_true")
parser.add_argument("-t", "--tshape", type=str, default="500,2000")
parser.add_argument("-C", "--comps",  type=str, default="TQU")
parser.add_argument("-s", "--sim",    type=str, default=None)
args = parser.parse_args()
import numpy as np, time
from numpy.lib import recfunctions
from pixell import enmap, utils, bunch, analysis, uharm, powspec, pointsrcs, curvedsky, mpi
from enlib import mapdata, array_ops, wavelets, multimap
from scipy import ndimage

utils.mkdir(args.odir)
comm = mpi.COMM_WORLD
tshape= np.zeros(2,int)+utils.parse_ints(args.tshape)

class Nmat:
	def matched_filter(self, map, cache=None): raise NotImplementedError

class Finder:
	def __call__(self, map): raise NotImplementedError

class Measurer:
	def __call__(self, map, cat): raise NotImplementedError

class Modeller:
	def __call__(self, cat): raise NotImplementedError

def cache_get(cache, key, op):
	if not cache: return op()
	if key not in cache:
		cache[key] = op()
	return cache[key]

class _get_slice_class:
	def __getitem__(self, a): return a
get_slice = _get_slice_class()

def parse_slice(desc):
	if desc is None: return None
	else: return eval("get_slice" + desc)

def to_cov(M, n, npost=0):
	"""Turn M into a covmat of shape (n,n) (excluding npost post-dimensions).
	If M already has enough dimensions, it is returned as is. If it has 1
	dimension too little, that becomes the diagonal. If it has two dimensions
	too little, then it's broadcasted along the diagonal."""
	res = np.zeros_like(M, shape=(n,n)+M.shape[M.ndim-npost:])
	if    M.ndim == res.ndim: res[:] = M
	elif  M.ndim == res.ndim-1:
		for i in range(n):
			res[i,i] = M[i]
	else:
		for i in range(n):
			res[i,i] = M
	return res

def broadcst_ndim(A, ndim):
	return A[(None,)*(ndim-A.ndim)]

def sanitize_kappa(kappa, tol=1e-4, inplace=False):
	if not inplace: kappa = kappa.copy()
	for i in range(len(kappa)):
		kappa[i,i] = np.maximum(kappa[i,i], np.max(kappa[i,i])*tol)
	return kappa

def solve_mapsys(kappa, rho):
	if kappa.ndim == 2:
		flux  = rho/kappa
		dflux = kappa**-0.5
	else:
		# Check if this is slow
		flux  = enmap.samewcs(np.linalg.solve(kappa.T, rho.T).T, rho)
		dflux = enmap.samewcs(np.einsum("aayx->ayx",np.linalg.inv(kappa.T).T)**0.5, kappa)
	return flux, dflux

def smooth_ps_angular(ps2d, brel=5):
	ps1d, l1d = ps2d.lbin(brel=brel)
	l = ps2d.modlmap()
	return enmap.samewcs(utils.interp(l, l1d, ps1d),ps2d)

def smooth_ps_gauss(ps2d, lsigma=200):
	"""Smooth a 2d power spectrum to the target resolution in l. Simple
	gaussian smoothing avoids ringing."""
	# This hasn't been tested in isolation, but breaks when used in smooth_ps_mixed
	# First get our pixel size in l
	ly, lx = enmap.laxes(ps2d.shape, ps2d.wcs)
	ires   = np.array([ly[1],lx[1]])
	sigma_pix = np.abs(lsigma/ires)
	fmap  = enmap.fft(ps2d)
	ky    = np.fft.fftfreq(ps2d.shape[-2])*sigma_pix[0]
	kx    = np.fft.fftfreq(ps2d.shape[-1])*sigma_pix[1]
	kr2   = ky[:,None]**2+kx[None,:]**2
	fmap *= np.exp(-0.5*kr2)
	return enmap.ifft(fmap).real

def smooth_downup(map, n):
	n = np.minimum(n, map.shape[-2:])
	o = (np.array(map.shape[-2:]) % n)//2
	return enmap.upgrade(enmap.downgrade(map, n, off=o, inclusive=True), n, off=o, oshape=map.shape, inclusive=True)

def smooth_ps_mixed(ps2d, brel=5, lsigma=100):
	# FIXME this doens't work. The noise matrix breaks when
	# inverted after doing this
	enmap.write_map("ps2d_a.fits", ps2d)
	radial = smooth_ps_angular(ps2d, brel=brel)
	enmap.write_map("ps2d_b.fits", radial)
	resid  = ps2d/radial
	enmap.write_map("ps2d_c.fits", resid)
	resid  = smooth_ps_gauss(resid, lsigma=lsigma)
	enmap.write_map("ps2d_d.fits", resid)
	model  = resid*radial
	enmap.write_map("ps2d_e.fits", model)
	return model

def dump_ps1d(fname, ps2d):
	ps, l = ps2d.lbin(brel=2)
	ps = powspec.sym_compress(ps)
	np.savetxt(fname, np.concatenate([l[None], ps],0).T, fmt="%15.7e")

def dtype_concat(dtypes):
	# numpy isn't cooperating, so I'm making this function myself
	return sum([np.dtype(dtype).descr for dtype in dtypes],[])

def merge_arrays(arrays):
	odtype = dtype_concat([a.dtype for a in arrays])
	res    = np.zeros(arrays[0].shape, odtype)
	for a in arrays:
		for key in a.dtype.names:
			res[key] = a[key]
	return res

def rpow(fmap, exp=2):
	"""Given a fourier-space map fmap corresponding to a real map, take it to the given exponent in
	*real space*, and return the fourier-space version of the result."""
	norm = fmap.area()**0.5
	map  = enmap.ifft(fmap/norm+0j, normalize="phys").real
	return enmap.fft (map**exp, normalize="phys").real*norm

def rmul(*args):
	norm = args[0].area()**0.5
	work = None
	for arg in args:
		rmap = enmap.ifft(arg/norm+0j, normalize="phys").real
		if work is None: work  = rmap
		else:            work *= rmap
	return enmap.fft(work, normalize="phys").real*norm

def rop(*args, op=np.multiply):
	norm = args[0].area()**0.5
	return enmap.fft(op(*[enmap.ifft(arg/norm+0j, normalize="phys").real for arg in args]), normalize="phys").real*norm

def snr_split(snrs, sntol=0.25, snmin=5):
	"""Given a list of S/N ratios, split split them into groups that
	can be processed together without interfering with each other.
	Returns [inds1, inds2, inds3, ...], where inds1 has the indices
	of the strongest snrs, inds2 slightly weaker snrs, and so on.
	The weakest element in a group will be at least sntol times the strongest
	element. Values below snmin are bunched into a single group."""
	v  = np.log(np.maximum(np.abs(snrs), snmin))/np.log(1/sntol)
	v -= np.max(v)+1e-9
	v  = utils.floor(v)
	return utils.find_equal_groups(v)[::-1]

def make_invertible(N, mintol=1e-3, corrtol=0.99, inplace=False):
	if not inplace: N = N.copy()
	v     = np.einsum("aa...->a...", N)
	# Get the typical values in the map
	mask  = np.all(v!=0,0)
	if np.sum(mask) == 0: ref = np.zeros(len(N), n.dtype)
	else: ref  = np.mean(v[:,mask],-1)
	# Avoid super-tiny values
	ref   = np.maximum(ref, np.finfo(N.dtype).tiny*1e6)
	# Cap too-low-noise areas
	v     = np.maximum(v.T, ref*mintol).T
	for i in range(len(N)):
		N[i,i] = v[i]
	if corrtol:
		scale = v**0.5
		N    /= scale[:,None]
		N    /= scale[None,:]
		np.clip(N, -corrtol, corrtol, N)
		for i in range(len(N)):
			N[i,i] = 1
		N    *= scale[:,None]
		N    *= scale[None,:]
	return N

# I don't like that these functions take data objects. Those objects are
# temporary container that's handy for this program, but is not meant to
# be reusable.
def build_iN_constcorr_prior(data, cmb=None, lknee0=2000, constcov=False):
	N    = enmap.zeros((data.n,data.n)+data.maps.shape[-2:], data.maps.wcs, data.maps.dtype)
	ref  = np.mean(data.ivars,(-2,-1))
	ref  = np.maximum(ref, np.max(ref)*1e-4)
	norm = 1/(ref/data.ivars.pixsize())
	for i, freq in enumerate(data.freqs):
		lknee  = lknee0*freq/100
		N[i,i] = (1+(np.maximum(0.5,data.l)/lknee)**-3.5) * norm[i]
	# Deconvolve the pixel window from the theoretical flat-at-high-l spectrum
	N  = N / data.wy[:,None]**2 / data.wx[None,:]**2
	# Apply the beam-convolved cmb if available
	if cmb is not None:
		Bf = data.fconvs[:,None,None]*data.beams
		N += cmb*Bf[:,None]*Bf[None,:]
		del Bf
	if not constcov:
		N /= norm[:,None,None,None]**0.5*norm[None,:,None,None]**0.5
	iN = array_ops.eigpow(N, -1, axes=[0,1])
	return iN

def build_iN_constcov_prior(data, cmb=None, lknee0=2000):
	return build_iN_constcorr_prior(data, cmb=cmb, lknee0=lknee0, constcov=True)

def build_iN_constcorr(data, maps, smooth="angular", brel=2, lsigma=500, lmin=100, constcov=False):
	if not constcov: maps = maps*data.ivars**0.5
	fhmap = enmap.map2harm(maps, spin=0, normalize="phys") / maps.pixsize()**0.5
	del maps
	N     = (fhmap[:,None]*np.conj(fhmap[None,:])).real
	N    /= np.maximum(data.fapod, 1e-8)
	del fhmap
	# Smooth in piwwin-space, since things are expected to be more isotopic there
	N = N * data.wy[:,None]**2 * data.wx[None,:]**2
	if smooth == "angular":
		N = smooth_ps_angular(N, brel=brel)
	elif smooth == "gauss":
		N = smooth_ps_gauss(N, lsigma=lsigma)
	elif smooth == "mixed":
		N = smooth_ps_mixed(N, brel=brel, lsigma=lsigma)
	else:
		raise ValueError("Unrecognized smoothing '%s'" % str(smooth))
	N = make_invertible(N, mintol=1e-10, corrtol=0)
	# Restore the pixwin
	N = N / data.wy[:,None]**2 / data.wx[None,:]**2
	# And invert and kill the lowest modes. This is useful because these can have
	# artificially low variance due to the tf
	iN  = np.linalg.inv(N.T).T
	iN[:,:,data.l<lmin] = 0
	return iN

def build_iN_constcov(data, maps, smooth="isotropic", brel=2, lsigma=500, lmin=100):
	return build_iN_constcorr(data, maps, smooth=smooth, brel=brel, lsigma=lsigma, lmin=lmin, constcov=True)

def build_wiN(maps, wt, smooth=5):
	ncomp  = len(maps)
	wnoise = wt.map2wave(maps)
	wiN    = multimap.zeros([geo.with_pre((ncomp,ncomp)) for geo in wnoise.geometries], dtype=maps.dtype)
	for i, m in enumerate(wnoise.maps):
		srad = 2*np.pi/wt.basis.lmaxs[i]*smooth
		Nmap = enmap.smooth_gauss(m[:,None]*m[None,:], srad)
		Nmap = make_invertible(Nmap)
		wiN.maps[i] = np.linalg.inv(Nmap.T).T
	return wiN

def build_wiN_ivar(maps, ivars, wt, smooth=5, tol=1e-4):
	ncomp  = len(maps)
	wnoise = wt.map2wave(maps)
	wivar  = wt.map2wave(ivars, half=True)
	wiN    = multimap.zeros([geo.with_pre((ncomp,ncomp)) for geo in wnoise.geometries], dtype=maps.dtype)
	for i, (m, iv) in enumerate(zip(wnoise.maps, wivar.maps)):
		# Want to estimate smooth behavior with wivar as weight
		srad = 2*np.pi/wt.basis.lmaxs[i]*smooth
		rhs  = enmap.smooth_gauss(m[:,None]*m[None,:]*iv, srad)
		div  = enmap.smooth_gauss(iv, srad)
		div  = np.maximum(div, np.max(div,(-2,-1))[:,None,None]*tol)
		Nmap = rhs/div
		del rhs, div
		Nmap = make_invertible(Nmap)
		wiN.maps[i] = np.linalg.inv(Nmap.T).T
	return wiN

def get_flat_sky_correction(pixratio):
		return (0.5*(1+pixratio**2))**-0.5, 1/pixratio

# Implemented independently of pixell.analysis for now. May backport later.
# Here we only care about flat sky
class NmatConstcov(Nmat):
	def __init__(self, iN, apod):
		"""Initialize a Constcov noise model from an inverse noise power spectrum
		enmap. iN must have shape [n,n,ny,nx]. For a simple scalar filter just
		insert scalar dimensions with None before constructing, e.g.
		iN[None,None]."""
		self.iN      = iN
		self.apod    = apod
		assert self.iN.ndim == 4, "iN   must be an enmap with 4 dims"
		self.pixsize  = enmap.pixsize(self.iN.shape, self.iN.wcs)
		self.pixratio = enmap.pixsizemap(self.iN.shape, self.iN.wcs, broadcastable=True)/self.pixsize
		self.fsky    = enmap.area(self.iN.shape, self.iN.wcs)/(4*np.pi)
	def matched_filter(self, map, beam, cache=None):
		"""Apply a matched filter to the given map [n,ny,nx], which must agree in shape
		with the beam transform beam [n,ny,nx]. Returns rho[n,ny,nx], kappa[n,n,ny,nx].
		From these the best-fit fluxes in pixel y,x can be recovered as
		np.linalg.solve(kappa[:,:,y,x],rho[:,y,x]), and the combined flux as
		rho_tot = np.sum(rho,0); kappa_tot = np.sum(kappa,(0,1)); flux_tot = rho_tot[y,x]/kappa_tot[y,x]"""
		assert map .ndim  == 3, "Map must be an enmap with 3 dims"
		assert beam.ndim  == 3, "Beam must be an enmap with 3 dims"
		assert map .shape == beam.shape, "Map and beam shape must agree"
		# Flat sky corrections. Don't work that well here. Without them
		# we have an up to 2% error with a 10° tall patch. With them this
		# is reduced to 1%. I don't understand why they don't work better here.
		flatcorr_rho, flatcorr_kappa = get_flat_sky_correction(self.pixratio)
		rho = cache_get(cache, "rho_pre", lambda: enmap.map_mul(self.iN,enmap.map2harm(map*self.apod, spin=0, normalize="phys"))/self.pixsize)
		rho = enmap.map2harm_adjoint(beam*rho, spin=0, normalize="phys")*flatcorr_rho
		kappa0 = np.sum(beam[:,None]*self.iN[:,:]*beam[None,:],(-2,-1))/(4*np.pi*self.fsky)
		kappa  = np.empty_like(rho, shape=kappa0.shape+rho.shape[-2:])
		kappa[:] = kappa0[:,:,None,None]*flatcorr_kappa
		# Done! What we return will always be [n,ny,nx], [n,n,ny,nx]
		return rho, kappa

#class NmatConstcorr_old(Nmat):
#	def __init__(self, iN, ivar):
#		"""Initialize a Constcov noise model from an inverse noise power spectrum
#		enmap. iN must have shape [n,n,ny,nx]. For a simple scalar filter just
#		insert scalar dimensions with None before constructing, e.g.
#		iN[None,None]."""
#		self.iN      = iN
#		self.ivar    = ivar
#		assert self.iN  .ndim == 4, "iN   must be an enmap with 4 dims"
#		assert self.ivar.ndim == 3, "ivar must be an enmap with 3 dims"
#		self.pixarea = enmap.pixsizemap(self.iN.shape, self.iN.wcs, broadcastable=True)
#		#self.pixarea = enmap.pixsize(self.iN.shape, self.iN.wcs)
#	def matched_filter(self, map, beam, beam2=None, cache=None):
#		"""Apply a matched filter to the given map [n,ny,nx], which must agree in shape
#		with the beam transform beam [n,ny,nx]. Returns rho[n,ny,nx], kappa[n,n,ny,nx].
#		From these the best-fit fluxes in pixel y,x can be recovered as
#		np.linalg.solve(kappa[:,:,y,x],rho[:,y,x]), and the combined flux as
#		rho_tot = np.sum(rho,0); kappa_tot = np.sum(kappa,(0,1)); flux_tot = rho_tot[y,x]/kappa_tot[y,x]"""
#		assert map.ndim == 3, "Map must be an enmap with 3 dims"
#		assert beam.ndim == 3, "Beam must be an enmap with 3 dims"
#		assert map.shape == beam.shape, "Map and beam shape must agree"
#		V    = self.ivar**0.5
#		# Square the beam in real space if not provided
#		if beam2 is None: beam2 = rpow(beam, 2)
#		# Find a white approximation for iN. Is doing this element-wise correct?
#		iN_white = np.sum(beam[:,None]*self.iN*beam[None,:],(-2,-1))/np.sum(beam[:,None]*beam[None,:],(-2,-1))
#		# The actual filter
#		rho   = cache_get(cache, "rho_pre", lambda: enmap.harm2map_adjoint(V*enmap.map2harm_adjoint(enmap.map_mul(self.iN, enmap.map2harm(V*map, spin=0, normalize="phys")), spin=0, normalize="phys"), spin=0, normalize="phys")/self.pixarea)
#		rho   = enmap.harm2map(beam*rho, spin=0, normalize="phys")
#
#		kappa = enmap.map2harm_adjoint(enmap.map_mul(beam2,enmap.harm2map_adjoint(self.ivar+0j, spin=0, normalize="phys")), spin=0, normalize="phys")/self.pixarea
#		kappa = np.maximum(kappa,0)**0.5
#		kappa = kappa[:,None]*iN_white[:,:,None,None]*kappa[None,:]
#		# Done! What we return will always be [n,ny,nx], [n,n,ny,nx]
#		return rho, kappa

class NmatConstcorr(Nmat):
	def __init__(self, iN, ivar):
		"""Initialize a Constcov noise model from an inverse noise power spectrum
		enmap. iN must have shape [n,n,ny,nx]. For a simple scalar filter just
		insert scalar dimensions with None before constructing, e.g.
		iN[None,None]."""
		self.iN      = iN
		self.ivar    = ivar
		assert self.iN  .ndim == 4, "iN   must be an enmap with 4 dims"
		assert self.ivar.ndim == 3, "ivar must be an enmap with 3 dims"
		self.pixsize  = enmap.pixsize(self.iN.shape, self.iN.wcs)
		self.pixratio = enmap.pixsizemap(self.iN.shape, self.iN.wcs, broadcastable=True)/self.pixsize
		#self.pixarea = enmap.pixsize(self.iN.shape, self.iN.wcs)
	def matched_filter(self, map, beam, beam2=None, cache=None):
		"""Apply a matched filter to the given map [n,ny,nx], which must agree in shape
		with the beam transform beam [n,ny,nx]. Returns rho[n,ny,nx], kappa[n,n,ny,nx].
		From these the best-fit fluxes in pixel y,x can be recovered as
		np.linalg.solve(kappa[:,:,y,x],rho[:,y,x]), and the combined flux as
		rho_tot = np.sum(rho,0); kappa_tot = np.sum(kappa,(0,1)); flux_tot = rho_tot[y,x]/kappa_tot[y,x]"""
		assert map.ndim == 3, "Map must be an enmap with 3 dims"
		assert beam.ndim == 3, "Beam must be an enmap with 3 dims"
		assert map.shape == beam.shape, "Map and beam shape must agree"
		V    = self.ivar**0.5
		# Square the beam in real space if not provided
		if beam2 is None: beam2 = rpow(beam, 2)
		# Find a white approximation for iN. Is doing this element-wise correct?
		iN_white = np.sum(beam[:,None]*self.iN*beam[None,:],(-2,-1))/np.sum(beam[:,None]*beam[None,:],(-2,-1))
		# Our model that our map contains a single point source + noise,
		#  m = BPa+n
		# where a is the total flux in that pixel, P is something that puts all that flux in the correct
		# pixel, and B is our beam, which preserves total flux but smears it out. So B[l=0] = 1. This means
		# that P should have units of 1/pixarea so that Pa has units of flux per steradian.
		#
		# Our ML estimate of a is then
		#  a = (P'B'N"BP)"P'B'N"m = rhs/kappa
		#  rhs = P'B'N"m, kappa = P'B'N"BP
		# In real space, each column of B would be a set of numbers that sum to 1. Near the equator
		# these would be more concentrated, further away they would be broader. The peak height would
		# be position-independent. The pixel size wouldn't enter here (aside from deciding how many
		# pixels are hit) since the flux density is an intensive quantity.
		#
		# Regardless of projection B(x,y) would look like b(dist(x,y)), which is symmetric.
		#
		# How does the flat sky approximation enter here? Affects both B and N, but N doesn't affect
		# the flux expectation value, so let's ignore it for now. Flat sky means that we replace B
		# with B2(x,y) = b(dist0(x,y)), dist0(x,y) = ((x.ra-y.ra)**2*cos_dec + (x.dec-y.dec)**2)**0.5.
		#
		# What is (P'B2'B2P)"P'B2'BP? 
		#
		# BP  = a beam map centered on some location dec with total flux of 1. In car it's stretched hor by 1/cos(dec)
		# B2P = similar, but stretched hor by 1/cos(dec0).
		# (B2P)'(BP) is the dot product of these. Let's try for a gaussian:
		# int 1/sqrt(2pi s1**2) 1/sqrt(2pi s2**2) exp(-(x*s1)**2) exp(-(x*s2)**2) dx =
		# int 1/... exp(-(x*(s1**2+s2**2))) dx = sqrt(2 pi (s1**2+s2**2))/sqrt(2 pi s1**2)/sqrt(2 pi s2**2)
		# = 1/sqrt(2 pi) * sqrt(s1**-2+s2**-2)
		# What it should have been:
		# 1/sqrt(2 pi) * sqrt(2 s1**-2)
		#
		# So (P'B2'B2P)"P'B2'BP = sqrt(s1**-2+s2**-2)/sqrt(s2**-2+s2**-2)
		# where it should have been 1 if B2 were B. So we're off by a factor
		# sqrt(cos(dec)**-2 + cos(dec0)**-2)/(2 cos(dec0)**-2)
		#
		# Hm, here I assumed that the normalization was different for the two cases,
		# but is it? No, we have the same peak in all cases. The flat sky approximation
		# only affects how we compute distances. So let's try again:
		#
		#   int exp(-0.5*(x*s1)**2)*exp(-0.5*(x*s2)**2) dx
		# = int exp(-0.5*x**2*(s1**2+s2**2))
		# = sqrt(2*pi*(s1**2+s2**2))
		#
		# (P'B2'B2P)"P'B2'P = sqrt(s1**2+s2**2)/sqrt(s2**2+s2**2)
		# For my case that means we make a flux error of
		# sqrt(cos(dec)**2+cos(dec0))/sqrt(2 cos(dec0)**2)
		# = sqrt(0.5*(1+cos(dec)**2/cos(dec0)**2))
		#
		# This depends on it being gaussian, but let's just use it for now.
		# What happens to rho and kappa separately?
		# rho is wrong by sqrt(s1**2+s2**2)/sqrt(2*s1**2)
		# = sqrt(cos(dec)**2+cos(dec0)**2)/sqrt(2*cos(dec)**2)
		# = sqrt(0.5*(1+(cos(dec)/cos(dec0))**-2))
		# kappa is wrong by sqrt(2 s2**2)/sqrt(2 s1**2) = (cos(dec)/cos(dec0))**-1
		#
		# But wait! We've forgotten about P! P should go as 1/pixsizemap, but
		# in the flat sky it goes as pixsize. So we should keep P and P2 separate.
		#
		# Let's try again:
		#
		# rho error is:
		#  rho2/rho = P2'B2'BP/(P'B'BP) = sqrt(0.5*(1+(cos(dec)/cos(dec0))**-2)) * area/area0
		#           = sqrt(0.5*(1+(area/area0)**-2)) * (area/area0)
		#           = sqrt(0.5*(1+(area/area0)**2))
		# kappa error is
		#  kappa2/kappa = (P2'B2'B2P2)'/(P'B'BP) = (cos(dec)/cos(dec0))**-1 * (area0/area)**-2
		#           = area/area0
		# Flat-sky correction factors. For a 10° tall patch we get up to 2% errors in flux without
		# them. This is reduced to 1% with them. Not sure why it doesn't do better - is the
		# gaussian approximation too limiting? Or do I have a bug?
		flatcorr_rho, flatcorr_kappa = get_flat_sky_correction(self.pixratio)
		# Numerator
		rho   = cache_get(cache, "rho_pre", lambda: enmap.map2harm(flatcorr_rho*V*enmap.harm2map(enmap.map_mul(self.iN, enmap.map2harm(V*map, spin=0, normalize="phys")), spin=0, normalize="phys"), spin=0, normalize="phys")/self.pixsize)
		rho   = enmap.harm2map(beam*rho, spin=0, normalize="phys")
		# Denominator
		kappa = enmap.harm2map(enmap.map_mul(beam2,enmap.map2harm(self.ivar+0j, spin=0, normalize="phys")), spin=0, normalize="phys")/self.pixsize * flatcorr_kappa
		kappa = np.maximum(kappa,0)**0.5
		kappa = kappa[:,None]*iN_white[:,:,None,None]*kappa[None,:]
		# Done! What we return will always be [n,ny,nx], [n,n,ny,nx]
		return rho, kappa

# I'm getting a -1-2% sigma bias in flux and a +1-2% sigma bias in dflux, leading to a
# -2.5% sigma bias in S/N. Assuming constcorr is right. Should check with sims.
class NmatWavelet(Nmat):
	def __init__(self, wt, wiN):
		self.wt   = wt
		self.wiN  = wiN
	def matched_filter(self, map, beam, cache=None):
		# We get 2% flat-sky errors with a 10° tall patch without corrections, and 1% with the corrections.
		pixsize  = enmap.pixsize(map.shape, map.wcs)
		pixratio = enmap.pixsizemap(map.shape, map.wcs, broadcastable=True)/pixsize
		flatcorr_rho, flatcorr_kappa = get_flat_sky_correction(pixratio)
		# Get rho
		rho = cache_get(cache, "rho_pre", lambda: enmap.map2harm(self.wt.wave2map(multimap.map_mul(self.wiN, self.wt.map2wave(map))), spin=0, normalize="phys")/pixsize)
		rho = enmap.harm2map(beam*rho, spin=0, normalize="phys")*flatcorr_rho
		# Then get kappa
		fkappa = enmap.zeros(self.wiN.pre + map.shape[-2:], map.wcs, utils.complex_dtype(map.dtype))
		for i in range(self.wt.nlevel):
			sub_Q  = self.wt.filters[i]*enmap.resample_fft(beam, self.wt.geometries[i][0], norm=None, corner=True)
			# Is it right to do this component-wise?
			sub_Q2 = rop(sub_Q, op=lambda a: a[:,None]*a[None,:])
			fsmall = sub_Q2*enmap.fft(self.wiN.maps[i], normalize=False)/self.wiN.npixs[i]
			enmap.resample_fft(fsmall, map.shape, fomap=fkappa, norm=None, corner=True, op=np.add)
		kappa = enmap.ifft(fkappa, normalize=False).real/pixsize*flatcorr_kappa
		return rho, kappa

class FinderSimple(Finder):
	def __init__(self, nmat, beam, scaling=1, save_snr=False):
		self.beam   = beam
		self.nmat   = nmat
		self.scaling= scaling
		self.order  = 3
		self.grow   = 0.75*utils.arcmin
		self.grow0  = 20*utils.arcmin
		self.save_snr = save_snr
		self.snr    = None
	def __call__(self, map, snmin=5, snrel=None, penalty=None):
		assert map.ndim == 3, "Map must be an enmap with 3 dims"
		ncomp = len(map)
		dtype = [("ra","d"),("dec","d"),("snr","d"),("flux_tot","d"),("dflux_tot","d"),
				("flux","d",(ncomp,)),("dflux","d",(ncomp,))]
		if penalty is None: penalty = 1
		# Apply the matched filter
		rho, kappa = self.nmat.matched_filter(map, self.beam)
		kappa     = sanitize_kappa(kappa)
		scaling   = np.zeros(len(rho),rho.dtype)+self.scaling
		rho_tot   = np.sum(rho*scaling[:,None,None],0)
		# Build the total detection significance and find peaks
		kappa_tot = np.sum(scaling[:,None,None,None]*kappa*scaling[None,:,None,None],(0,1))
		snr_tot   = rho_tot/kappa_tot**0.5
		# Find the effective S/N threshold, taking into account any position-dependent
		# penalty and the (penalized) maximum value in the map
		if snrel   is not None: snmin = max(snmin, np.max(snr_tot/penalty)*snrel)
		snlim = snmin*penalty
		# Detect objects, and grow them a bit
		labels, nlabel = ndimage.label(snr_tot >= snlim)
		cat            = np.zeros(nlabel, dtype).view(np.recarray)
		if nlabel == 0: return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)
		labels = enmap.samewcs(labels, map)
		dists, labels = labels.labeled_distance_transform(rmax=self.grow0)
		labels *= dists <= self.grow
		allofthem = np.arange(1,nlabel+1)
		if len(cat) == 0: return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)
		# Find the position and snr of each object
		pixs    = np.array(ndimage.center_of_mass(snr_tot**2, labels, allofthem)).T
		cat.ra, cat.dec = map.pix2sky(pixs)[::-1]
		cat.snr = ndimage.maximum(snr_tot, labels, allofthem)
		del labels
		# Interpolating before solving is faster, but inaccurate. So we do the slow thing.
		# First get the total flux and its uncertainty.
		flux_tot, dflux_tot = solve_mapsys(kappa_tot, rho_tot)
		del rho_tot, kappa_tot
		cat.flux_tot        = flux_tot .at(pixs, unit="pix", order=self.order)
		cat.dflux_tot       = dflux_tot.at(pixs, unit="pix", order=0)
		del flux_tot, dflux_tot
		# Then get the per-freq versions
		flux, dflux = solve_mapsys(kappa, rho)
		del rho, kappa
		cat.flux    = flux .at(pixs, unit="pix", order=self.order).T
		cat.dflux   = dflux.at(pixs, unit="pix", order=0).T
		del flux, dflux
		# Hack
		if self.save_snr and self.snr is None: self.snr = snr_tot
		# Sort by SNR and return
		cat = cat[np.argsort(cat.snr)[::-1]]
		return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)

class FinderMulti(Finder):
	def __init__(self, nmat, beams, scalings, save_snr=False):
		self.nmat     = nmat
		self.beams    = beams
		if scalings is None:
			scalings = np.ones(len(beams))
		self.scalings = scalings
		self.order    = 3
		self.grow     = 0.75*utils.arcmin
		self.grow0    = 20*utils.arcmin
		self.save_snr = save_snr
		self.snr      = None
	def __call__(self, map, snmin=5, snrel=None, penalty=None):
		assert map.ndim == 3, "Map must be an enmap with 3 dims"
		ncase = len(self.beams)
		ncomp = len(map)
		dtype = [("ra","d"),("dec","d"),("snr","d"),("flux_tot","d"),("dflux_tot","d"),
				("flux","d",(ncomp,)),("dflux","d",(ncomp,)),("case","i")]
		if penalty is None: penalty = 1
		# Apply the matched filter for each profile and keep track of the best beam
		# in each pixel.
		cache = {}
		snr_tot, rho, kappa, rho_tot, kappa_tot, case = None, None, None, None, None, None
		for ca, (beam, scaling) in enumerate(zip(self.beams, self.scalings)):
			def f():
				rho, kappa = self.nmat.matched_filter(map, beam, cache=cache)
				kappa      = sanitize_kappa(kappa)
				return rho, kappa
			my_rho, my_kappa = cache_get(cache, "beam:"+str(id(beam)), f)
			my_rho_tot    = np.sum(my_rho*scaling[:,None,None],0)
			my_kappa_tot  = np.sum(scaling[:,None,None,None]*my_kappa*scaling[None,:,None,None],(0,1))
			my_snr_tot    = my_rho_tot/my_kappa_tot**0.5
			if snr_tot is None:
				cases = enmap.full(my_snr_tot.shape, my_snr_tot.wcs, ca, np.int8)
				snr_tot, rho, kappa, rho_tot, kappa_tot= my_snr_tot, my_rho, my_kappa, my_rho_tot, my_kappa_tot
			else:
				mask = my_snr_tot > snr_tot
				cases      = enmap.samewcs(np.where(mask, ca,           cases),      map)
				snr_tot    = enmap.samewcs(np.where(mask, my_snr_tot,   snr_tot),   map)
				rho        = enmap.samewcs(np.where(mask, my_rho,       rho),       map)
				kappa      = enmap.samewcs(np.where(mask, my_kappa,     kappa),     map)
				rho_tot    = enmap.samewcs(np.where(mask, my_rho_tot,   rho_tot),   map)
				kappa_tot  = enmap.samewcs(np.where(mask, my_kappa_tot, kappa_tot), map)
			del my_rho, my_kappa, my_rho_tot, my_kappa_tot, my_snr_tot
		del cache
		# Find the effective S/N threshold, taking into account any position-dependent
		# penalty and the (penalized) maximum value in the map
		if snrel   is not None: snmin = max(snmin, np.max(snr_tot/penalty)*snrel)
		snlim = snmin*penalty
		labels, nlabel = ndimage.label(snr_tot >= snlim)
		cat            = np.zeros(nlabel, dtype).view(np.recarray)
		if nlabel == 0: return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)
		# grow labels
		labels = enmap.samewcs(labels, map)
		dists, labels = labels.labeled_distance_transform(rmax=self.grow0)
		labels *= dists <= self.grow
		allofthem = np.arange(1,nlabel+1)
		if len(cat) == 0: return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)
		# Find the position and snr of each object. This is a bit questionable since
		# only the central pixel is guaranteed to belong to relevant case. We therefore
		# check that the center-of-mass location has the same case, and if it doesn't,
		# we fall back on just using the pixel center. This is an acceptable solution,
		# but not a good one. It means that fluxes and positions will be inaccurate for
		# the cases where we do end up taking the fallback. Hopefully those should be
		# pretty weak cases anyway. I think a better solution would require the individual
		# case information, e.g. via a two-pass approach or by saving per-case flux, dflux,
		# snr etc. maps.
		pixs    = np.array(ndimage.center_of_mass(snr_tot**2, labels, allofthem)).T
		pixs0   = np.array(ndimage.maximum_position(snr_tot, labels, allofthem)).T

		cat.snr = ndimage.maximum(snr_tot, labels, allofthem)
		# Interpolating before solving is faster, but inaccurate. So we do the slow thing.
		flux_tot, dflux_tot = solve_mapsys(kappa_tot, rho_tot)
		case0    = cases.at(pixs0, unit="pix", order=0)
		case_com = cases.at(pixs,  unit="pix", order=0)
		flux0    = flux_tot.at(pixs0, unit="pix", order=0)
		flux_com = flux_tot.at(pixs,  unit="pix", order=self.order)
		unsafe   = (case_com != case0) | (np.abs((flux_com-flux0))/np.maximum(np.abs(flux_com),np.abs(flux0)) > 0.2)
		# Build the total part of the catalog
		cat.ra, cat.dec = map.pix2sky(np.where(unsafe, pixs0, pixs))[::-1]
		cat.case      = np.where(unsafe, case0, case_com)
		cat.flux_tot  = np.where(unsafe, flux0, flux_com)
		cat.dflux_tot = dflux_tot.at(np.where(unsafe, pixs0, pixs), unit="pix", order=0)
		del flux_tot, dflux_tot
		# Then get the per-freq versions
		flux, dflux = solve_mapsys(kappa, rho)
		del rho, kappa
		cat.flux    = np.where(unsafe, flux.at(pixs0, unit="pix", order=0), flux.at(pixs, unit="pix", order=self.order)).T
		cat.dflux   = dflux.at(np.where(unsafe, pixs0, pixs), unit="pix", order=0).T
		del flux, dflux
		# Hack
		if self.save_snr and self.snr is None: self.snr = snr_tot
		# Sort by SNR and return
		cat = cat[np.argsort(cat.snr)[::-1]]
		return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)

class FinderIterative(Finder):
	def __init__(self, finder, modeller, maxiter=10, sntol=0.50,
			grid_max=4, grid_res=0.1*utils.degree, grid_dominance=10):
		self.finder   = finder
		self.modeller = modeller
		self.maxiter  = maxiter
		self.sntol    = sntol
		self.grid_max = grid_max
		self.grid_res = grid_res
		self.grid_dominance = grid_dominance
	def __call__(self, map, snmin=5, snrel=None, verbose=False):
		cat      = []
		own_vals = []
		model = map*0
		# Build a grid that's used to penalize areas with too many detections
		if self.grid_max:
			box  = utils.widen_box(map.corners(),2*self.grid_res,relative=False)
			cgeo = enmap.geometry(box, res=self.grid_res)
			hits = enmap.zeros(*cgeo, np.int32)
		else:
			hits = None
		for i in range(self.maxiter):
			sntol = self.sntol
			if self.grid_max:
				hits_big = hits.project(*map.geometry, order=0)
				penalty  = enmap.samewcs(np.where(hits_big > self.grid_max, 1000, 1).astype(map.dtype), map)
			else: penalty = None
			res = self.finder(map-model, snmin=snmin, snrel=sntol, penalty=penalty)
			if verbose:
				print("it %d snmin %5.1f nsrc %6d tot %6d" % (i+1, res.snmin, len(res.cat), len(res.cat)+sum([len(c) for c in cat])))
			if len(res.cat) == 0: break
			# Measure own contribution to peak pixel
			model1 = self.modeller(res.cat)
			own_vals.append(model1.at([res.cat.dec,res.cat.ra]).T)
			# update total model
			model += model1; del model1
			# Optionally update our grid
			if self.grid_max:
				pix   = hits.sky2pix([res.cat.dec,res.cat.ra])
				hits += utils.bin_multi(utils.nint(pix), hits.shape)
			cat.append(res.cat)
			if res.snmin <= snmin: break
		# The final result will have an extra contamination column
		if len(cat) == 0: cat = res.cat
		else:             cat = np.concatenate(cat, 0).view(np.recarray)
		contam = np.zeros(len(cat), [("contam", "d", (len(map),))]).view(np.recarray)
		if len(cat) > 0:
			own_vals = np.concatenate(own_vals,0)
			tot_vals = model.at([cat.dec,cat.ra]).T
			contam.contam = (tot_vals-own_vals)/np.abs(own_vals)
		cat = merge_arrays([cat, contam]).view(np.recarray)
		# Disqualify sources in overly contaminated regions, unless they dominate alone.
		# Maybe this could be moved into a separate class, but single-pass finders will
		# never get very dense detectors, and the grid is needed here anyway to avoid
		# the finding being too slow.
		if self.grid_max and self.grid_dominance and len(cat) > 0:
			pix   = utils.nint(hits.sky2pix([cat.dec,cat.ra]))
			nhit  = hits.at(pix, unit="pix", order=0)
			# Get the most significant object in each cell
			pix1d = pix[0]*hits.shape[-1]+pix[1]
			nr1   = np.zeros(hits.size)
			np.maximum.at(nr1, pix1d, cat.snr)
			# Get rest of the stuff in the cell
			rest  = np.bincount(pix1d, cat.snr, minlength=hits.size)-nr1
			# If a cell has too many objects and nr1 does not dominate the rest, cut all sources in it
			nr1, rest = nr1[pix1d], rest[pix1d]
			bad  = (nhit > self.grid_max) & (rest > nr1/self.grid_dominance)
			cat, cat_bad = cat[~bad], cat[bad]
			# Update model to reflect the cut sources
			if len(cat_bad) > 0:
				model -= self.modeller(cat_bad)
			print("cut %d srcs, %d remaining" % (len(cat_bad), len(cat)))
		return bunch.Bunch(cat=cat, snmin=snmin, model=model, hits=hits)

class MeasurerSimple(Measurer):
	def __init__(self, nmat, beam, scaling=1):
		self.beam   = beam
		self.nmat   = nmat
		self.scaling= scaling
		self.order  = 3
	def __call__(self, map, icat):
		assert map.ndim == 3, "Map must be an enmap with 3 dims"
		ncomp = len(map)
		cat   = icat.copy()
		pixs  = map.sky2pix([icat.dec,icat.ra])
		# Apply the matched filter
		rho, kappa = self.nmat.matched_filter(map, self.beam)
		kappa     = sanitize_kappa(kappa)
		# Read off the total values at the given positions
		scaling   = np.zeros(len(rho),rho.dtype)+self.scaling
		rho_tot   = np.sum(rho*scaling[:,None,None],0)
		kappa_tot = np.sum(scaling[:,None,None,None]*kappa*scaling[None,:,None,None],(0,1))
		snr_tot   = rho_tot/kappa_tot**0.5
		flux_tot, dflux_tot = solve_mapsys(kappa_tot, rho_tot)
		del rho_tot, kappa_tot
		cat.snr             = snr_tot  .at(pixs, unit="pix", order=0)
		cat.flux_tot        = flux_tot .at(pixs, unit="pix", order=self.order)
		cat.dflux_tot       = dflux_tot.at(pixs, unit="pix", order=0)
		del snr_tot, flux_tot, dflux_tot
		# Read off the individual values
		flux, dflux = solve_mapsys(kappa, rho)
		del rho, kappa
		cat.flux    = flux .at(pixs, unit="pix", order=self.order).T
		cat.dflux   = dflux.at(pixs, unit="pix", order=0).T
		return bunch.Bunch(cat=cat)

class MeasurerMulti(Measurer):
	def __init__(self, measurers):
		self.measurers = measurers
	def __call__(self, map, icat):
		cat = icat.copy()
		if len(icat) == 0: return bunch.Bunch(cat=cat)
		uvals, order, edges = utils.find_equal_groups_fast(icat.case)
		for i, ca in enumerate(uvals):
			subicat = icat[order[edges[i]:edges[i+1]]]
			if len(subicat) == 0: continue
			subocat = self.measurers[i](map, subicat).cat
			cat[order[edges[i]:edges[i+1]]] = subocat
		return bunch.Bunch(cat=cat)

class MeasurerIterative(Measurer):
	def __init__(self, measurer, modeller, sntol=0.25, snscale=1):
		self.measurer = measurer
		self.modeller = modeller
		self.sntol    = sntol
		self.snscale  = snscale
		self.snmin    = 0.1 # do everything at once below this
	def __call__(self, map, icat, verbose=False):
		cat    = icat.copy()
		if cat.size == 0: return bunch.Bunch(cat=cat, model=self.modeller(cat))
		snr    = icat.snr * self.snscale
		groups = snr_split(snr, sntol=self.sntol, snmin=self.snmin)
		model  = np.zeros_like(map)
		for gi, group in enumerate(groups):
			if verbose: print("Measuring group %d with snmin %6.2f" % (gi+1, np.min(snr[group])))
			subcat = self.measurer(map-model, icat[group]).cat
			model += self.modeller(subcat)
			cat[group] = subcat
		return bunch.Bunch(cat=cat, model=model)

class ModellerPerfreq(Modeller):
	def __init__(self, shape, wcs, beam_profiles, dtype=np.float32, nsigma=5):
		self.shape = shape
		self.wcs   = wcs
		self.dtype = dtype
		self.nsigma= nsigma
		self.beam_profiles = []
		for i, (r,b) in enumerate(beam_profiles):
			self.beam_profiles.append(np.array([r,b/np.max(b)]))
		self.areas = [utils.calc_beam_area(prof) for prof in self.beam_profiles]
	def __call__(self, cat):
		ncomp = len(self.beam_profiles)
		omap  = enmap.zeros((ncomp,)+self.shape[-2:], self.wcs, self.dtype)
		if len(cat) == 0: return omap
		for i in range(ncomp):
			# This just subtracts the raw measurement at each frequencies. Some frequencies may have
			# bad S/N. Maybe consider adding the scaled tot flux as a weak prior?
			srcparam = np.concatenate([cat.dec[:,None],cat.ra[:,None],cat.flux[:,i:i+1]/self.areas[i]],-1)
			pointsrcs.sim_srcs(self.shape[-2:], self.wcs, srcparam, self.beam_profiles[i], omap=omap[i], nsigma=self.nsigma)
		return omap

class ModellerScaled(Modeller):
	def __init__(self, shape, wcs, beam_profiles, scaling, dtype=np.float32, nsigma=5):
		self.shape = shape
		self.wcs   = wcs
		self.dtype = dtype
		self.nsigma= nsigma
		self.scaling = scaling
		self.beam_profiles = []
		for i, (r,b) in enumerate(beam_profiles):
			self.beam_profiles.append(np.array([r,b/np.max(b)]))
		self.areas = [utils.calc_beam_area(prof) for prof in self.beam_profiles]
	def __call__(self, cat):
		ncomp = len(self.beam_profiles)
		omap  = enmap.zeros((ncomp,)+self.shape[-2:], self.wcs, self.dtype)
		if len(cat) == 0: return omap
		for i in range(ncomp):
			srcparam = np.concatenate([cat.dec[:,None],cat.ra[:,None],cat.flux_tot[:,None]*self.scaling[i]/self.areas[i]],-1)
			pointsrcs.sim_srcs(self.shape[-2:], self.wcs, srcparam, self.beam_profiles[i], omap=omap[i], nsigma=self.nsigma)
		return omap

class ModellerMulti(Modeller):
	def __init__(self, modellers):
		self.modellers = modellers
	def __call__(self, cat):
		# If the cat is empty, just pass it on to the first modeller to have
		# it generate an empty map
		if len(cat) == 0: return self.modellers[0](cat)
		# Loop through the catalog entries of each type and have the corresponding
		# modeller build a model for them
		uvals, order, edges = utils.find_equal_groups_fast(cat.case)
		omap = None
		for i, ca in enumerate(uvals):
			subcat = cat[order[edges[i]:edges[i+1]]]
			if len(subcat) == 0: continue
			map    = self.modellers[ca](subcat)
			if omap is None: omap  = map
			else:            omap += map
		return omap

def format_cat(cat):
	nfield, ncomp = cat.flux.shape[-2:]
	names  = "TQU"
	header = "#%7s %8s %9s" % ("ra", "dec", "snr_T")
	for i in range(1,ncomp): header += " %7s" % ("snr_"+names[i])
	for i in range(ncomp):   header += " %8s %7s" % ("ftot_"+names[i], "dftot_"+names[i])
	for i in range(nfield):
		for j in range(ncomp):
			header += " %8s %7s" % ("flux_"+names[j]+"%d"%(i+1), "dflux_"+names[j]+"%d"%(i+1))
	header += " %2s" % "ca"
	for i in range(nfield): header += " %6s" % ("cont_%d" % (i+1))
	header += "\n"
	res = ""
	for i in range(len(cat)):
		res += "%8.4f %8.4f" % (cat.ra[i]/utils.degree, cat.dec[i]/utils.degree)
		snr  = cat.snr[i].reshape(-1)
		res += " %9.2f" % snr[0] + " %6.2f"*(len(snr)-1) % tuple(snr[1:])
		flux = cat. flux_tot[i].reshape(-1)
		dflux= cat.dflux_tot[i].reshape(-1)
		for j in range(len(flux)):
			res += "  %8.2f %7.2f" % (flux[j], dflux[j])
		flux = cat. flux[i].reshape(-1)
		dflux= cat.dflux[i].reshape(-1)
		for j in range(len(flux)):
			res += "  %8.2f %7.2f" % (flux[j], dflux[j])
		try: res += " %2d" % (cat.case[i])
		except (KeyError, AttributeError): pass
		try:
			for j in range(len(cat.contam[i])):
				res += " %6.1f" % (cat.contam[i,j])
		except (KeyError, AttributeError): pass
		res += "\n"
	return header + res

def write_catalog(ofile, cat):
	if ofile.endswith(".fits"): write_catalog_fits(ofile, cat)
	else: write_catalog_txt (ofile, cat)

def read_catalog(ifile):
	if ifile.endswith(".fits"): return read_catalog_fits(ifile)
	else: return read_catalog_txt(ifile)

def write_catalog_fits(ofile, cat):
	from astropy.io import fits
	ocat = cat.copy()
	for field in ["ra","dec"]: ocat[field] /= utils.degree # angles in degrees
	hdu = fits.hdu.table.BinTableHDU(ocat)
	hdu.writeto(ofile, overwrite=True)

def read_catalog_fits(fname):
	from astropy.io import fits
	hdu = fits.open(fname)[1]
	cat = np.asarray(hdu.data).view(np.recarray)
	for field in ["ra","dec"]: cat[field] *= utils.degree # deg -> rad
	return cat

def write_catalog_txt(ofile, cat):
	with open(ofile, "w") as ofile:
		ofile.write(format_cat(cat))

def read_data(fnames, sel=None, pixbox=None, box=None, geometry=None, comp=0, split=0, unit="flux", dtype=np.float32,
		beam_rmax=5*utils.degree, beam_res=2*utils.arcsec, deconv_pixwin=True, apod=15*utils.arcmin,
		ivscale=[1,0.5,0.5]):
	"""Read multi-frequency data for a single split of a single component, preparing it for
	analysis."""
	# Read in our data files and harmonize
	br   = np.arange(0,beam_rmax,beam_res)
	data = bunch.Bunch(maps=[], ivars=[], beams=[], freqs=[], l=None, bls=[], names=[], beam_profiles=[])
	for ifile in fnames:
		d = mapdata.read(ifile, sel=sel, pixbox=pixbox, box=box, geometry=geometry)
		# The 0 here is just selecting the first split. That is, we don't support splits
		data.maps .append(d.maps [split].astype(dtype)[comp])
		data.ivars.append(d.ivars[split].astype(dtype)*ivscale[comp])
		data.freqs.append(d.freq)
		if data.l is None: data.l = d.maps[0].modlmap()
		data.beams.append(enmap.ndmap(np.interp(data.l, np.arange(len(d.beam)), d.beam/np.max(d.beam)), d.maps[0].wcs).astype(dtype))
		data.names.append(".".join(os.path.basename(ifile).split(".")[:-1]))
		data.bls.append(d.beam)
		data.beam_profiles.append(np.array([br, curvedsky.harm2profile(d.beam,br)]).astype(dtype))

	data.maps  = enmap.enmap(data.maps )
	data.ivars = enmap.enmap(data.ivars)
	data.beams = enmap.enmap(data.beams)
	data.freqs = np.array(data.freqs)
	if unit == "uK":
		data.fconvs = np.full(len(data.freqs), 1.0, dtype)
	elif unit == "flux":
		data.fconvs= (utils.dplanck(data.freqs*1e9, utils.T_cmb)/1e3).astype(dtype) # uK -> mJy/sr
	else: raise ValueError("Unrecognized unit '%s'" % str(unit))
	data.n     = len(data.freqs)

	# Apply the unit
	data.maps  *= data.fconvs[:,None,None]
	data.ivars /= data.fconvs[:,None,None]**2

	# Should generalize this to handle internal map edges and frequency differences
	mask      = enmap.shrink_mask(enmap.grow_mask(data.ivars>0, 1*utils.arcmin), 1*utils.arcmin)
	apod_map  = enmap.apod_mask(mask, apod)
	data.apod = apod_map
	data.fapod= np.mean(apod_map**2)
	data.maps  *= apod_map
	data.ivars *= apod_map**2

	# Get the pixel window and optionall deconvolve it
	data.wy, data.wx = [w.astype(dtype) for w in enmap.calc_window(data.maps.shape)]
	if deconv_pixwin:
		data.maps = enmap.ifft(enmap.fft(data.maps)/data.wy[:,None]/data.wx[None,:]).real

	return data

def build_cmb_2d(shape, wcs, cl_cmb, dtype=np.float32):
	lmap = enmap.lmap(shape, wcs)
	l    = np.sum(lmap**2,0)**0.5
	cmb  = enmap.samewcs(utils.interp(l, np.arange(cl_cmb.shape[-1]), cl_cmb), l).astype(dtype)
	# Rotate [TEB,EB] -> [TQU,TQU]. FIXME: not a perfect match
	R = enmap.queb_rotmat(lmap, spin=2, inverse=True)
	cmb[1:,:] = np.einsum("abyx,bcyx->acyx", R, cmb[1:,:])
	cmb[:,1:] = np.einsum("abyx,cbyx->acyx", cmb[:,1:], R)
	return cmb

def build_case_ptsrc(data, scaling=None):
	if scaling is None: scaling = np.full(data.n, 1.0)
	scaling  = np.asarray(scaling).astype(data.maps.dtype)
	modeller = ModellerPerfreq(data.maps.shape, data.maps.wcs, data.beam_profiles)
	return bunch.Bunch(profile=data.beams, scaling=scaling, modeller=modeller)

def build_case_tsz(data, size=1):
	scaling = utils.tsz_spectrum(data.freqs*1e9)/np.abs(utils.tsz_spectrum(data.freq0*1e9))
	# Get the fourier shapes
	lprofs  = (utils.tsz_tform(size, data.l)*data.beams).astype(data.maps.dtype)
	lprofs /= np.max(lprofs, (-2,-1))[:,None,None]
	# Get the real-space templates for the model
	profs1d = []
	for i in range(data.n):
		lprof1d  = utils.tsz_tform(size, np.arange(len(data.bls[i])))*data.bls[i]
		lprof1d /= np.max(lprof1d)
		br = data.beam_profiles[i][0]
		profs1d.append(np.array([br, curvedsky.harm2profile(lprof1d, br)]))
	modeller = ModellerScaled(data.maps.shape, data.maps.wcs, profs1d, scaling)
	return bunch.Bunch(profile=lprofs, scaling=scaling, modeller=modeller)

def build_nmat_prior(data, type="constcorr", cmb=None, pol=False):
	if type == "constcov":
		iN    = build_iN_constcov_prior(data, cmb=cmb, lknee0=800 if pol else 2000)
		nmat  = NmatConstcov(iN, data.apod)
	elif type == "constcorr":
		iN    = build_iN_constcorr_prior(data, cmb=cmb, lknee0=800 if pol else 2000)
		nmat  = NmatConstcorr(iN, data.ivars)
	else:
		raise ValueError("Unsupported prior nmat: '%s'" % str(args.nmat1))
	return nmat

def build_nmat_empirical(data, noise_map, type="constcorr", smooth="angular"):
	if type == "constcov":
		iN    = build_iN_constcov(data, noise_map, smooth=smooth)
		nmat  = NmatConstcov(iN)
	elif type == "constcorr":
		iN    = build_iN_constcorr(data, noise_map, smooth=smooth)
		nmat  = NmatConstcorr(iN, data.ivars)
	elif type == "wavelet":
		from pixell import uharm
		wbasis= wavelets.ButterTrim(step=2**0.5, lmin=50)
		uht   = uharm.UHT(data.maps.shape, data.maps.wcs, mode="flat")
		wt    = wavelets.WaveletTransform(uht, basis=wbasis)
		wiN   = build_wiN(noise_map, wt)
		nmat  = NmatWavelet(wt, wiN)
	elif type == "weighted-wavelet":
		from pixell import uharm
		wbasis= wavelets.ButterTrim(step=2**0.5, lmin=50)
		uht   = uharm.UHT(data.maps.shape, data.maps.wcs, mode="flat")
		wt    = wavelets.WaveletTransform(uht, basis=wbasis)
		wiN   = build_wiN_ivar(noise_map, data.ivars, wt)
		nmat  = NmatWavelet(wt, wiN)
	else:
		raise ValueError("Unsupported empirical nmat: '%s'" % str(args.nmat1))
	return nmat

def find_objects(data, cases, nmat, snmin=5, resid=False, verbose=False):
	# Build an interative finder from a multi-case finder and a multi-case modeller
	raw_finder    = FinderMulti(nmat, cases.profile, cases.scaling, save_snr=True)
	modeller      = ModellerMulti(cases.modeller)
	finder        = FinderIterative(raw_finder, modeller)
	# Run the finder
	res           = finder(data.maps, snmin=snmin, verbose=verbose)
	# Add some useful quantities to the result object
	res.maps      = data.maps
	res.snr       = raw_finder.snr
	res.fconvs    = data.fconvs
	if resid:
		res.resid     = data.maps-res.model
		res.resid_snr = raw_finder(res.resid).snr
	return res

def measure_objects(data, cases, nmat, cat, snmin=5, resid=False, verbose=False):
	raw_measurer = MeasurerMulti([MeasurerSimple(nmat, profile, scaling) for profile, scaling in zip(cases.profile, cases.scaling)])
	modeller     = ModellerMulti(cases.modeller)
	measurer     = MeasurerIterative(raw_measurer, modeller)
	res          = measurer(data.maps, cat, verbose=verbose)
	# Add some useful quantities to the result object
	res.maps     = data.maps
	res.fconvs   = data.fconvs
	if resid:
		res.resid  = data.maps-res.model
	return res

def build_cases(data, templates):
	cases = []
	for params in templates:
		type = params[0]
		if type == "ptsrc":
			specind = params[1]
			cases.append(build_case_ptsrc(data, (data.freqs/data.freq0)**specind))
		elif type == "graysrc":
			dust_temp = params[1]
			cases.append(build_case_ptsrc(data, utils.graybody(data.freqs*1e9, dust_temp)/utils.graybody(data.freq0*1e9, dust_temp)))
		elif type == "tsz":
			cluster_scale = params[1]
			cases.append(build_case_tsz(data, cluster_scale))
	cases = bunch.concatenate(cases)
	return cases

def inject_objects(data, cases, cat):
	modeller = ModellerMulti(cases.modeller)
	data.maps += modeller(cat)

default_templates = [("ptsrc",-0.66), ("ptsrc",0),("graysrc",10), ("tsz",0.1),
		("tsz",2), ("tsz",4), ("tsz",6), ("tsz",8)]
def search_maps(ifiles, sel=None, pixbox=None, box=None, templates=default_templates, cl_cmb=None, freq0=98.0,
		nmat1="constcorr", nmat2="constcorr", snr1=5, snr2=4, mode="TQU", dtype=np.float32,
		verbose=False, sim_cat=None):
	# Read in the total intensity data
	if verbose: print("Reading T from %s" % str(ifiles))
	data   = read_data(ifiles, sel=sel, pixbox=pixbox, box=box, dtype=dtype)
	data.freq0 = freq0
	ncomp  = len(mode)
	nfield = len(data.maps)
	cat_dtype  = [("ra", "d"), ("dec", "d"), ("snr", "d", (ncomp,)), ("flux_tot", "d", (ncomp,)),
			("dflux_tot", "d", (ncomp,)), ("flux", "d", (nfield,ncomp)), ("dflux", "d", (nfield,ncomp)),
			("case", "i"), ("contam", "d", (nfield,))]
	cases = build_cases(data, templates)

	# Abort if we have no data to process
	if np.all(data.ivars == 0):
		map_tot = enmap.zeros((nfield,ncomp)+data.maps.shape[-2:], data.maps.wcs, dtype)
		cat     = np.zeros(0, cat_dtype)
		return bunch.Bunch(cat=cat, maps=map_tot, model=map_tot, snr=map_tot[0,0],
			resid_snr=map_tot[0,0], hits=map_tot[0,0], fconvs=data.fconvs)

	cmb   = build_cmb_2d(*data.maps.geometry, cl_cmb, dtype=data.maps.dtype) if cl_cmb is not None else None

	# Optionally inject signal
	if sim_cat is not None: inject_objects(data, cases, sim_cat)

	# Total intensity
	if verbose: print("1st pass T find")
	nmat   = build_nmat_prior(data, type=nmat1, cmb=cmb[0,0] if cmb is not None else None)
	res_t  = find_objects(data, cases, nmat, snmin=snr1, resid=nmat2=="none", verbose=verbose)
	if nmat2 != "none":
		if verbose: print("2nd pass T find")
		nmat   = build_nmat_empirical(data, data.maps-res_t.model, type=nmat2)
		res_t  = find_objects(data, cases, nmat, snmin=snr2, resid=True, verbose=verbose)

	res = [res_t]
	if mode == "T":
		pass
	elif mode == "TQU":
		# Measure polarization too
		for comp in [1,2]:
			if verbose: print("Reading %s from %s" % (mode[comp], str(ifiles)))
			data  = read_data(ifiles, sel=sel, pixbox=pixbox, box=box, comp=comp)
			data.freq0 = freq0
			if verbose: print("1st pass %s measure" % mode[comp])
			nmat  = build_nmat_prior(data, type=nmat1, pol=True, cmb=cmb[comp,comp] if cmb is None else None)
			res_p = measure_objects(data, cases, nmat, res_t.cat, verbose=verbose)
			if nmat2 != "none":
				if verbose: print("2nd pass %s measure" % mode[comp])
				nmat  = build_nmat_empirical(data, noise_map=data.maps-res_p.model, type=nmat2)
				res_p = measure_objects(data, cases, nmat, res_t.cat, verbose=verbose)
			res.append(res_p)
	# First the catalog
	cat = np.zeros(len(res_t.cat), cat_dtype).view(np.recarray)
	cat.ra     = res_t.cat.ra
	cat.dec    = res_t.cat.dec
	cat.case   = res_t.cat.case
	cat.contam = res_t.cat.contam
	for i in range(len(res)):
		cat.snr[:,i]       = res[i].cat.snr
		cat. flux_tot[:,i] = res[i].cat. flux_tot
		cat.dflux_tot[:,i] = res[i].cat.dflux_tot
		cat. flux[:,:,i]   = res[i].cat. flux
		cat.dflux[:,:,i]   = res[i].cat.dflux
	# Then the maps
	map_tot   = enmap.samewcs(np.concatenate([r.maps [:,None] for r in res],1), data.maps)
	model_tot = enmap.samewcs(np.concatenate([r.model[:,None] for r in res],1), data.maps)
	return bunch.Bunch(cat=cat, maps=map_tot, model=model_tot, snr=res_t.snr,
			resid_snr=res_t.resid_snr, hits=res_t.hits, fconvs=data.fconvs)

def search_maps_tiled(ifiles, odir, tshape=(1000,1000), margin=100, padding=100,
		box=None, pixbox=None, sel=None,
		templates=default_templates, cl_cmb=None, freq0=98.0, nmat1="constcorr",
		nmat2="constcorr", snr1=5, snr2=4, mode="TQU", dtype=np.float32, comm=None,
		cont=False, verbose=False):
	wdir = odir + "/work"
	utils.mkdir(wdir)
	if comm is None: comm = bunch.Bunch(rank=0, size=1)
	tshape = np.zeros(2,int)+tshape
	meta   = mapdata.read_meta(ifiles[0])
	# Allow us to slice the map that will be tiled
	geo    = enmap.Geometry(*meta.map_geometry)
	if pixbox is not None or box is not None:
		geo  = geo.submap(pixbox=pixbox, box=box)
	if sel is not None: geo = geo[sel]
	shape  = np.array(geo.shape[-2:])
	ny,nx  = (shape+tshape-1)//tshape
	def is_done(ty,tx): return os.path.isfile("%s/cat_%03d_%03d.fits" % (wdir, ty,tx))
	tyxs   = [(ty,tx) for ty in range(ny) for tx in range(nx) if (not cont or not is_done(ty,tx))]
	for ind in range(comm.rank, len(tyxs), comm.size):
		# Get basic area of this tile
		tyx = np.array(tyxs[ind])
		if verbose: print("%2d Processing tile %2d %2d of %2d %2d" % (comm.rank, tyx[0], tyx[1], ny, nx))
		yx1 = tyx*tshape
		yx2 = np.minimum((tyx+1)*tshape, shape)
		# Apply padding
		wyx1 = yx1-margin-padding
		wyx2 = yx2+margin+padding
		# Process this tile
		res = search_maps(ifiles, pixbox=[wyx1,wyx2], templates=templates,
				cl_cmb=cl_cmb, freq0=freq0, nmat1=nmat1, nmat2=nmat2,
				snr1=snr1, snr2=snr2, mode=mode, dtype=dtype, verbose=verbose)
		# Write tile results to work directory. We do this to avoid using too much memory,
		# and to allow us to continue
		def unpad(map): return map[...,padding:-padding,padding:-padding]
		def fix(map): return unpad(enmap.apply_window(map))/res.fconvs[:,None,None,None]
		enmap.write_map("%s/map_%03d_%03d.fits"  % (wdir,*tyx), fix(res.maps))
		enmap.write_map("%s/model_%03d_%03d.fits" % (wdir,*tyx), fix(res.model))
		enmap.write_map("%s/resid_%03d_%03d.fits" % (wdir,*tyx), fix(res.maps-res.model))
		enmap.write_map("%s/map_snr_%03d_%03d.fits"   % (wdir,*tyx), unpad(res.snr))
		enmap.write_map("%s/resid_snr_%03d_%03d.fits" % (wdir,*tyx), unpad(res.resid_snr))
		write_catalog("%s/cat_%03d_%03d.fits"     % (wdir,*tyx), res.cat)
	# When everything's done, merge things into single files
	if comm.rank == 0:
		# Get the tile catalogs and their area of responsibility
		cats = []; boxes = []
		for ty in range(ny):
			for tx in range(nx):
				tyx    = np.array([ty,tx])
				pixbox = np.array([tyx*tshape, np.minimum((tyx+1)*tshape, shape)])
				boxes.append(np.sort(enmap.pixbox2skybox(*geo, pixbox),0))
				cats .append(read_catalog(wdir + "/cat_%03d_%03d.fits" % (ty,tx)))
		cat = merge_tiled_cats(cats, boxes)
		write_catalog("%s/cat.fits" % odir, cat)
		write_catalog("%s/cat.txt"  % odir, cat)
		for name in ["map", "model", "resid", "map_snr", "resid_snr"]:
			paths = [["%s/%s_%03d_%03d.fits" % (wdir,name,ty,tx) for tx in range(nx)] for ty in range(ny)]
			map = merge_tiles(*geo, paths, dtype=dtype, margin=margin)
			enmap.write_map("%s/%s.fits" % (odir,name), map)
			del map
	# Done

def merge_tiled_cats(cats, boxes, margin=2*utils.arcmin):
	"""Merge the list of catalogs cats, where each catalog owns the corresponding
	bounding box in boxes, but also also an overlapping area around. For the area
	select(box,0,edge-margin) the catalog will be used directly. For the area
	select(box,edge-margin,edge+margin) duplicates will be looked for and removed."""
	# First get the central part of each catalog, which can be used as they are
	boxes = np.asarray(boxes) # [nbox,{from,to},{dec,ra}]
	boxes_inner = np.concatenate([boxes[:,None,0,:]+margin,boxes[:,None,1,:]-margin],1)
	boxes_outer = np.concatenate([boxes[:,None,0,:]-margin,boxes[:,None,1,:]+margin],1)
	cats_inner  = []
	cats_border = []
	for ci, cat in enumerate(cats):
		pos    = np.array([cat.dec,cat.ra]).T
		inner  = np.all(pos > boxes_inner[ci][0],1) & np.all(pos < boxes_inner[ci][1],1)
		outer  = np.any(pos < boxes_outer[ci][0],1) | np.any(pos > boxes_outer[ci][1],1)
		border = ~inner & ~outer
		cats_inner .append(cat[inner ])
		cats_border.append(cat[border])
	cat_inner  = np.concatenate(cats_inner)
	cat_border = merge_duplicates(cats_border)
	ocat   = np.concatenate([cat_inner, cat_border]).view(np.recarray)
	order  = np.argsort(ocat.snr[:,0])[::-1]
	ocat   = ocat[order]
	return ocat

def merge_duplicates(cats, dr=1*utils.arcmin, dsnr=1.5):
	"""Given a list of catalogs that could contain duplicates, choose one result
	for each dr*dr cell spatially and log-step dsnr in S/N ratio."""
	from scipy import spatial
	cat  = np.concatenate(cats).view(np.recarray)
	lsnr = np.log(np.maximum(np.abs(cat.snr[:,0]),1))/np.log(dsnr)
	pos  = utils.ang2rect([cat.ra,cat.dec]).T/dr
	x    = np.concatenate([pos,lsnr[:,None]],1)
	tree = spatial.cKDTree(x)
	groups = tree.query_ball_tree(tree, 1)
	done = np.zeros(len(cat),bool)
	ocat = []
	for gi, group in enumerate(groups):
		# Remove everything that's done
		group = np.array(group)
		group = group[~done[group]]
		if len(group) == 0: continue
		# Keep the first entry
		first = group[0]
		ocat.append(cat[first])
		# Mark all entries as done. Assume hte rest were duplicates
		done[group] = True
	ocat = np.array(ocat).view(np.recarray)
	return ocat

def make_edge_interp(n, w, left=True, right=True, dtype=np.float32):
	x = (np.arange(2*w)+1).astype(dtype)/(2*w+1)
	l = x       if left  else x*0+1
	r = x[::-1] if right else x*0+1
	m = np.full(n-2*w, 1.0, dtype)
	return np.concatenate([l,m,r])

def merge_tiles(shape, wcs, paths, margin=100, dtype=np.float32):
	# Get the pre-dimensions from the first tile
	shape = enmap.read_map_geometry(paths[0][0])[0][:-2]+shape[-2:]
	omap  = enmap.zeros(shape, wcs, dtype)
	ny    = len(paths)
	nx    = len(paths[0])
	for ty in range(ny):
		for tx in range(nx):
			fname = paths[ty][tx]
			imap  = enmap.read_map(fname)
			h, w  = imap.shape[-2:]
			wy    = make_edge_interp(h-2*margin, margin, ty>0, ty<ny-1, dtype=imap.dtype)
			wx    = make_edge_interp(w-2*margin, margin, tx>0, tx<nx-1, dtype=imap.dtype)
			imap  = imap * wy[:,None] * wx[None,:]
			omap.insert(imap, op=np.ndarray.__iadd__)
	return omap

dtype  = np.float32
sel    = parse_slice(args.slice)
box    = utils.parse_box(args.box)*utils.degree if args.box else None
cl_cmb = powspec.read_spectrum(args.cmb) if args.cmb else None
sim_cat= read_catalog(args.sim) if args.sim else None

if False:
	res    = search_maps(args.ifiles, sel=sel, box=box, cl_cmb=cl_cmb, nmat1=args.nmat1, nmat2=args.nmat2, snr1=args.snr1, snr2=args.snr2, dtype=dtype, verbose=True, mode=args.comps, sim_cat=sim_cat)

	write_catalog(args.odir + "/cat.fits", res.cat)
	write_catalog(args.odir + "/cat.txt",  res.cat)
	enmap.write_map(args.odir + "/map.fits",   enmap.apply_window(res.maps )/res.fconvs[:,None,None,None])
	enmap.write_map(args.odir + "/model.fits", enmap.apply_window(res.model)/res.fconvs[:,None,None,None])
	enmap.write_map(args.odir + "/resid.fits", enmap.apply_window(res.maps-res.model)/res.fconvs[:,None,None,None])
	enmap.write_map(args.odir + "/map_snr.fits",   res.snr)
	enmap.write_map(args.odir + "/resid_snr.fits", res.resid_snr)
	enmap.write_map(args.odir + "/hits.fits", res.hits)
else:
	search_maps_tiled(args.ifiles, args.odir, tshape=tshape, sel=sel, box=box, cl_cmb=cl_cmb, nmat1=args.nmat1, nmat2=args.nmat2, snr1=args.snr1, snr2=args.snr2, dtype=dtype, verbose=True, cont=args.cont, comm=comm, mode=args.comps)

#data   = read_data(args.ifiles, sel=sel, box=box)
#data.freq0 = 98.0
#cl_cmb = powspec.read_spectrum(args.cmb) if args.cmb else None
#
## Set up our signal types
#cases = []
#cases.append(build_case_ptsrc(data, (data.freqs/data.freq0)**-0.66))
#cases.append(build_case_ptsrc(data, (data.freqs/data.freq0)**0))
#cases.append(build_case_ptsrc(data, utils.graybody(data.freqs*1e9, 10)/utils.graybody(data.freq0*1e9, 10)))
#for cluster_scale in [0.1, 2, 4, 6, 8]:
#	cases.append(build_case_tsz(data, cluster_scale))
#cases = bunch.concatenate(cases)
#
#cmb   = build_cmb_2d(*data.maps.geometry, cl_cmb, dtype=data.maps.dtype)[0,0] if cl_cmb is not None else None
#nmat  = build_nmat_prior(data, type=args.nmat1, cmb=cmb)
#res   = find_objects(data, cases, nmat, snmin=args.snr1, resid=True)
#
#cat_str = format_cat(res.cat)
#print(cat_str)
#with open(args.odir + "/cat.txt", "w") as ofile: ofile.write(cat_str)
#enmap.write_map(args.odir + "/map.fits",   enmap.apply_window(res.maps )/data.fconvs[:,None,None])
#enmap.write_map(args.odir + "/model.fits", enmap.apply_window(res.model)/data.fconvs[:,None,None])
#enmap.write_map(args.odir + "/resid.fits", enmap.apply_window(res.resid)/data.fconvs[:,None,None])
#enmap.write_map(args.odir + "/map_snr.fits",   res.snr)
#enmap.write_map(args.odir + "/resid_snr.fits", res.resid_snr)
#enmap.write_map(args.odir + "/hits.fits", res.hits)
#
#nmat = build_nmat_empirical(data, res.resid, type=args.nmat2)
#res  = find_objects(data, cases, nmat, snmin=args.snr2, resid=True)
#
#cat_str = format_cat(res.cat)
#print(cat_str)
#with open(args.odir + "/cat2.txt", "w") as ofile: ofile.write(cat_str)
#enmap.write_map(args.odir + "/map2.fits",   enmap.apply_window(res.maps )/data.fconvs[:,None,None])
#enmap.write_map(args.odir + "/model2.fits", enmap.apply_window(res.model)/data.fconvs[:,None,None])
#enmap.write_map(args.odir + "/resid2.fits", enmap.apply_window(res.resid)/data.fconvs[:,None,None])
#enmap.write_map(args.odir + "/map2_snr.fits",   res.snr)
#enmap.write_map(args.odir + "/resid2_snr.fits", res.resid_snr)
#enmap.write_map(args.odir + "/hits2.fits", res.hits)
#
#cat_t = res.cat
#
## Measure polarization
#for comp in [1,2]:
#	data   = read_data(args.ifiles, sel=sel, box=box, comp=comp)
#	data.freq0 = 98.0
#
#	cmb   = build_cmb_2d(*data.maps.geometry, cl_cmb, dtype=data.maps.dtype)[comp,comp] if cl_cmb is not None else None
#	nmat  = build_nmat_prior(data, type=args.nmat1, cmb=cmb, pol=True)
#	res   = measure_objects(data, cases, nmat, cat_t, resid=True)
#
#	cat_str = format_cat(res.cat)
#	print(cat_str)
#	# NB! This is still based on cat2's positions!
#	with open(args.odir + "/cat_pol%d.txt" % comp, "w") as ofile: ofile.write(cat_str)
#	enmap.write_map(args.odir + "/map_pol%d.fits" % comp,   enmap.apply_window(res.maps)/data.fconvs[:,None,None])
#	enmap.write_map(args.odir + "/model_pol%d.fits" % comp, enmap.apply_window(res.model)/data.fconvs[:,None,None])
#	enmap.write_map(args.odir + "/resid_pol%d.fits" % comp, enmap.apply_window(res.resid)/data.fconvs[:,None,None])
#
#	nmat  = build_nmat_empirical(data, noise_map=res.resid, type=args.nmat2)
#	res   = measure_objects(data, cases, nmat, cat_t, resid=True)
#
#	cat_str = format_cat(res.cat)
#	print(cat_str)
#	# NB! This is still based on cat2's positions!
#	with open(args.odir + "/cat2_pol%d.txt" % comp, "w") as ofile: ofile.write(cat_str)
#	enmap.write_map(args.odir + "/map2_pol%d.fits" % comp,   enmap.apply_window(res.maps)/data.fconvs[:,None,None])
#	enmap.write_map(args.odir + "/model2_pol%d.fits" % comp, enmap.apply_window(res.model)/data.fconvs[:,None,None])
#	enmap.write_map(args.odir + "/resid2_pol%d.fits" % comp, enmap.apply_window(res.resid)/data.fconvs[:,None,None])
#
### Measure polarization
##for comp in [1,2]:
##	data   = read_data(args.ifiles, sel=sel, box=box, comp=comp)
##	data.freq0 = 98.0
##
##	if args.nmat1 == "constcov":
##		iN    = build_iN_constcov_prior(data, lknee0=800, cmb=cmb[comp,comp] if cmb is not None else None)
##		nmat  = NmatConstcov(iN)
##	elif args.nmat1 == "constcorr":
##		iN    = build_iN_constcorr_prior(data, lknee0=800, cmb=cmb[comp,comp] if cmb is not None else None)
##		nmat  = NmatConstcorr(iN, data.ivars)
##	else:
##		raise ValueError("Unsupported nmat1: '%s'" % str(args.nmat1))
##
##	raw_measurer = MeasurerMulti([MeasurerSimple(nmat, profile, scaling) for profile, scaling in zip(cases.profile, cases.scaling)])
##	modeller     = ModellerMulti(cases.modeller)
##	measurer     = MeasurerIterative(raw_measurer, modeller)
##	res          = measurer(data.maps, cat_t)
##	resid_map    = data.maps-res.model
##	cat_str      = format_cat(res.cat)
##
##	# NB! This is still based on cat2's positions!
##	with open(args.odir + "/cat_pol%d.txt" % comp, "w") as ofile: ofile.write(cat_str)
##	enmap.write_map(args.odir + "/map_pol%d.fits" % comp,   enmap.apply_window(data.maps)/data.fconvs[:,None,None])
##	enmap.write_map(args.odir + "/model_pol%d.fits" % comp, enmap.apply_window(res.model)/data.fconvs[:,None,None])
##	enmap.write_map(args.odir + "/resid_pol%d.fits" % comp, enmap.apply_window(resid_map)/data.fconvs[:,None,None])
##
##	# Now that we have a clean map, we can try to empirically measure the noise model
##	if args.nmat2 == "constcov":
##		iN    = build_iN_constcov(data, resid_map, smooth=args.nmat_smooth)
##		nmat  = NmatConstcov(iN)
##	elif args.nmat2 == "constcorr":
##		iN    = build_iN_constcorr(data, resid_map, smooth=args.nmat_smooth)
##		nmat  = NmatConstcorr(iN, data.ivars)
##	elif args.nmat2 == "wavelet":
##		from pixell import uharm
##		wbasis= wavelets.ButterTrim(step=2**0.5, lmin=50)
##		uht   = uharm.UHT(data.maps.shape, data.maps.wcs, mode="flat")
##		wt    = wavelets.WaveletTransform(uht, basis=wbasis)
##		wiN   = build_wiN(resid_map, wt)
##		nmat  = NmatWavelet(wt, wiN)
##	elif args.nmat2 == "weighted-wavelet":
##		from pixell import uharm
##		wbasis= wavelets.ButterTrim(step=2**0.5, lmin=50)
##		uht   = uharm.UHT(data.maps.shape, data.maps.wcs, mode="flat")
##		wt    = wavelets.WaveletTransform(uht, basis=wbasis)
##		wiN   = build_wiN_ivar(resid_map, data.ivars, wt)
##		nmat  = NmatWavelet(wt, wiN)
##	else:
##		raise ValueError("Unsupported nmat2: '%s'" % str(args.nmat1))
##
##	raw_measurer = MeasurerMulti([MeasurerSimple(nmat, profile, scaling) for profile, scaling in zip(cases.profile, cases.scaling)])
##	modeller     = ModellerMulti(cases.modeller)
##	measurer     = MeasurerIterative(raw_measurer, modeller)
##	res          = measurer(data.maps, cat_t)
##	resid_map    = data.maps-res.model
##	cat_str      = format_cat(res.cat)
##	print(cat_str)
##
##	with open(args.odir + "/cat2_pol%d.txt" % comp, "w") as ofile: ofile.write(cat_str)
##	enmap.write_map(args.odir + "/map2_pol%d.fits" % comp,   enmap.apply_window(data.maps)/data.fconvs[:,None,None])
##	enmap.write_map(args.odir + "/model2_pol%d.fits" % comp, enmap.apply_window(res.model)/data.fconvs[:,None,None])
##	enmap.write_map(args.odir + "/resid2_pol%d.fits" % comp, enmap.apply_window(resid_map)/data.fconvs[:,None,None])
##
##
##
##
##
##
##
###enmap.write_map("iN_theory.fits", iN)
###dump_ps1d(args.odir + "/iN_theory.txt", iN)
###dump_ps1d(args.odir + "/N_theory.txt", array_ops.eigpow(iN, -1, axes=[0,1]))
##
### Simulate some stuff.
###simmer  = ModellerPerfreq(data.maps.shape, data.maps.wcs, data.beam_profiles)
###sim_cat = np.zeros(10, dtype=[("pos","2d"),("snr","d"),("flux_tot","d"),("dflux_tot","d"), ("flux","d",(3,)),("dflux","d",(3,))]).view(np.recarray)
###np.random.seed(1)
###box     = data.maps.box()
###sim_cat.pos  = (box[0] + (box[1]-box[0])*np.random.uniform(0,1,size=(len(sim_cat),2)))[:,::-1]
###sim_cat.flux = 1000
###sim     = simmer(sim_cat)
###data.maps += sim
### Result of simulation:
### 1010.12 ± 1.98 1.37  1008.64 ± 1.91 1.51  1000.97 ± 5.22 5.38  constcorr0
### 1017.65 ± 1.95 1.33  1020.17 ± 2.06 1.52  1022.03 ± 5.45 5.29  constcorr
### 1003.81 ± 0.97 1.34  1002.23 ± 1.48 1.54   996.19 ± 4.87 5.48  weighted-wavelet
###
### So wavelets have both more accurate flux and scatter that reflects the
### uncertainty. But all the methods are biased slightly in flux. For wavelets
### the bias is 0.381±0.032 / 0.223±0.049 / -0.419±0.162 percent at f090/f150/f220,
### which is pretty small.
##
### And perform the actual search
##raw_finder = FinderMulti(nmat, cases.profile, cases.scaling, save_snr=True)
##modeller   = ModellerMulti(cases.modeller)
##finder     = FinderIterative(raw_finder, modeller)
##res        = finder(data.maps, snmin=args.snr1)
##resid_map  = data.maps-res.model
##resid_snr  = raw_finder(resid_map).snr
##hits       = res.hits.project(*data.maps.geometry, order=0)
##
##cat_str = format_cat(res.cat)
##print(cat_str)
##
##with open(args.odir + "/cat.txt", "w") as ofile: ofile.write(cat_str)
##enmap.write_map(args.odir + "/map.fits",   enmap.apply_window(data.maps)/data.fconvs[:,None,None])
##enmap.write_map(args.odir + "/model.fits", enmap.apply_window(res.model)/data.fconvs[:,None,None])
##enmap.write_map(args.odir + "/resid.fits", enmap.apply_window(resid_map)/data.fconvs[:,None,None])
##enmap.write_map(args.odir + "/map_snr.fits",   raw_finder.snr)
##enmap.write_map(args.odir + "/resid_snr.fits", resid_snr)
##enmap.write_map(args.odir + "/hits.fits", hits)
##
### Now that we have a clean map, we can try to empirically measure the noise model
##if args.nmat2 == "constcov":
##	iN    = build_iN_constcov(data, resid_map, smooth=args.nmat_smooth)
##	nmat  = NmatConstcov(iN)
##elif args.nmat2 == "constcorr":
##	iN    = build_iN_constcorr(data, resid_map, smooth=args.nmat_smooth)
##	nmat  = NmatConstcorr(iN, data.ivars)
##elif args.nmat2 == "wavelet":
##	from pixell import uharm
##	wbasis= wavelets.ButterTrim(step=2**0.5, lmin=50)
##	uht   = uharm.UHT(data.maps.shape, data.maps.wcs, mode="flat")
##	wt    = wavelets.WaveletTransform(uht, basis=wbasis)
##	wiN   = build_wiN(resid_map, wt)
##	nmat  = NmatWavelet(wt, wiN)
##elif args.nmat2 == "weighted-wavelet":
##	from pixell import uharm
##	wbasis= wavelets.ButterTrim(step=2**0.5, lmin=50)
##	uht   = uharm.UHT(data.maps.shape, data.maps.wcs, mode="flat")
##	wt    = wavelets.WaveletTransform(uht, basis=wbasis)
##	wiN   = build_wiN_ivar(resid_map, data.ivars, wt)
##	nmat  = NmatWavelet(wt, wiN)
##else:
##	raise ValueError("Unsupported nmat2: '%s'" % str(args.nmat1))
##
##enmap.write_map(args.odir + "/map2.fits",   enmap.apply_window(data.maps)/data.fconvs[:,None,None])
##
##raw_finder = FinderMulti(nmat, cases.profile, cases.scaling, save_snr=True)
##modeller   = ModellerMulti(cases.modeller)
##finder     = FinderIterative(raw_finder, modeller)
##res        = finder(data.maps, snmin=args.snr2)
##resid_map  = data.maps-res.model
##resid_snr  = raw_finder(resid_map).snr
##hits       = res.hits.project(*data.maps.geometry, order=0)
##
##cat_str = format_cat(res.cat)
##print(cat_str)
##
##with open(args.odir + "/cat2.txt", "w") as ofile: ofile.write(cat_str)
##enmap.write_map(args.odir + "/map2.fits",   enmap.apply_window(data.maps)/data.fconvs[:,None,None])
##enmap.write_map(args.odir + "/model2.fits", enmap.apply_window(res.model)/data.fconvs[:,None,None])
##enmap.write_map(args.odir + "/resid2.fits", enmap.apply_window(resid_map)/data.fconvs[:,None,None])
##enmap.write_map(args.odir + "/map_snr2.fits",   raw_finder.snr)
##enmap.write_map(args.odir + "/resid_snr2.fits", resid_snr)
##enmap.write_map(args.odir + "/hits2.fits", hits)
##
##cat_t = res.cat
##
### Measure polarization
##for comp in [1,2]:
##	data   = read_data(args.ifiles, sel=sel, box=box, comp=comp)
##	data.freq0 = 98.0
##
##	cmb   = build_cmb_2d(*data.maps.geometry, cl_cmb, dtype=data.maps.dtype)[comp,comp] if cl_cmb is not None else None
##	nmat  = build_nmat_prior(data, type=args.nmat1, cmb=cmb, pol=True)
##	res   = measure_objects(data, cases, nmat, cat_t, resid=True)
##
##	cat_str = format_cat(res.cat)
##	# NB! This is still based on cat2's positions!
##	with open(args.odir + "/cat_pol%d.txt" % comp, "w") as ofile: ofile.write(cat_str)
##	enmap.write_map(args.odir + "/map_pol%d.fits" % comp,   enmap.apply_window(res.maps)/data.fconvs[:,None,None])
##	enmap.write_map(args.odir + "/model_pol%d.fits" % comp, enmap.apply_window(res.model)/data.fconvs[:,None,None])
##	enmap.write_map(args.odir + "/resid_pol%d.fits" % comp, enmap.apply_window(res.resid)/data.fconvs[:,None,None])
##
##	nmat  = build_nmat_empirical(data, noise_map=res.resid, type=args.nmat2)
##	res   = measure_objects(data, cases, nmat, cat_t, resid=True)
##
##	cat_str = format_cat(res.cat)
##	# NB! This is still based on cat2's positions!
##	with open(args.odir + "/cat2_pol%d.txt" % comp, "w") as ofile: ofile.write(cat_str)
##	enmap.write_map(args.odir + "/map2_pol%d.fits" % comp,   enmap.apply_window(res.maps)/data.fconvs[:,None,None])
##	enmap.write_map(args.odir + "/model2_pol%d.fits" % comp, enmap.apply_window(res.model)/data.fconvs[:,None,None])
##	enmap.write_map(args.odir + "/resid2_pol%d.fits" % comp, enmap.apply_window(res.resid)/data.fconvs[:,None,None])
##
### The iterative finder has problems when it encounters data that doesn't follow the
### model. It keeps iterating trying to subtact everything, which can take a long time,
### and which also isn't what we want. We want e.g. dust to be left in the map so that
### it can become part of a position-dependent noise model.
###
### If I could estimate how a source contaminates its surroundings, then one could convolve
### the snr map to get a contamination map, and use that to set a position-dependent snmin.
### That could let me empose a maximum number of iterations, e.g. 3-4, which would read the
### noise floor everywhere but in a contaminated area, and would limit how many fake sources
### could be detected in those. It's not a perfect solution, but it should help.
###
### How much things spread should be given by the real-space form of the filter. But that's
### what the snr map is already. What if one sets a position-dependent snlim map of
### maximum(snmin,np.abs(snr)*tol)? No, that won't work. snr will always be higher than snr*tol,
### so that doesn't actually do anything. Well, at least in a single step. But what about iterative?
###
### 1. snlim = max(snmin,snr*tol), fit and subtract
### 2. snlim = max(snmin,snlim*sntol,snr*tol), fit and subtract, repeat
### That does seem like it should work
###
### Here snmin = ultimate S/N limit (a number) and snlim = current S/N limit (a map)
###
### No, this approach won't work. If snmin is not a big number, then it will end up including
### the whole contaminated area around a bright source in its fit since snr*tol < snr.
###
### The cutoff profile needs to have a shallower slope than the snr profile.
###
###             ^
###          /.....\
###   ..../...     ..\....
### ..  /              \  ..
###   /                 \
###  /                   \
###
### Could convolve it with some arbitrary broadening kernel that preserves peak height,
### multiplied by sntol. How about smooth_gauss(np.abs(snr))/norm * sntol? But norm would
### be tricky, since it will depend on what's being blurred.
###
### Ok, what if we stick with the current iterative approach, but just impose a snr penalty
### based on the local value of np.abs(snr)? So one would have snnoise = max(snmin, smooth(np.abs(snr))*tol)
### snr_corrected = snr/(1+snnoise) or something. But this will require a lot of smoothing to avoid being noisy,
### and that will degrade the ability to handle thin dust structures. And they often are thin.
###
### How about this?
### After each stage, build a catalog and bin it into relatively small rectangles like 0.1°.
### Count the number of objects in each. As this gets higher (maybe after a threshold) the
### local S/N penalty starts increasing. E.g. snr = snr / maximum(1, count-3)
