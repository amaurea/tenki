# Fit a global position and per-detector amplitudes to planet tods
from __future__ import division, print_function
import numpy as np, sys, os, h5py
from enlib import config, pmat, mpi, errors, gapfill, utils, enmap, bench
from enlib import fft, array_ops
from enact import filedb, actscan, actdata, cuts

parser = config.ArgumentParser(os.environ["HOME"]+"./enkirc")
parser.add_argument("planet")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("-R", "--dist", type=float, default=0.2)
args = parser.parse_args()

comm = mpi.COMM_WORLD
filedb.init()
ids  = filedb.scans[args.sel]
R    = args.dist * utils.degree
csize= 100

dtype= np.float32
shape= area.shape[-2:]
model_fknee = 10
model_alpha = 10
sys = "hor:"+args.planet + "/0_0"
utils.mkdir(args.odir)
prefix = args.odir + "/"

# Set up a point source model for the planet
srcpos = np.array([[0,0]],float)

# This has a lot of overlap with src_tod_fit and other fitting stuff, and should
# probably have some common functionality extracted.

class PmatTot:
	def __init__(self, data, srcpos, ndir=1, sys=sys):
		# Build source parameter struct for PmatPtsrc
		self.params = np.zeros([srcpos.shape[-1],ndir,8],np.float)
		self.params[:,:,:2] = srcpos[::-1,None,:].T
		self.params[:,:,5:7] = 1
		scan = actscan.ACTScan(data.entry, d=data)
		self.psrc = pmat.PmatPtsrc(scan, self.params, sys=sys)
		self.pcut = pmat.PmatCut(scan)
		# Extract basic offset
		self.off0 = data.point_correction
		self.off  = self.off0*1
		self.el   = np.mean(data.boresight[2,::100])
		self.point_template = data.point_template
	def set_offset(self, off):
		self.off = off*1
		self.psrc.scan.offsets[:,1:] = actdata.offset_to_dazel(self.point_template + off, [0,self.el])
	def forward(self, tod, amps, pmul=1):
		params = self.params.copy()
		params[:,:,2]   = amps
		junk = np.zeros(self.pcut.njunk,tod.dtype)
		self.psrc.forward(tod, params, pmul=pmul)
		self.pcut.forward(tod, junk)
	def backward(self, tod, amps=None, pmul=1):
		params = self.params.copy()
		junk = np.zeros(self.pcut.njunk,tod.dtype)
		self.pcut.backward(tod, junk)
		self.psrc.backward(tod, params, pmul=pmul)
		if amps is None: amps = params[:,:,2]
		else: amps[:] = params[:,:,2]
		return amps

class NmatWhite:
	def __init__(self, ivar):
		self.ivar = ivar
	def apply(self, tod):
		tod *= ivar[:,None]
		return tod

class Likelihood:
	def __init__(self, data, srcpos, amps):
		# Fid nsrc sources with a common position, but individual amplitudes per source and detector
		srcpos = np.zeros([1,1])+srcpos
		self.P = PmatTot(data, srcpos)
		self.N = NmatWhite(data)
		self.tod  = data.tod # might only need the one below
		self.Nd   = self.N.apply(self.tod.copy())
		self.i    = 0
		# Initial values
		self.amp0   = np.zeros([1,1,1])+amps
		self.off0   = self.P.off0
		self.chisq0 = None
		# These are for internal mapmaking
		self.thumb_mapper = ThumbMapper(data, srcpos, self.P.pcut, self.N.nmat)
		self.amp_unit, self.off_unit = 1e3, utils.arcmin
	#def zip(self, off, amps): return np.concatenate([off/self.off_unit, amps[:,0]/self.amp_unit],0)
	#def unzip(self, x): return x[:2]*self.off_unit, x[2:,None]*self.amp_unit
	def zip(self, off): return off/self.off_unit
	def unzip(self, x): return x*self.off_unit
	def fit_amp(self):
		"""Compute the ML amplitude for each point source, along with their covariance"""
		rhs = self.P.backward(self.Nd)
		work = np.zeros(self.tod.shape, self.tod.dtype)
		self.P.forward(work, rhs*0+1)
		self.N.apply(work)
		div  = self.P.backward(work)
		div[div==0] = 1
		return rhs/div, div
	def calc_chisq_fixamp(self, off):
		self.P.set_offset(off)
		amps = self.amp0
		Nr = self.tod.copy()
		self.P.forward(Nr, amps, pmul=-1)
		self.N.apply(Nr)
		PNPa = self.P.backward(Nr)
		return -np.sum(PNPa*amps), amps, amps*0
	def calc_chisq_fitamp(self, off):
		self.P.set_offset(off)
		ahat, aicov = self.fit_amp()
		return -np.sum(ahat**2*aicov), ahat, aicov
	def chisq_wrapper(self, method="fitamp", thumb_path=None, thumb_interval=0, verbose=True):
		if method == "fitamp": fun = self.calc_chisq_fitamp
		else:                  fun = self.calc_chisq_fixamp
		def wrapper(off):
			t1 = time.time()
			chisq, amps, aicov = fun(self.unzip(off))
			t2 = time.time()
			if thumb_path and thumb_interval and self.i % thumb_interval == 0:
				tod2 = self.tod.copy()
				self.P.forward(tod2, amps, pmul=-1)
				thumbs = self.thumb_mapper.map(tod2)
				enmap.write_map(thumb_path % self.i, thumbs)
				del tod2, thumbs
			if self.chisq0 is None: self.chisq0 = chisq
			if verbose:
				msg = "%4d %6.3f %6.3f" % (self.i,off[0],off[1])
				for i in range(len(amps)):
					nsigma = (amps[i,0]**2*aicov[i,0])**0.5
					msg += " %7.3f %4.1f" % (amps[i,0]/self.amp_unit, nsigma)
				msg += " %12.5e %7.2f" % (self.chisq0-chisq, t2-t1)
				print(msg)
			self.i += 1
			return chisq
		return wrapper

for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	bid   = id.replace(":","_")
	entry = filedb.data[id]
	# Read the tod as usual
	try:
		with bench.show("read"):
			d = actdata.read(entry)
		with bench.show("calibrate"):
			d = actdata.calibrate(d, exclude=["autocut"])
		if d.ndet == 0 or d.nsamp < 2: raise errors.DataMissing("no data in tod")
	except errors.DataMissing as e:
		print("Skipping %s (%s)" % (id, e))
		continue
	print("Processing %s" % id)
	# Very simple white noise model
	with bench.show("ivar"):
		tod  = d.tod
		del d.tod
		tod -= np.mean(tod,1)[:,None]
		tod  = tod.astype(dtype)
		diff = tod[:,1:]-tod[:,:-1]
		diff = diff[:,:diff.shape[-1]//csize*csize].reshape(d.ndet,-1,csize)
		ivar = 1/(np.median(np.mean(diff**2,-1),-1)/2**0.5)
		del diff
	# Generate planet cut
	with bench.show("planet cut"):
		planet_cut = cuts.avoidance_cut(d.boresight, d.point_offset, d.site,
				args.planet, R)
	# Subtract atmospheric model
	with bench.show("atm model"):
		model= gapfill.gapfill_joneig(tod, planet_cut, inplace=False)
	# Estimate noise level
	asens = np.sum(ivar)**-0.5 / d.srate**0.5
	print(asens)
	with bench.show("smooth"):
		ft   = fft.rfft(model)
		freq = fft.rfftfreq(model.shape[-1])*d.srate
		flt  = 1/(1+(freq/model_fknee)**model_alpha)
		ft  *= flt
		fft.ifft(ft, model, normalize=True)
		del ft, flt, freq
	with bench.show("atm subtract"):
		tod -= model
		del model
		tod  = tod.astype(dtype, copy=False)
	# Should now be reasonably clean of correlated noise, so we can from now on use
	# a white noise model.
	#with bench.show("pmat"):
	#	P = PmatTot(scan, srcpos, sys=sys)
	#	N = NmatWhite(ivar)
	with bench.show("pmat"):
		pmap = pmat.PmatMap(scan, area, sys=sys)
		pcut = pmat.PmatCut(scan)
		rhs  = enmap.zeros((ncomp,)+shape, area.wcs, dtype)
		div  = enmap.zeros((ncomp,ncomp)+shape, area.wcs, dtype)
		junk = np.zeros(pcut.njunk, dtype)
	with bench.show("rhs"):
		tod *= ivar[:,None]
		pcut.backward(tod, junk)
		pmap.backward(tod, rhs)
	with bench.show("hits"):
		for i in range(ncomp):
			div[i,i] = 1
			pmap.forward(tod, div[i])
			tod *= ivar[:,None]
			pcut.backward(tod, junk)
			div[i] = 0
			pmap.backward(tod, div[i])
	with bench.show("map"):
		idiv = array_ops.eigpow(div, -1, axes=[0,1], lim=1e-5)
		map  = enmap.map_mul(idiv, rhs)
	# Estimate central amplitude
	c = np.array(map.shape[-2:])/2
	crad  = 50
	mcent = map[:,c[0]-crad:c[0]+crad,c[1]-crad:c[1]+crad]
	mcent = enmap.downgrade(mcent, 4)
	amp   = np.max(mcent)
	print("%s amp %7.3f asens %7.3f" % (id, amp/1e6, asens))
	with bench.show("write"):
		enmap.write_map("%s%s_map.fits" % (prefix, bid), map)
		enmap.write_map("%s%s_rhs.fits" % (prefix, bid), rhs)
		enmap.write_map("%s%s_div.fits" % (prefix, bid), div)
	del d, scan, pmap, pcut, tod, map, rhs, div, idiv, junk
