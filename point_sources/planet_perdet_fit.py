# Fit amplitudes and time constants to each detector in a planet tod individually
from __future__ import division, print_function
import numpy as np, argparse, sys, os, h5py
from scipy import optimize
from enlib import mpi, utils, bench, fft, bunch, enmap
from enact import filters

parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("odir")
parser.add_argument("-v", "--verbose", action="count", default=1)
parser.add_argument("-q", "--quiet",   action="count", default=0)
parser.add_argument("-c", "--cont",    action="store_true")
args = parser.parse_args()

comm    = mpi.COMM_WORLD
dtype   = np.float64
verbosity = args.verbose - args.quiet
utils.mkdir(args.odir)

def read_rangedata(fname):
	rdata = bunch.Bunch()
	with h5py.File(fname, "r") as hfile:
		for key in hfile:
			rdata[key] = hfile[key].value
	return rdata

def build_detinds(detmap, dets):
	return np.searchsorted(dets, detmap)

class Model:
	"""The full model, which includes per-detector position, amplitudes and time constants."""
	def __init__(self, rdata):
		self.rdata = rdata
		self.mask  = rdata.ivar > 0
		self.dr    = self.rdata.beam[0,1]-self.rdata.beam[0,0]
	def __call__(self, poss, amps, taus, dets=None):
		if dets is None: dets = np.arange(len(self.rdata.dets))
		rmask   = np.in1d(self.rdata.detmap, dets)
		if np.sum(rmask) == 0: return np.zeros([0,self.rdata.pos.shape[1]],float)
		rpos    = self.rdata.pos[rmask]
		mask    = self.mask[rmask]
		detinds = build_detinds(self.rdata.detmap[rmask], dets)
		# Evaluate the plain beam
		r       = np.sum((rpos - poss[detinds][:,None])**2,-1)**0.5
		bpix    = r/self.dr
		model   = utils.interpol(self.rdata.beam[1], bpix[None], mask_nan=False, order=1)
		# Must mask invalid regions *before* fourier stuff
		model  *= mask
		# Apply the butterworth filter and time constants
		fmodel  = fft.rfft(model)
		tfilters= filters.tconst_filter(self.rdata.freqs[None], taus[:,None])*self.rdata.butter
		fmodel *= tfilters[detinds]
		fft.ifft(fmodel, model, normalize=True)
		# Apply the amplitudes
		model  *= amps[detinds,None]
		return model

class Likelihood:
	def __init__(self, rdata):
		self.rdata = rdata
		self.model = Model(rdata)
		self.ndata = np.sum(self.model.mask)
		self.cache = None
		self.rlim  = 3*utils.arcmin
		self.taulim= [1e-5,1e-1]
		self.prior_scaling = 1e2
		self.chisq0= np.bincount(self.rdata.detmap, np.sum(self.rdata.tod**2*self.rdata.ivar,1), minlength=len(self.rdata.dets))
		#print "chisq0", self.chisq0
	def prior(self, poss, amps, taus):
		penalty  = np.maximum(np.sum(poss**2,1)/self.rlim**2-1,0)
		penalty += np.maximum(self.taulim[0]**2/taus**2-1,0)
		penalty += np.maximum(taus**2/self.taulim[1]**2-1,0)
		penalty *= self.prior_scaling
		return penalty
	def calc_chisqs_amps(self, poss, taus, dets=None):
		# Restrict to the given detectors
		if dets is None: dets = np.arange(len(self.rdata.dets))
		ndet   = len(dets)
		rmask = np.in1d(self.rdata.detmap, dets)
		data  = self.rdata.tod[rmask]
		ivar  = self.rdata.ivar[rmask]
		detinds = build_detinds(self.rdata.detmap[rmask], dets)
		# int da 1/sqrt(|2piN|) exp(-0.5*(d-Pa)'N"(d-Pa))
		# int da 1/sqrt(|2piN|) exp(-0.5*[(a-ah)'A"(a-ah) + d'N"d - ah'A"ah])
		# with A" = P'N"P and ah = AP'N"d. Integrating and taking -2log we get
		# -ah'A"ah + d'N"d + log|2piN| - log|2piA|
		# the N terms are constant and can be ignored. Ignoring the A-det term
		# turns out to be equivalent to a jeffrey's prior, which we will use
		# here. That leaves us with -ah'A"ah
		adummy = np.full(ndet, 1.0)
		profile= self.model(poss, adummy, taus, dets=dets)
		arhs   = np.bincount(detinds, np.sum(profile*ivar*data,-1),    minlength=ndet)
		adiv   = np.bincount(detinds, np.sum(profile*ivar*profile,-1), minlength=ndet)
		adiv   = np.maximum(adiv, 1e-12)
		ahat   = arhs/adiv
		norm   = -np.log(adiv/(2*np.pi))
		if False:
			foomask = dets == 5
			if np.sum(foomask) > 0:
				print("ahat", ahat[foomask], adiv[foomask])
				print("foomask", foomask)
				print("dets", dets)
				print("ahat full", ahat)
				print("chisqs", self.chisq0[dets][foomask], ahat[foomask]**2*adiv[foomask], norm[foomask])
				print("prior", self.prior(poss, ahat, taus)[foomask])
		chisqs = self.chisq0[dets] - ahat**2*adiv + norm
		chisqs+= self.prior(poss, ahat, taus)
		return chisqs, ahat, adiv
	def calc_chisqs_amps_cached(self, poss, taus):
		"""Calc the total chisquare for all detectos, but reuse contributions
		from unchanged detectors."""
		if self.cache is None:
			chisqs, amps, adiv = self.calc_chisqs_amps(poss, taus)
			self.cache = bunch.Bunch(
					poss = poss.copy(), taus = taus.copy(),
					chisqs = chisqs, amps = amps, adiv = adiv)
		else:
			# Find the set of changed detectors
			changed = np.any(poss != self.cache.poss,1)|(taus != self.cache.taus)
			dets    = np.where(changed)[0]
			if len(dets) > 0:
				#print "cached call"
				#print " with dets", dets
				#print " with poss", poss[dets]
				#print " with taus", taus[dets]
				chisqs, amps, adiv = self.calc_chisqs_amps(poss[dets], taus[dets], dets)
				self.cache.chisqs[dets] = chisqs
				self.cache.amps[dets] = amps
				self.cache.adiv[dets] = adiv
				self.cache.poss[dets] = poss[dets]
				self.cache.taus[dets] = taus[dets]
			#print "raw call"
			#print " with poss", poss
			#print " with taus", taus
			#chisqs2, amps2, adiv2 = self.calc_chisqs_amps(poss, taus)
			## Find actual changes
			#changed2 = chisqs2 != self.cache.chisqs
			#print np.sum(self.cache.chisqs), np.sum(chisqs2)
			#print "A", np.where(changed), np.where(changed2)
			#print "B", self.cache.chisqs[changed], chisqs2[changed2]
		return self.cache.chisqs.copy(), self.cache.amps.copy(), self.cache.adiv.copy()
	#def calc_chisqs_raw(self, poss, amps, taus, dets=None):
	#	"""Given poss[nuse,2], amps[nuse], taus[nuse] and optionally dets[nuse],
	#	compute the chisquare for every detector indicated in dets. dets is
	#	a list of detector indices, not the det uid. These
	#	can be summed to get the total chisq"""
	#	if dets is None: dets = np.arange(len(self.rdata.dets))
	#	ndet  = len(dets)
	#	rmask = np.in1d(self.rdata.detmap, dets)
	#	model = self.model(poss, amps, taus, dets=dets)
	#	data  = self.rdata.tod[rmask]
	#	ivar  = self.rdata.ivar[rmask]
	#	_, detinds = np.unique(self.rdata.detmap[rmask], return_inverse=True)
	#	resid = data-model
	#	# Find chisquare per detector
	#	rchisq  = np.sum(resid**2*ivar,1)
	#	chisqs  = np.bincount(detinds, rchisq, minlength=ndet)
	#	chisqs += self.prior(poss, amps, taus)
	#	return chisqs
	#def calc_chisqs_cached(self, poss, amps, taus):
	#	"""Calc the total chisquare for all detectos, but reuse contributions
	#	from unchanged detectors."""
	#	if self.cache is None:
	#		self.cache = bunch.Bunch(
	#				poss = poss.copy(), amps = amps.copy(), taus = taus.copy(),
	#				chisqs = self.calc_chisqs_raw(poss, amps, taus))
	#	else:
	#		# Find the set of changed detectors
	#		changed = np.any(poss != self.cache.poss,1)|(amps != self.cache.amps)|(taus != self.cache.taus)
	#		dets    = np.where(changed)[0]
	#		if len(dets) > 0:
	#			self.cache.chisqs[dets] = self.calc_chisqs_raw(poss[dets], amps[dets], taus[dets], dets)
	#			self.cache.poss[dets] = poss[dets]
	#			self.cache.amps[dets] = amps[dets]
	#			self.cache.taus[dets] = taus[dets]
	#	return self.cache.chisqs.copy()
	def wrapper(self, param_zipper, verbosity=0):
		class fun:
			def __init__(fself): fself.i = 0
			def __call__(fself, x):
				params = param_zipper.unzip(x)
				chisqs, amps, adiv = self.calc_chisqs_amps_cached(params.poss, params.taus)
				#print amps[433], adiv[433]
				chisq  = np.sum(chisqs)
				if verbosity > 0 and fself.i % 10 == 0:
					if verbosity == 1:
						print("%6d %12.4f" % (fself.i, chisq/self.ndata))
					else:
						print("%6d %12.4f" % (fself.i, chisq/self.ndata) + " %6.3f"*len(x) % tuple(x))
					sys.stdout.flush()
					if False and fself.i % 10000 == 0:
						model = self.model(params.poss, amps, params.taus)
						map = enmap.enmap([self.rdata.tod,model,self.rdata.tod-model])
						enmap.write_map(args.odir + "/foo%06d.fits" % fself.i, map)
				fself.i += 1
				return chisq
		return fun()

def calc_errors(likfun, zipper, params, delta=1e-3, keys=["poss","taus"]):
	xvals = zipper.zip(params)
	# First compute the second derivative along each axis
	def offx(i,x):
		res = xvals.copy()
		res[i] = x
		return res
	d2 = [ (likfun(offx(i,x-delta)) - 2*likfun(offx(i,x)) + likfun(offx(i,x+delta)))/delta**2 for i,x in enumerate(xvals)]
	d2 = np.array(d2)
	# dd(a'A'a)/dada = 2 A', so sigma = (0.5*d2)**-0.5
	xsigma = (0.5*d2)**-0.5
	# Then translate this uncertainty into the actual parameter uncertainties
	p1 = zipper.unzip(xvals)
	p2 = zipper.unzip(xvals + xsigma*delta)
	oparams = params.copy()
	for key in keys:
		err = (p2[key]-p1[key])/delta
		err[~np.isfinite(err)] = np.inf # get rid of nans
		oparams["d"+key] = err
	return oparams

def debug(likfun, zipper):
	params = zipper.unzip(np.full(zipper.nparam,1.0))
	params.poss[:] = np.array([-9.4540568e-06,-4.4684049e-05])
	taus = np.linspace(1e-5,1e-1,5000)
	liks = []
	for tau in taus:
		params.taus[433] = tau
		liks.append(likfun(zipper.zip(params)))
	np.savetxt("test.txt", np.array([taus,liks]).T, fmt="%30.20e")
	1//0


#class ZipParamsIndividual:
#	def __init__(self, ndet, scale):
#		self.ndet, self.scale = ndet, scale
#		self.nparam = 4*self.ndet
#	def zip(self, vals): return np.concatenate([
#		x.reshape(-1) for x in [vals.amps/self.scale.amps, vals.pos/self.scale.poss, np.log(vals.taus/self.scale.taus)]])
#	def unzip(self, x):
#		return bunch.Bunch(
#				amps = x[0:self.ndet]*self.scale.amps,
#				poss = x[self.ndet:self.ndet*3].reshape(-1,2)*self.scale.poss,
#				taus = np.exp(x[self.ndet*3:self.ndet*4])*self.scale.taus)
#
#class ZipParamsCommonPos:
#	def __init__(self, ndet, scale):
#		self.ndet, self.scale = ndet, scale
#		self.nparam = 2+2*self.ndet
#	def zip(self, vals): return np.concatenate([
#		x.reshape(-1) for x in [vals.pos[0]/self.scale.poss, vals.amps/self.scale.amps, np.log(vals.taus/self.scale.taus)]])
#	def unzip(self, x):
#		return bunch.Bunch(
#				poss = np.tile(x[None,:2]*self.scale.poss, (self.ndet,1)),
#				amps = x[2:2+self.ndet]*self.scale.amps,
#				taus = np.exp(x[2+self.ndet:2+self.ndet*2])*self.scale.taus)

class ZipParamsCommonPosNoamp:
	def __init__(self, ndet, scale):
		self.ndet, self.scale = ndet, scale
		self.nparam = 2+self.ndet
	def zip(self, vals): return np.concatenate([
		x.reshape(-1) for x in [vals.poss[0]/self.scale.poss, np.log(vals.taus/self.scale.taus)]])
	def unzip(self, x):
		return bunch.Bunch(
				poss = np.tile(x[None,:2]*self.scale.poss, (self.ndet,1)),
				taus = np.exp(x[2:2+self.ndet])*self.scale.taus)

def write_params(fname, params):
	with h5py.File(fname, "w") as hfile:
		for key in params:
			hfile[key] = params[key]

def make_dummy(fname, msg=""):
	with open(fname, "w") as f:
		f.write(msg + "\n")

for ind in range(comm.rank, len(args.ifiles), comm.size):
	ifile = args.ifiles[ind]
	# This is ugly, since it assumes a given filename format for the input files.
	id    = os.path.basename(ifile).replace("rdata_","").replace(".hdf","")
	ofile = args.odir + "/fit_%s.hdf" % id
	odummy= args.odir + "/fit_%s.dummy" % id
	if args.cont and (os.path.isfile(ofile) or os.path.isfile(odummy)):
		print("Skipping %s (already done)" % id)
		continue
	rdata = read_rangedata(ifile)
	print("Processing %s" % rdata.id)

	L  = Likelihood(rdata)
	# Minimize the chisquare
	scale  = bunch.Bunch(poss=1*utils.arcmin, amps=1e6, taus=1e-3)
	zipper = ZipParamsCommonPosNoamp(len(rdata.dets), scale)
	likfun = L.wrapper(zipper, verbosity=verbosity)
	#debug(likfun, zipper)
	x0     = np.full(zipper.nparam, 1.0)
	x, chisq, _, niter, _, _ = optimize.fmin_powell(likfun, x0, disp=False, full_output=True)
	params = zipper.unzip(x)
	params = calc_errors(likfun, zipper, params)
	# Get the amplitudes too
	chisqs, params.amps, adiv = L.calc_chisqs_amps(params.poss, params.taus)
	params.damps = adiv**-0.5
	# Get the diagonal-fisher uncertainty

	ndatas = np.bincount(rdata.detmap, np.sum(rdata.ivar>0,1), minlength=len(rdata.dets))
	# Ok, we have the solution. Output it
	params.chisqs= chisqs / ndatas
	params.chisq = np.sum(chisqs)
	params.dets  = rdata.dets
	params.ndatas= ndatas
	params.ndata = np.sum(ndatas)
	params.niter = niter
	write_params(ofile, params)
