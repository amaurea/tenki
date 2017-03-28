import numpy as np, os, time, h5py, astropy.io.fits, sys, argparse, copy
from scipy import optimize, stats
from enlib import utils, mpi, fft, enmap, bunch, coordinates
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("odir")
parser.add_argument("-b", "--fwhm",  type=float, default=1.3)
parser.add_argument("-x", "--ignore-ctime", action="store_true")
parser.add_argument("--orad",        type=float, default=10)
parser.add_argument("--ores",        type=float, default=0.1)
args = parser.parse_args()

comm = mpi.COMM_WORLD
utils.mkdir(args.odir)
fwhm = args.fwhm * utils.arcmin

def marg_weights(dchisq, npoint, nsamp=10000):
	"""Compute the marginalization weights for a detection
	with the given chisquare improvement and npoint alternative
	positions to marginalize over, each of which has standard
	normal distributed significance, and hence standard chisquare
	chisq improvements. This weight is explicitly
	< exp(0.5*chi)/(sum_npoint exp(0.5*chi_i) + exp(0.5*dchisq)) >"""
	#chisqs= np.random.standard_normal((npoint,nsamp))**2
	chisqs = np.random.chisquare(1, size=(npoint,nsamp))
	p  = np.exp(0.5*chisqs)
	p0 = np.exp(0.5*dchisq)
	denom = np.sum(p,0)+p0
	return np.mean(p[0]/denom), np.mean(p0/denom)

def marg_pos(pos, pos_cov, dchisq, npoint, R):
	"""Given a maximum-likelihood source position pos and its covariance
	pos_cov and the chisquare improvement of this ML point, compute a
	look elsewhere effect-correction based on having npoint alternative
	positions (this should be multiplied by nmap when fitting multiple
	maps jointly) within a radius R. Returns a new position and cov."""
	w, w0   = marg_weights(dchisq, npoint)
	pos     = w0*pos
	pos_cov = npoint*w*0.5*np.eye(2)*R**2/2 + w0*pos_cov
	return pos, pos_cov

def fixorder(res):
	if res.dtype.byteorder not in ['=','<' if sys.byteorder == 'little' else '>']:
		res = res.byteswap().newbyteorder()
	return res

def read_sdata(ifile):
	# Output thumb for this tod
	with h5py.File(ifile, "r") as hfile:
		sdata = [None for key in hfile]
		for key in hfile:
			ind  = int(key)
			g    = hfile[key]
			sdat = bunch.Bunch()
			# First parse the wcs
			hwcs = g["wcs"]
			header = astropy.io.fits.Header()
			for key in hwcs:
				header[key] = hwcs[key].value
			wcs = enmap.enlib.wcs.WCS(header).sub(2)
			# Then get the site
			sdat.site= bunch.Bunch(**{key:g["site/"+key].value for key in g["site"]})
			# And the rest
			for key in ["map","div","srcpos","sid","vel","fknee","alpha",
					"id", "ctime", "dur", "el", "az", "off"]:
				sdat[key] = g[key].value
			sdat.map = enmap.ndmap(fixorder(sdat.map),wcs)
			sdat.div = enmap.ndmap(fixorder(sdat.div),wcs)
			sdata[ind] = sdat
	return sdata

class BeamModel:
	def __init__(self, fwhm, vel, dec_ref, fknee, alpha, nsigma=2000, nsub=100, order=3):
		# Vel is [dra,ddec]/s.
		sigma  = fwhm/(8*np.log(2))**0.5
		res    = sigma/nsub
		vel    = np.array(vel)
		vel[0]*= np.cos(dec_ref)
		speed  = np.sum(vel**2)**0.5
		# Build coordinate system along velocity
		npoint = 2*nsigma*nsub
		x      = (np.arange(npoint)-npoint/2)*res
		# Build the beam along the velocity
		vbeam  = np.exp(-0.5*(x**2/sigma**2))
		# Apply fourier filter. Our angular step size is res radians. This
		# corresponds to a time step of res/speed in seconds.
		fbeam  = fft.rfft(vbeam)
		freq   = fft.rfftfreq(npoint, res/speed)
		fbeam[1:] /= 1 + (freq[1:]/fknee)**-alpha
		vbeam  = fft.ifft(fbeam, vbeam, normalize=True)
		# Prefilter for fast lookups
		vbeam  = utils.interpol_prefilter(vbeam, npre=0, order=order)
		# The total beam will be this beam times a normal one in the
		# perpendicular direction.
		self.dec_ref = dec_ref
		self.e_para  = vel/np.sum(vel**2)**0.5
		self.e_orto  = np.array([-self.e_para[1],self.e_para[0]])
		self.sigma   = sigma
		self.res     = res
		self.vbeam   = vbeam
		self.order   = order
		# Not really necessary to store these
		self.fwhm  = fwhm
		self.vel   = vel # physical
		self.fknee = fknee
		self.alpha = alpha
	def eval(self, rvec):
		"""Evaluate beam at positions rvec[{dra,dec},...] relative to beam center"""
		# Decompose into parallel and orthogonal parts
		rvec     = np.asanyarray(rvec).copy()
		rvec[0] *= np.cos(self.dec_ref)
		rpara    = np.sum(rvec*self.e_para[:,None,None],0)
		rorto    = np.sum(rvec*self.e_orto[:,None,None],0)
		# Evaluate each beam component
		ipara    = rpara/self.res+self.vbeam.size/2
		bpara    = utils.interpol(self.vbeam, ipara[None], mask_nan=False, order=self.order, prefilter=False)
		borto    = np.exp(-0.5*rorto**2/self.sigma**2)
		res      = enmap.samewcs(bpara*borto, rvec)
		return res

# Single-source likelihood evaluator. Computes chisquares
# for absolute positions in celestial coordinates.
class Srclik:
	def __init__(self, map, div, beam, maxdist=8):
		self.map   = map.preflat[0]
		#enmap.write_map("map.fits", map)
		self.div   = div
		self.posmap= map.posmap()
		self.off   = map.size/3
		self.chisq0= np.sum(map**2*div)
		#enmap.write_map("posmap.fits", self.posmap)
		self.beam  = beam
	def calc_profile(self, pos):
		dpos = self.posmap[::-1] - pos[:,None,None]
		return self.beam.eval(dpos)
	def calc_amp(self, profile):
		ivamp= np.sum(profile**2*self.div)
		if ivamp == 0: return 0, np.inf
		vamp = 1/ivamp
		amp  = vamp*np.sum(profile*self.div*self.map)
		if ~np.isfinite(amp): amp = 0
		return amp, vamp
	def eval(self, pos):
		res = bunch.Bunch()
		res.profile = self.calc_profile(pos)
		res.amp, res.vamp = self.calc_amp(res.profile)
		res.model = res.profile*res.amp
		res.resid = self.map - res.model
		res.chisq = np.sum(res.resid**2*self.div)
		res.chisq0= self.chisq0
		res.marg  = -res.amp**2/res.vamp
		res.pos   = pos
		return res
	def simple_max(self):
		srhs = enmap.smooth_gauss(self.map*self.div, self.beam.sigma)
		sdiv = enmap.smooth_gauss(self.div, self.beam.sigma)
		sdiv = np.maximum(sdiv, np.median(np.abs(sdiv))*0.1)
		smap = srhs/sdiv**0.5
		pos  = enmap.argmax(smap)
		val  = smap.at(pos)
		return pos[::-1], val

# Transform between focalplane coordinates for a given boresight pointing
# and celestial coordinates.
def foc2cel(fpos, site, mjd, bore):
	baz, bel = bore
	hpos = coordinates.recenter(fpos, [0,0,baz,bel])
	cpos = coordinates.transform("hor","cel",hpos,time=mjd,site=site)
	return cpos
def cel2foc(cpos, site, mjd, bore):
	baz, bel = bore
	hpos = coordinates.transform("cel","hor",cpos,time=mjd,site=site)
	fpos = coordinates.recenter(hpos, [baz,bel,0,0])
	return fpos

class DposTransFoc:
	"""Transform between focalplane coordinate *offsets* and celestial
	coordinage *offsets*."""
	def __init__(self, sdat):
		self.mjd  = utils.ctime2mjd(sdat.ctime)
		self.site = sdat.site
		self.ref_cel = sdat.srcpos
		# Find the boresight pointing that defines our focalplane
		# coordinates. This is not exact for two reasons:
		# 1. The array center is offset from the boresight, by about 1 degree.
		# 2. The detectors are offset from the array center by half that.
		# We could store the former to improve our accuracy a bit, but
		# to get #2 we would need a time-domain fit, which is what we're
		# trying to avoid here. The typical error from using the array center
		# instead would be about 1' offset * 1 degree error = 1.05 arcsec error.
		# We *can* easily get the boresight elevation since we have constant
		# elevation scans, so that removes half the error. That should be
		# good enough.
		self.bore= [
				coordinates.transform("cel","hor",sdat.srcpos,time=self.mjd,site=self.site)[0],
				sdat.el ]
		self.ref_foc = cel2foc(self.ref_cel, self.site, self.mjd, self.bore)
	def foc2cel(self, dfoc):
		foc = self.ref_foc + dfoc
		cel = foc2cel(foc, self.site, self.mjd, self.bore)
		dcel = utils.rewind(cel-self.ref_cel)
		return dcel
	def cel2foc(self, dcel):
		cel = self.ref_cel + dcel
		foc = cel2foc(cel, self.site, self.mjd, self.bore)
		dfoc = utils.rewind(foc-self.ref_foc)
		return dfoc

class SrclikMulti:
	def __init__(self, sdata, fwhm, ctrans=DposTransFoc, rmax=6*utils.arcmin):
		self.sdata = sdata
		self.nsrc  = len(sdata)
		self.liks  = []
		self.rmax  = rmax
		for s in sdata:
			beam = BeamModel(fwhm, s.vel, s.srcpos[1], s.fknee, s.alpha)
			lik  = Srclik(s.map, s.div, beam)
			self.liks.append(lik)
		if ctrans is None: ctrans = lambda (dpos,sdat): dpos
		self.trfs  = [ctrans(s) for s in sdata]
	def eval(self, dpos):
		res = bunch.Bunch(chisq=0, chisq0=0, marg=0, amps=[], vamps=[],
			poss=[], models=[], dpos=dpos)
		for lik, trf, s in zip(self.liks, self.trfs, self.sdata):
			dpos_cel = trf.foc2cel(dpos)
			sub = lik.eval(s.srcpos + dpos_cel)
			res.chisq += sub.chisq
			res.chisq0+= sub.chisq0
			res.marg  += sub.marg
			res.amps.append(sub.amp)
			res.vamps.append(sub.vamp)
			res.poss.append(sub.pos)
			res.models.append(sub.model)
		res.amps  = np.array(res.amps)
		res.vamps = np.array(res.vamps)
		# Add prior
		rrel   = np.sum(dpos**2)**0.5/self.rmax
		if rrel > 1:
			penalty = (20*(rrel-1))**2
			res.chisq += penalty
			res.marg  += penalty
		return res

class SrcFitterML:
	def __init__(self, sdata, fwhm, ctrans=DposTransFoc):
		self.sdata = sdata
		self.nsrc  = len(self.sdata)
		self.lik   = SrclikMulti(sdata, fwhm, ctrans=ctrans)
		self.fwhm  = fwhm
		self.scale = 2*utils.arcmin
		self.i     = 0
		self.verbose = False
	def calc_chisq(self, dpos):
		return self.lik.eval(dpos).chisq
	def calc_chisq_wrapper(self, x):
		dpos  = x*self.scale
		chisq = self.calc_chisq(dpos)
		if self.verbose:
			print "%4d %9.4f %9.4f %15.7e" % (self.i, dpos[0]/utils.arcmin, dpos[1]/utils.arcmin, chisq)
		self.i += 1
		return chisq
	def calc_deriv(self, dpos, step=0.05*utils.arcmin):
		return np.array([
			self.calc_chisq(dpos+[step,0])-self.calc_chisq(dpos-[step,0]),
			self.calc_chisq(dpos+[0,step])-self.calc_chisq(dpos-[0,step])])/(2*step)
	def calc_hessian(self, dpos, step=0.05*utils.arcmin):
		return np.array([
			self.calc_deriv(dpos+[step,0])-self.calc_deriv(dpos-[step,0]),
			self.calc_deriv(dpos+[0,step])-self.calc_deriv(dpos-[0,step])
		])/(2*step)
	def calc_full_result(self, dpos, marginalize=True):
		res = self.lik.eval(dpos)
		res.poss_cel = res.poss
		res.poss_hor = np.array([
				coordinates.transform("cel","hor",res.poss_cel[i],utils.ctime2mjd(self.sdata[i].ctime),site=self.sdata[i].site) for i in range(self.nsrc)
			])
		# Get the position uncertainty
		hess  = self.calc_hessian(dpos, step=0.1*utils.arcmin)
		hess  = 0.5*(hess+hess.T)
		try:
			pcov  = np.linalg.inv(0.5*hess)
		except np.linalg.LinAlgError:
			pcov  = np.diag([np.inf,np.inf])
		dchisq  = res.chisq0-res.chisq
		# Apply marginalization correction
		if marginalize:
			R = self.lik.rmax
			A = np.pi*R**2
			Abeam = 2*np.pi*self.fwhm**2/(8*np.log(2))
			npoint= int(np.round(A/Abeam * self.nsrc))
			# Correct position and uncertainty
			dpos, pcov = marg_pos(dpos, pcov, res.chisq0-res.chisq, npoint, R)
			# Correct total chisquare
			dchisq = (stats.norm.ppf(stats.norm.cdf(-dchisq**0.5)*npoint))**2
		ddpos = np.diag(pcov)**0.5
		pcorr = pcov[0,1]/ddpos[0]/ddpos[1]
		res.dpos  = -dpos
		res.ddpos = ddpos
		res.damps = res.vamps**0.5
		res.pcorr = pcorr
		res.nsrc  = self.nsrc
		res.nsigma= dchisq**0.5
		return res
	def find_starting_point(self):
		if True:
			poss, vals = [], []
			for i in range(self.nsrc):
				pos, val = self.lik.liks[i].simple_max()
				poss.append(pos)
				vals.append(val)
			best = np.argmax(vals)
			dpos_cel = poss[best] - self.sdata[best].srcpos
			dpos = self.lik.trfs[best].cel2foc(dpos_cel)
			return dpos
		else:
			bpos, bval = None, np.inf
			for ddec in np.linspace(-self.rmax, self.rmax, self.ngrid):
				for dra in np.linspace(-self.rmax, self.rmax, self.ngrid):
					dpos  = np.array([dra,ddec])
					chisq = self.calc_chisq(dpos)
					print dpos/utils.arcmin, chisq
					if chisq < bval: bpos, bval = dpos, chisq
			return bpos
	def likgrid(self, R, n, marg=False):
		shape, wcs = enmap.geometry(pos=np.array([[-R,-R],[R,R]]), shape=(n,n), proj="car")
		res = enmap.zeros(shape, wcs)
		pos = res.posmap()
		for i,p in enumerate(pos.reshape(2,-1).T):
			L = self.lik.eval(p)
			chisq = L.marg if marg else L.chisq
			print "%6d %7.3f %7.3f %15.7e" % (i, p[0]/utils.arcmin, p[1]/utils.arcmin, chisq)
			res.reshape(-1)[i] = chisq
		return res
	def fit(self, verbose=False, marginalize=True):
		self.verbose = verbose
		t1   = time.time()
		dpos = self.find_starting_point()
		dpos = optimize.fmin_powell(self.calc_chisq_wrapper, dpos/self.scale, disp=False)*self.scale
		res  = self.calc_full_result(dpos, marginalize=marginalize)
		res.time = time.time()-t1
		return res

class SrcFitterMC:
	def __init__(self, sdata, fwhm, ctrans=DposTransFoc, nburn=100, atarg=0.3, thin=3):
		self.sdata = sdata
		self.lik   = SrclikMulti(sdata, fwhm, ctrans=ctrans)
		self.nsamp = 0
		self.naccept = 0
		self.dpos  = np.zeros(2)
		self.marg  = np.inf
		self.step  = 2*utils.arcmin
		self.step_interval = 50
		self.lpar  = None
		self.atarg = atarg
		self.thin  = thin
		self.nburn = nburn
		self.verbose = False
	def draw(self, burnin=False, models=False):
		# Sample position
		t1 = time.time()
		dpos = self.dpos + self.step * np.random.standard_normal(2)
		lpar = self.lik.eval(dpos)
		if not models:
			del lpar.models
		if np.exp(-0.5*(lpar.marg-self.marg)) > np.random.uniform(0,1):
			self.lpar = lpar
			self.dpos = dpos
			self.marg = lpar.marg
			self.naccept += 1
		lpar = self.lpar
		self.nsamp += 1
		# We don't really need to sample amplitude, as this can be
		# done post-hoc using amps and vamps.
		if True and burnin:
			# Tune step length
			if self.nsamp % self.step_interval == 0:
				arate = float(self.naccept)/self.nsamp
				scale = min(2,max(1.0/2,(arate/self.atarg)**0.3))
				self.step *= scale
				print scale, self.step/utils.arcmin, arate, self.atarg, arate/self.atarg
				self.nsamp, self.naccept = 0, 0
		t2 = time.time()
		lpar.time = t2-t1
		return copy.deepcopy(lpar)
	def summarize(self, chain):
		"""Given a chain of lpars, compute a summary in the
		same format as returned by SrcFitterML."""
		dposs    = np.array([c.dpos for c in chain])
		poss_cel = np.array([c.poss for c in chain])
		poss_hor = poss_cel*0
		for i in range(len(self.sdata)):
			poss_hor[:,i] = coordinates.transform("cel","hor",poss_cel[:,i].T,
					time=utils.ctime2mjd(self.sdata[i].ctime), site=self.sdata[i].site).T
		ampss  = np.array([c.amps for c in chain])
		vampss = np.array([c.vamps for c in chain])
		chisq0 = chain[0].chisq0
		chisq  = np.mean([c.chisq for c in chain])
		# Compute means
		dpos    = np.mean(dposs,0)
		pos_cel = np.mean(poss_cel,0)
		pos_hor = np.mean(poss_hor,0)
		amps    = np.mean(ampss,0)
		# And uncertainties
		dpos_cov= np.cov(dposs.T)
		ddpos   = np.diag(dpos_cov)**0.5
		pcorr   = dpos_cov[0,1]/ddpos[0]/ddpos[1]
		# mean([da0_i + da_ij]**2). Uncorrelated, so this is
		# mean(da0**2) + mean(da**2)
		arel    = ampss + np.random.standard_normal(ampss.shape)*vampss**0.5
		amps    = np.mean(arel,0)
		vamps   = np.var(arel,0)
		#vamps   = np.var(ampss,0)+np.mean(vampss,0)
		damps   = vamps**0.5
		models  = self.lik.eval(dpos).models
		time    = sum([c.time for c in chain])
		nsigma  = (chisq0-chisq)**0.5
		# We want how much to offset detector by, not how much to offset
		# source by
		dpos = -dpos
		res  = bunch.Bunch(
				dpos = dpos,  poss_cel=pos_cel, poss_hor=pos_hor,
				ddpos= ddpos, amps=amps, damps=damps, pcorr=pcorr,
				nsrc = len(self.sdata), models=models, time=time,
				nsigma = nsigma, chisq0 = chisq0, chisq = chisq)
		return res
	def fit(self, verbose=False, nsamp=500):
		for i in range(self.nburn):
			for j in range(self.thin):
				lpar = self.draw(burnin=True)
			if verbose:
				print "%4d %9.4f %9.4f %15.7e %6.2f" % (i-self.nburn,
						lpar.dpos[0]/utils.arcmin, lpar.dpos[1]/utils.arcmin, lpar.chisq,
						100.*self.naccept/(max(1,self.nsamp)))
		chain = []
		for i in range(nsamp):
			for j in range(self.thin):
				lpar = self.draw()
			if verbose:
				print "%4d %9.4f %9.4f %15.7e %6.2f" % (i,
						lpar.dpos[0]/utils.arcmin, lpar.dpos[1]/utils.arcmin, lpar.chisq,
						100.*self.naccept/self.nsamp)
			chain.append(lpar)
		res = self.summarize(chain)
		return res

def project_maps(imaps, pos, shape, wcs):
	pos   = np.asarray(pos)
	omaps = enmap.zeros((len(imaps),)+imaps[0].shape[:-2]+shape, wcs, imaps[0].dtype)
	pmap  = omaps.posmap()
	for i, imap in enumerate(imaps):
		omaps[i] = imaps[i].at(pmap+pos[i,::-1,None,None])
	return omaps

# Load source database
#srcpos = np.loadtxt(args.srclist, usecols=(args.rcol, args.dcol)).T*utils.degree
# We use ra,dec ordering in source positions here

# Set up shifted map geometry
shape, wcs = enmap.geometry(
		pos=np.array([[-1,-1],[1,1]])*args.orad*utils.arcmin, res=args.ores*utils.arcmin, proj="car")

# Set up fit output
f = open(args.odir + "/fit_rank_%03d.txt" % comm.rank, "w")

for ind in range(comm.rank, len(args.ifiles), comm.size):
	ifile = args.ifiles[ind]
	sdata = read_sdata(ifile)

	#for i, sdat in enumerate(sdata):
	#	print np.sum(sdat.map**2*sdat.div), sdat.div.size
	#	enmap.write_map("rmap%d.fits" % (i+1), sdat.map*sdat.div**0.5)
	#	print np.array(np.sort((sdat.map*sdat.div**0.5).reshape(-1)))
	#1/0

	## Fill with fake data
	#np.random.seed(2)
	#for i, sdat in enumerate(sdata):
	#	sdat.map[:] = np.random.standard_normal(sdat.map.shape)*np.maximum(sdat.div,1e-10)**-0.5
	#	#enmap.write_map("smap%d.fits"%(i+1),sdat.map)
	#	#print np.sum(sdat.map**2*sdat.div)/np.sum(sdat.div>0)

	# The MC method correctly handles the look elsewhere effect,
	# but it's slow. Can we simulate it with the ML method?
	# We know that in the region inside our prior we effectively
	# have Nb = A_prior/A_beam potential independent source positions.
	# These will each have an amplitude drawn from a standard
	# normal distribution. So we have Nb stdn numbers uniformly
	# distributed in our prior.
	#
	# Consider the simple case of a single source with local
	# significance z_0 at position p_0 with position errors dp.
	# Averaging over the noise would give
	#  ptot = sum(z_i**2 * p_i)/sum(z_i**2)
	# dptot = sqrt(sum(z_i**2 * (dp_i**2 + (p_i-ptot)**2))/sum(z_i**2))
	#  atot = sum(z_i**2 * a_i)/sum(z_i**2)
	# datot = sqrt(sum(z_i**2 * (da_i**2 + (a_i-atot)**2))/sum(z_i**2))
	# This is just an inverse variance weighted average
	# The expectation value of this over the noise is
	# <ptot> = [z_0**2 p_0]/[z_0**2 + Nb]
	#<dptot> = sqrt([z_0**2 dp_0**2 + Nb*(A/Nb + R**2/2)]/(z_0**2+Nb))
	#        = sqrt([z_0**2 dp_0**2 + A + Nb*R**2/2]/(z_0**2+Nb)
	#        = sqrt([z_0**2 dp_0**2 + A(1+Nb/(2pi))]/(z_0**2+Nb))
	# <atot> = [z_0**2 a_0]/[z_0**2 + Nb]
	#<datot> = sqrt([z_0**2 da_0**2 + Nb*(1 + 1 + ztot**2)*Q**2]/(z_0**2+Nb)
	# where Q = a0/da0 and R is the radius of the prior.
	# The overall effect of this is to move us towards the middle of the prior,
	# and to increase the error bars.
	#
	# In my test case, this massively overpredicts the errors. R is much
	# bigger than dp_0. I get sensible numbers if I exclude the position
	# variance term, but that makes no sense: The fake sources do scatter
	# inside the radius R, and this should give a big uncertainty.
	#
	# Let's look at an even simpler case. A pixelated 1-d line from -R to
	# R. The source is located at position 0 with amplitude z(0). All the
	# other values have z <- N(0,1). What is the full likelihood?
	# P(z,x|d) = P(x|z,x)P(z,x)/P(d) = P(x|z,x)
	# P(z,x|d) = 1/sqrt(2pi)**n exp(-0.5*sum_i(d(i)-z*delta(i,x))**2)
	# Integrate out amplitude
	# P(x|d) = int_z P(z,x|d) dz = 1/sqrt(2pi)**(n-1) exp(-0.5*d[not x]**2)
	# Find expected position
	# <x> = sum(x*P(x))/sum(P(x)) = sum(x*exp(-0.5*d[not x]**2))/sum(exp(-0.5*d[not x]**2))
	#     = sum(x*exp(0.5*d_x**2))/sum(exp(0.5*d_x**2))
	# So the weight is exponential in chisq, not just chisq.
	# The expectation value of this is
	# <x> = <x*exp(0.5*r**2)>/[exp(0.5*z0**2)+N*<exp(0.5*r**2)>] = 0
	# var = <sum(x**2*P(x))/sum(P(x))>
	#     = N<x**2*exp(0.5*r**2)>/[w0+(N-1)*<exp(0.5*r**2)>]
	#     = N<x**2>*<exp(0.5*r**2)>/([w0+(N-1)*<exp(0.5*r**2)>]
	# So we need that exp expectation.
	# <exp(0.5*r**2)> = avgint exp(-0.5*r**2)*exp(0.5*r**2) = 1
	# var = N<x**2>/(w0+(N-1))
	# <dx>= R/sqrt(3*[w0+(N-1)]/N)
	# If there is no signal at x=0, this becomes 0.58 R





	sdata=sdata[:]
	if True:
		##sdata = sdata[0:1]
		fitter = SrcFitterML(sdata, fwhm)
		fit    = fitter.fit(verbose=False)
		#cmap = fitter.likgrid(6*utils.arcmin, 80, marg=True)
		#enmap.write_map("mmap.fits", cmap)
		#1/0
	else:
		fitter = SrcFitterMC(sdata, fwhm, nburn=500)
		fit    = fitter.fit(verbose=True, nsamp=10000)

	# Output summary
	for i in range(fit.nsrc):
		ostr = "%s %7.4f %7.4f %7.4f %7.4f %3d %7.4f %7.4f %7.4f %15.7e" % (sdata[i].id,
			fit.dpos[0]/utils.arcmin, fit.ddpos[0]/utils.arcmin,
			fit.dpos[1]/utils.arcmin, fit.ddpos[1]/utils.arcmin,
			sdata[i].sid, fit.amps[i]/1e3, fit.damps[i]/1e3, fit.nsigma, fit.chisq)
		# Add some convenience data
		hour  = sdata[i].ctime/3600.%24
		ostr += " | %5.2f %9.4f %9.4f | %7.4f %2d" % (hour,
				fit.poss_hor[i,0]/utils.degree, fit.poss_hor[i,1]/utils.degree,
				fit.time/fit.nsrc, fit.nsrc)
		print ostr
		sys.stdout.flush()
		f.write(ostr + "\n")
		f.flush()
	# Build shifted models
	smap = project_maps([s.map.preflat[0] for s in sdata], fit.poss_cel, shape, wcs)
	sdiv = project_maps([s.div for s in sdata], fit.poss_cel, shape, wcs)
	smod = project_maps(fit.models, fit.poss_cel, shape, wcs)
	smap = enmap.samewcs([smap,smod,smap-smod],smap)
	# And build scaled coadd. When map is divided by a, div = ivar
	# is multiplied by a**2. Points in very noisy regions can have
	# large error bars in the amplitude, and hence might randomly
	# appear to have a very strong signal. We don't want to let these
	# dominate, so downweight points with atypically high variance.
	# FIXME: Need a better approach than this. This fixed some things,
	# but broke others.
	#beam_exposure = []
	#for i, s in enumerate(sdata):
	#	med_div = np.median(s.div[s.div!=0])
	#	profile = fit.models[i]/fit.amps[i]
	#	avg_div = np.sum(s.div*profile)/np.sum(profile)
	#	beam_exposure.append(avg_div/med_div)
	#beam_exposure = np.array(beam_exposure)
	#weight = np.minimum(1,beam_exposure*1.25)**4
	weight = 1
	trhs = np.sum(smap*sdiv*(fit.amps*weight)[None,:,None,None],1)
	tdiv = np.sum(sdiv*(fit.amps*weight)[:,None,None]**2,0)
	tdiv[tdiv==0] = np.inf
	tmap = trhs/tdiv
	# And write them
	enmap.write_map(args.odir + "/totmap_%s.fits" % sdata[0].id, tmap)
	for i in range(fit.nsrc):
		enmap.write_map(args.odir + "/shiftmap_%s_%03d.fits" % (sdata[i].id, sdata[i].sid), smap[:,i])
		# Output unshifted map too
		omap = enmap.samewcs([sdata[i].map.preflat[0],fit.models[i],sdata[i].map.preflat[0]-fit.models[i]],sdata[i].map)
		enmap.write_map(args.odir + "/fitmap_%s_%03d.fits" % (sdata[i].id,sdata[i].sid), omap)
f.close()
