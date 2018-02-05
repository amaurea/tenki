import numpy as np, os, time, h5py, astropy.io.fits, sys, argparse, copy, warnings
from scipy import optimize, stats, ndimage
from enlib import utils, mpi, fft, enmap, bunch, coordinates
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("odir")
parser.add_argument("-B", "--beam",      type=str,   default="1.3")
parser.add_argument("-m", "--method",    type=str,   default="mlg")
parser.add_argument("-n", "--nsamp",     type=int,   default=1000)
parser.add_argument("-v", "--verbose",   action="count", default=0)
parser.add_argument("-q", "--quiet",     action="count", default=0)
parser.add_argument("-g", "--grid-res",  type=float, default=0.5)
parser.add_argument("-M", "--minimaps",  action="store_true")
parser.add_argument("-p", "--prior",      type=str,   default="semifix")
parser.add_argument("-N", "--num-walker", type=int,   default=10)
args = parser.parse_args()

comm = mpi.COMM_WORLD
utils.mkdir(args.odir)
verbosity  = args.verbose - args.quiet
num_walker = args.num_walker

# Set up beam
try:
	sigma  = float(args.beam)*utils.fwhm * utils.arcmin
	nsigma = 10
	r      = np.arange(0,nsigma,0.01)*sigma
	beam   = np.exp(-0.5*(r/sigma)**2)
except ValueError:
	r, beam = np.loadtxt(args.beam).T
	r *= utils.degree
dr = r[1]

def read_thumb_data(fname):
	res    = bunch.Bunch()
	hdus   = astropy.io.fits.open(fname)
	header = hdus[0].header
	with warnings.catch_warnings():
		wcs = enmap.enlib.wcs.WCS(header).sub(2)
	res.rhs, res.div, res.corr = enmap.fix_endian(enmap.ndmap(hdus[0].data, wcs))
	res.srcinfo = hdus[1].data
	res.detinfo = hdus[2].data
	res.id  = header["id"]
	res.off = np.array([float(header["off_x"]),float(header["off_y"])])*utils.arcmin
	for key in ["bore_az1","bore_az2","bore_el"]:
		res[key] = float(header[key])*utils.degree
	res.ctime = float(header["ctime"])
	return res

def convolve(map, fmap):
	return fft.ifft(fft.fft(map, axes=(-2,-1))*fmap,axes=(-2,-1), normalize=True).real

def mpiwrite(mpifile, msg):
	mpifile.Write_shared(bytearray(msg))
	mpifile.Sync()

class Likelihood:
	def __init__(self, data, beam, dr, prior="uniform", verbose=False):
		self.data = data
		# Beam setup
		self.beam = beam
		self.beam_pre = utils.interpol_prefilter(beam)
		self.dr   = dr
		self.pos = self.data.rhs.posmap()
		self.box = np.sort(self.data.rhs.box(),1)
		# Noise setup
		self.hdiv  = data.div**0.5
		self.ihdiv = self.hdiv.copy()
		self.ihdiv[self.ihdiv>0] **= -1
		# Build fourier-version of correlation
		self.fcorr  = fft.fft(self.data.corr, axes=(-2,-1))
		if np.any(self.fcorr==0): raise ValueError("Invalid noise correlations")
		self.verbose = verbose
		# Prior
		self.prior = prior
		self.i = 0
		self.post0 = None
		self.min_profile = 0.2
	def nmat(self, map): return self.hdiv*convolve(self.hdiv*map,self.fcorr)
	def calc_profile(self, off):
		pos = self.pos + off[:,None,None]
		r   = np.sum(pos**2,0)**0.5
		pix = r/self.dr
		return enmap.samewcs(utils.interpol(self.beam_pre, pix[None], prefilter=False, mask_nan=False), pos)
	def calc_amp(self, profile):
		# (P'N"P)"P'N"d, P = profile, N" = hdiv corr hdiv and d = map
		# N"d should be rhs. Is it? N"d = hdiv corr hdiv ihdiv icorr ihdiv rhs = rhs. Good
		amp_rhs = np.sum(profile*self.data.rhs,(-2,-1))
		amp_div = np.sum(profile*self.nmat(profile),(-2,-1))
		amp = amp_rhs/amp_div
		return amp, amp_div
	def calc_chisq(self, profile):
		amp, amp_div = self.calc_amp(profile)
		return np.sum(amp**2*amp_div)
	@property
	def nsrc(self): return len(self.data.rhs)
	def posterior(self, profile):
		"""Returns -2log(P), so low values are more likely"""
		# Avoid solutions where we can't even see the data
		if np.max(profile) < self.min_profile: return np.inf, np.zeros(self.nsrc), np.zeros(self.nsrc)
		if   self.prior == "uniform":
			# int da 1/sqrt(|2piN|) exp(-0.5*(d-Pa)'N"(d-Pa))
			# int da 1/sqrt(|2piN|) exp(-0.5*[(a-ah)'A"(a-ah) + d'N"d - ah'A"ah])
			# with A" = P'N"P and ah = AP'N"d. Integrating and taking -2log we get
			# -ah'A"ah + d'N"d + log|2piN| - log|2piA|
			# the N terms are constant and can be ignored. Ignoring the A-det term
			# turns out to be equivalent to a jeffrey's prior, which we will use
			# here (so calling it "uniform" is misleading). That leavses us with
			# -ah'A"ah
			amp, amp_div = self.calc_amp(profile)
			return -np.sum(amp**2*amp_div), amp, amp_div
		elif self.prior == "semifix":
			# int da 1/sqrt(|2piN|) exp(-0.5*(d-Pa)'N"(d-Pa)) delta(a-a0)
			# = 1/sqrt(|2piN|) exp(-0.5*(d-Pa0)'N"(d-Pa0))
			# : d'N"d - 2(Pa0)'N"d + (Pa0)'N"(Pa0)
			# = d'N"d - 2(Pa0)'N"d + a0'A"a0
			# = d'N"d - 2 a0'A"a + a0'A"a0
			# Can ignore constant d'N"d term and constant determinant term
			# What if a0 is wrong by factor q => post too high by q**2 => overconfidence
			# So this doesn't fully solve the problem.
			amp, amp_div = self.calc_amp(profile)
			amp0 = self.data.srcinfo.amp
			post = np.sum(amp0*amp_div*(amp0-2*amp))
			return post, amp, amp_div
		elif self.prior == "range":
			raise NotImplementedError
		else:
			raise ValueError
	def full(self, off):
		profile = self.calc_profile(off*utils.arcmin)
		return self.posterior(profile)
	def __call__(self, off):
		off = off*utils.arcmin
		profile = self.calc_profile(off)
		post, amp, amp_div = self.posterior(profile)
		if self.post0 is None and np.isfinite(post): self.post0 = post
		# Flat sky is good enough for our thumbnail
		#off_tot = off[::-1] + self.data.off
		if self.verbose:
			msg = "%5d %8.4f %8.4f" % (self.i, off[0]/utils.arcmin, off[1]/utils.arcmin)
			for a,da in zip(amp, amp_div):
				msg += " %7.3f %4.1f" % (a/1e3, a*da**0.5)
			msg += (" %15.7e" % (self.post0-post) if self.post0 is not None else " inf")
			print msg
		self.i += 1
		return post
	def diag_plot(self, off):
		profile = self.calc_profile(off)
		amp, amp_div = self.calc_amp(profile)
		model_rhs = self.nmat(profile*amp[:,None,None])
		omap = enmap.samewcs(np.concatenate([self.data.rhs, model_rhs, self.data.rhs-model_rhs][::-1],-1),profile)
		return omap

def map_likelihood(likfun, rmax=5*utils.arcmin, res=0.7*utils.arcmin):
	shape, wcs = enmap.geometry(np.array([[-1,-1],[1,1]])*rmax, res=res, proj="car")
	map = enmap.zeros(shape, wcs)
	pos = map.posmap()
	for y in range(map.shape[0]):
		for x in range(map.shape[1]):
			map[y,x] = likfun(pos[:,y,x]/utils.arcmin)
	return map

def draw_positions(loglikmap, nwalker):
	lik  = np.exp(-0.5*(loglikmap-np.min(loglikmap)))
	lik /= np.sum(lik)
	cum  = np.cumsum(lik.reshape(-1))
	ipos = np.searchsorted(cum, np.random.uniform(0,1,size=nwalker))
	ipos = np.array(np.unravel_index(ipos, lik.shape),dtype=float)
	# We now have an integer pixel position. Scatter us randomly
	# inside it
	ipos += np.random.uniform(-0.5,0.5,size=(2,nwalker))
	pos   = lik.pix2sky(ipos).T
	return pos

# Sampling large-amplitude fixed templates with monte carlo will very easily get
# stuck in local minima. This is because a 1 sigma fluctuation in noise on top of
# N-sigma amplitude template will result in a chisquare contribution of (N+1)**2
# approx N**2 + 2N, while a -1 sigma fluctuation will gvie (N-1)**2 approx N**2 - 2N.
# So the expected chisquare flucutations have amplitude 2N, not 1. For high N,
# this means that metropolis would get stuck forever in the first local minimum it
# sees. That doesn't mean that the likelihood is wrong, though - just that metropolis
# can't explore it.
#
# We can get around this by using normal nonlinear optimization to find the true
# minimum, and then sample around that.

class Sampler:
	def __init__(self, likfun, x0, stepscale=2.0, nsamp=200, burn_frac=0.5, verbose=False):
		self.likfun  = likfun
		self.points  = pos
		self.likvals = [likfun.full(p/utils.arcmin) for p in self.points]
		self.next_points  = self.points.copy()
		self.next_likvals = list(self.likvals)
		self.stepscale = stepscale
		self.nsamp   = nsamp
		self.burn_frac = burn_frac
		self.verbose = verbose
		self.i       = 0
	@property
	def npoint(self): return self.points.shape[0]
	@property
	def ndim(self): return self.points.shape[1]
	@property
	def nsrc(self): return self.likvals[0][1].size
	def draw_scale(self):
	# scale pdf: 1/sqrt(z) between 1/a and a. so cdf is:
	# int_(1/a)^x z**-0.5 da = 2[z**0.5]_(1/a)^x = 2*(x**0.5 - a**-0.5)
	# normalize: (a**0.5-a**-0.5)**-1 * (x**0.5-a**-0.5)
	# inverse: ((a**0.5-a**-0.5)*p+a**-0.5)**2
		a = self.stepscale
		return (np.random.uniform(0,1)*(a**0.5-a**-0.5)+a**-0.5)**2
	def draw_sample(self):
		icurrent = self.i % self.npoint
		if icurrent == 0:
			# Start of a new cycle
			self.points  = self.next_points.copy()
			self.likvals = list(self.next_likvals)
		iother   = np.random.randint(self.npoint-1)
		if iother == icurrent: iother += 1
		scale    = self.draw_scale()
		cand_pos = self.points[iother] + scale*(self.points[icurrent]-self.points[iother])
		cand_lik = self.likfun.full(cand_pos/utils.arcmin)
		paccept  = scale**(self.ndim-1)*np.exp(0.5*(self.likvals[icurrent][0] - cand_lik[0]))
		r = np.random.uniform(0,1)
		#print scale, self.points[icurrent], self.likvals[icurrent][0], cand_pos, cand_lik[0], paccept
		if r < paccept:
			self.next_points[icurrent] = cand_pos
			self.next_likvals[icurrent] = cand_lik
		if self.verbose:
			self.print_sample(self.next_points[icurrent], self.next_likvals[icurrent])
		self.i += 1
		# Return our sample in the form pos amp adiv lik
		pos = self.next_points[icurrent]
		lik, amp, adiv = self.next_likvals[icurrent]
		return pos, amp, adiv
	def print_sample(self, pos, likvals):
		lik, amp, adiv = likvals
		msg = "%5d %8.4f %8.4f" % (self.i, pos[0]/utils.arcmin, pos[1]/utils.arcmin)
		for a,da in zip(amp, adiv):
			msg += " %7.3f %4.1f" % (a/1e3, a*da**0.5)
		msg += " %15.7e" % lik
		print msg
	def build_stats(self):
		nburn = int(self.nsamp*self.burn_frac)
		for i in range(nburn):
			self.draw_sample()
		poss  = np.zeros([self.nsamp, self.ndim])
		amps  = np.zeros([self.nsamp, self.nsrc])
		adivs = np.zeros([self.nsamp, self.nsrc])
		for i in range(self.nsamp):
			poss[i], amps[i], adivs[i] = self.draw_sample()
		pos_mean = np.mean(poss,0)
		amp_mean = np.mean(amps,0)
		pos_dev  = np.std(poss,0)
		# Because we're not fully sampling the amplitude, we can't just do np.std(amps,0)
		# here. We need to take into account adiv too. Conceptually we should
		# sample from N(amp,1/adiv) for each amp,adiv, and calculate the stats based
		# on all those. The mean of that will be mean(amp), but the variance will
		# be var(amp) + mean(vars).
		amp_dev  = (np.var(amps,0) + np.mean(1/adivs,0))**0.5
		return bunch.Bunch(pos=pos_mean, dpos=pos_dev, amp=amp_mean, damp=amp_dev)

def format_stats(stats, data):
	msg = ""
	tot_sn = np.sum(stats.amp**2/stats.damp**2)**0.5
	for i in np.argsort(data.srcinfo["amp"])[::-1]:
		dy,  dx  = stats.pos/utils.arcmin
		dy0, dx0 = data.off/utils.arcmin
		ddy, ddx = stats.dpos/utils.arcmin
		sid      = data.srcinfo[i]["sid"]
		afid     = data.srcinfo[i]["amp"]/1e3
		amp, damp = stats.amp[i]/1e3, stats.damp[i]/1e3
		bel      = data.bore_el/utils.degree
		baz      = 0.5*(data.bore_az1+data.bore_az2)/utils.degree
		waz      = (data.bore_az2-data.bore_az1)/utils.degree
		hour     = data.ctime/3600.%24
		src_ra   = data.srcinfo[i]["ra"]
		src_dec  = data.srcinfo[i]["dec"]
		msg += "%s %4d | %8.4f %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f %6.2f %6.2f | %7.2f %7.2f | %5.2f %7.2f %7.2f %7.2f | %8.4f %8.4f\n" % (
				data.id, sid,
				dx, dy, ddx, ddy,
				afid, amp, damp, amp/damp, tot_sn,
				src_ra, src_dec,
				hour, baz, bel, waz,
				dx0, dy0)
	return msg

# Priors:
# 1. No prior. Simple, but suffers from parameter volume effects when including too many
#    sources, leading to overconfident predictions.
# 2. Fix all amps at fiducial. Suboptimal, but may be a good way to find the position.
# 3. Top-hat prior per source, based on fiducial amp */ some factor. The smaller the factor
#    the smaller the volume problem and the more optimal the estimate, as long as it
#    actually contains the real value. Day-time beam variations can cause pretty large
#    apparent gain fluctuations, but these will be common between the sources in the tod.
#    After that we're left with intrinsic variability, which can still be substantial.
#    The 3 strongest sources show night-time amplitude changes of max/min = 2. If we
#    include the day, the variability is max/min = 4. That is a pretty broad span, especially
#    considering that the fiducial value might not be in the middle of that range. In the
#    worst case one might need */ 4, which doesn't do all that much to reduce the volume.
# 4. Volume division. Given N sources, there are more ways for all of them to be high
#    than for all of them to be low. This volume factor should be k*sqrt(sum(amp**2))**(n-1)
#    We can simply divide by that, but that means that if we have a single strong source
#    and 1000 ignorable tiny ones, then the strong one will be biased low by (n-1)/2 log(amp**2)
#    or something. Could rescale by expected amplitude before applying this factor? That way
#    the strong sources won't be more punished than other sources. This seems a bit ad hoc overall.
#
# I'm leaning towards #2. Even if you get the amplitudes a bit wrong, the best fit solution
# should be the same, just with suboptimal errors. The problem is that those errors might
# be too small, thus repeating the overconfidence problem. That happens if the fiducial amplitude
# is significantly higher than the actual ones.

f = open(args.odir + "/fits_%03d.txt" % comm.rank, "w")
#f = mpi.File.Open(comm, args.odir + "/fits.txt", mpi.MODE_WRONLY | mpi.MODE_CREATE)
#f.Set_atomicity(True)

for ind in range(comm.rank, len(args.ifiles), comm.size):
	print ind
	ifile = args.ifiles[ind]
	try:
		thumb_data = read_thumb_data(ifile)
		lik = Likelihood(thumb_data, beam, dr, prior=args.prior, verbose=verbosity>0)
	except Exception as e:
		sys.stderr.write("Exception for %s: %s\n" % (ifile, e.message))
		continue

	print "Processing %s" % thumb_data.id
	oname = thumb_data.id.replace(":","_")
	for si, src in enumerate(thumb_data.srcinfo):
		print "src %4d ra %8.3f dec %8.3f amp %8.3f" % (src["sid"], src["ra"], src["dec"], src["amp"]/1e3)

	# Find out starting point by gridding out the position likelihood coarsly
	likmap  = map_likelihood(lik, res=args.grid_res*utils.arcmin)
	pos     = draw_positions(likmap, num_walker)
	print pos/utils.arcmin
	# Sample to get some statistics
	sampler = Sampler(lik, pos, nsamp=args.nsamp, verbose=verbosity>0)
	stats   = sampler.build_stats()
	# But find the ML point exactly, so we don't have to waste too many samples
	pos_ml  = optimize.fmin_powell(lik, stats.pos/utils.arcmin, disp=0)*utils.arcmin
	stats.pos_ml = pos_ml

	# Output to stdout and to our indiviudal files
	msg     = format_stats(stats, thumb_data)
	f.write(msg + "\n")
	f.flush()
	#mpiwrite(f, msg + "\n")
	print msg
	sys.stdout.flush()

	if args.minimaps:
		diag = lik.diag_plot(stats.pos)
		enmap.write_map(args.odir + "/diag_%s.fits" % oname, diag)
		enmap.write_map(args.odir + "/lik_%s.fits" % oname, likmap)

	#if False:
	#	find_minimum(lik)
	#if False:
	#	likmap = map_likelihood(lik)
	#	enmap.write_map(args.odir + "/grid_%s.fits" % thumb_data.id, likmap)
	#if False:
	#	x0  = np.zeros(2)
	#	x   = optimize.fmin_powell(lik, x0)
	#	off = x*utils.arcmin
	#	diag= lik.diag_plot(off)
	#	enmap.write_map(args.odir + "/diag_%s.fits" % thumb_data.id, diag)
	#if False:
	#	sampler = Sampler(lik, verbose=True)
	#	stats   = sampler.build_stats()

f.close()
