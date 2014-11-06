"""Unlike srclik_join, this program keeps the global parameters fixed,
and samples each TOD independently. The parameters are offset[{dec,ra}],
beam[3], amps[nsrc]. The amplitudes have a gaussian conditional distribution,
and can be samples directly. So we will use Gibbs sampling. That should make
this very fast.

Most of the point sources are very faint, with S/N < 0.1. We still sample them
individually per tod. The result will be a large number of noise-dominated samples.
These can then later be binned across tods in a different program to look for
source variability and gain variability."""

import numpy as np, os, mpi4py.MPI, time, h5py, sys, warnings
from enlib import utils, config, ptsrc_data, log, bench, cg, array_ops
from enlib.degrees_of_freedom import DOF, Arg

warnings.filterwarnings("ignore")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("filelist")
parser.add_argument("srcs")
parser.add_argument("odir")
parser.add_argument("--ncomp", type=int, default=1)
parser.add_argument("--nsamp", type=int, default=300)
parser.add_argument("--burnin",type=int, default=100)
parser.add_argument("--thin", type=int, default=2)
parser.add_argument("--minrange", type=int, default=100)
parser.add_argument("-R", "--radius", type=float, default=5.0)
parser.add_argument("-r", "--resolution", type=float, default=0.25)
parser.add_argument("-d", "--dump", type=int, default=100)
parser.add_argument("--freeze-beam", type=float, default=0)
args = parser.parse_args()

verbosity = config.get("verbosity")

comm  = mpi4py.MPI.COMM_WORLD
ncomp = args.ncomp
dtype = np.float32
d2r   = np.pi/180
m2r   = np.pi/180/60
b2r   = np.pi/180/60/(8*np.log(2))**0.5

# prior on beam
beam_min = 0.6*b2r
beam_max = 3.0*b2r
beam_ratio_max = 3.0
# prior on position
pos_deviation_max = 2*m2r

log_level = log.verbosity2level(verbosity)
L = log.init(level=log_level, rank=comm.rank)
bench.stats.info = [("time","%6.2f","%6.3f",1e-3),("cpu","%6.2f","%6.3f",1e-3),("mem","%6.2f","%6.2f",2.0**30),("leak","%6.2f","%6.3f",2.0**30)]

# Allow filelist to take the format filename:[slice]
toks = args.filelist.split(":")
filelist, fslice = toks[0], ":".join(toks[1:])
filelist = [line.split()[0] for line in open(filelist,"r") if line[0] != "#"]
filelist = eval("filelist"+fslice)
ntod = len(filelist)

utils.mkdir(args.odir)
srcs = np.loadtxt(args.srcs)
nsrc = len(srcs)

# Our fixed global parameters are pos[nsrc,{dec,ra}]
pos  = srcs[:,[3,5]] * d2r
bfid = srcs[0,19]    * b2r

# BEAMS
# We will represent the beam as sigma[2], phi
def amp2pflux(amp, beam):
	return amp*np.product(beam[:2])**0.5
def pflux2amp(pflux, beam):
	return pflux/np.product(beam[:2])**0.5

# Define our sampling scheme
class SrcSampler:
	def __init__(self, data, pos, off0, beam0, amp0, doff, dbeam, damp):
		self.data = data
		self.pos  = pos
		self.off  = off0.copy()
		self.beam = beam0.copy()
		self.pflux= amp2pflux(amp0,beam0)
		self.doff = doff
		self.dpflux= amp2pflux(damp,beam0)
		self.dbeam= dbeam

		self.step = 0
		self.metro_nsamp = None
		self.metro_naccept = None
		self.A_groups = None
	def get_params(self, off, beam, pflux):
		p = np.zeros([len(self.pos), 2+3+pflux.shape[1]])
		p[:,:2]   = self.pos + off[None,:]
		p[:,2:-3] = pflux2amp(pflux,beam)
		p[:,-3:]  = utils.compress_beam(beam[:2],beam[2])[None,:]
		return p
	def get_model(self, params):
		tod = self.data.tod.astype(dtype)
		params = params.astype(dtype)
		ptsrc_data.pmat_model(tod, params, self.data)
		return tod
	def lik(self, off, beam, pflux):
		sigma, phi = beam[:2],beam[2]
		if not np.all(np.isfinite(sigma)): return np.inf
		if np.max(sigma) > beam_max: return np.inf
		if np.min(sigma) < beam_min: return np.inf
		if np.max(sigma)/np.min(sigma) > beam_ratio_max: return np.inf
		if np.sum(off**2)**0.5 > pos_deviation_max: return np.inf
		params = self.get_params(off, beam, pflux).astype(dtype)
		tod    = self.data.tod.astype(dtype)
		prob  = 0.5*np.sum(ptsrc_data.chisq_by_range(tod, params, self.data))
		return prob
	def draw_pflux(self):
		# Maximum likelihood amplitude given by P'N"Pa = P'N"d.
		# But we want to draw a sample from the distribution.
		# Mean value given by the above. Variance given by (P'N"P)",
		# So to sample, must add (P'N"P)**0.5 r to rhs
		params = self.get_params(self.off, self.beam, self.pflux).astype(dtype)
		# rhs = P'N"d
		tod    = self.data.tod.astype(dtype, copy=True)
		ptsrc_data.nmat_mwhite(tod, self.data)
		ptsrc_data.pmat_model(tod, params, self.data, dir=-1)
		rhs    = params[:,2:-3]
		dof    = DOF(Arg(default=rhs))
		# Set up functional form of A
		def Afun(x):
			p = params.copy()
			p[:,2:-3], = dof.unzip(x)
			ptsrc_data.pmat_model(tod, p, self.data, dir=+1)
			ptsrc_data.nmat_mwhite(tod, self.data)
			ptsrc_data.pmat_model(tod, p, self.data, dir=-1)
			return dof.zip(p[:,2:-3])
		# Compute our A matrix, which is almost diagonal
		def build_A_brute(dof):
			A = np.zeros([dof.n,dof.n])
			# Brute force construction
			I = np.eye(dof.n)
			correlated_groups = []
			group_done = np.zeros(dof.n,dtype=bool)
			for i in range(dof.n):
				A[i] = Afun(I[i])
				if group_done[i]: continue
				hit  = np.where(A[i]!=0)[0]
				correlated_groups.append(hit)
				group_done[hit] = True
			# Build uncorrelated groups
			return A, correlated_groups
		def build_A_fast(dof, ucorr):
			# Construct independent parts of matrix in parallel
			A = np.zeros([dof.n,dof.n])
			nmax = max([len(g) for g in ucorr])
			for i in range(nmax):
				# Loop through the elements of the uncorrelated groups in parallel
				u = np.zeros(dof.n)
				u[[g[i] for g in ucorr if len(g) > i]] = 1
				Au = Afun(u)
				# Extract result into full A
				for g in ucorr:
					if len(g) > i:
						A[g[i],g] = Au[g]
			return A

		if self.A_groups is None:
			A, self.A_groups = build_A_brute(dof)
		else:
			A = build_A_fast(dof, self.A_groups)
		try:
			Ah = np.linalg.cholesky(A)
			x = np.linalg.solve(A,dof.zip(rhs) +  Ah.dot(np.random.standard_normal(dof.n)))
		except np.linalg.LinAlgError:
			Ah = array_ops.eigpow(A, 0.5)
			Ai = array_ops.eigpow(A, -1)
			x = Ai.dot(dof.zip(rhs) + Ah.dot(np.random.standard_normal(dof.n)))
		amps, = dof.unzip(x)
		pflux = amp2pflux(amps, self.beam)

		## compare likelihood of before and after
		#L_before = self.lik(self.off, self.beam, self.pflux)
		#L_after  = self.lik(self.off, self.beam, pflux)
		#print "dlik %10.5f %10.5f %10.5f" % (L_before, L_after, L_before-L_after)

		return pflux
	def draw_pos_beam(self, nstep=1, verbose=False):
		# Sample pos and beam using a simple Metropolis sampler
		off   = self.off.copy()
		beam  = self.beam.copy()
		dof   = DOF(Arg(array=off),Arg(array=beam))
		x     = dof.zip(off,beam)
		dx    = dof.zip(self.doff, self.dbeam)
		L = self.lik(off, beam, self.pflux)

		if self.metro_nsamp is None:
			self.metro_nsamp   = np.zeros(dof.n,dtype=int)
			self.metro_naccept = np.zeros(dof.n,dtype=int)
		for i in range(nstep):
			for j in range(dof.n):
				if dx[j] == 0: continue
				x_new = x.copy()
				x_new[j] = x[j] + np.random.standard_normal()*dx[j]
				off_new, beam_new =dof.unzip(x_new)
				L_new = self.lik(off_new, beam_new, self.pflux)
				if np.random.uniform() < np.exp(L-L_new):
					x, L = x_new, L_new
					self.metro_naccept[j] += 1
				self.metro_nsamp[j] += 1
		if verbose:
			sigma, phi = beam[:2], beam[2]
			accepts = tuple(self.metro_naccept.astype(float)/self.metro_nsamp)
			vals  = tuple(off/m2r) + tuple(sigma/b2r) + (phi/d2r,) 
			print "%5d" % self.step,
			for j in range(dof.n):
				print " %10.5f %5.3f" % (vals[j],accepts[j]),
			print "%12.3f" % L
			sys.stdout.flush()
		return dof.unzip(x)
	def draw_metro_all(self, nstep=1, verbose=False):
		# Sample pos and beam using a simple Metropolis sampler
		off   = self.off.copy()
		beam  = self.beam.copy()
		pflux = self.pflux.copy()
		dof   = DOF(Arg(array=off),Arg(array=beam),Arg(array=pflux))
		x     = dof.zip(off,beam,pflux)
		dx    = dof.zip(self.doff, self.dbeam, self.dpflux)
		L = self.lik(off, beam, pflux)

		if self.metro_nsamp is None:
			self.metro_nsamp   = np.zeros(dof.n,dtype=int)
			self.metro_naccept = np.zeros(dof.n,dtype=int)
		for i in range(nstep):
			for j in range(dof.n):
				x_new = x.copy()
				x_new[j] = x[j] + np.random.standard_normal()*dx[j]
				off_new, beam_new, pflux_new =dof.unzip(x_new)
				L_new = self.lik(off_new, beam_new, pflux_new)
				if np.random.uniform() < np.exp(L-L_new):
					x, L = x_new, L_new
					self.metro_naccept[j] += 1
				self.metro_nsamp[j] += 1
		if verbose:
			off, beam, pflux = dof.unzip(x)
			sigma, phi = beam[:2], beam[2]
			amp = pflux2amp(pflux, beam)
			accepts = tuple(self.metro_naccept.astype(float)/self.metro_nsamp)
			vals  = tuple(off/m2r) + tuple(sigma/b2r) + (phi/d2r,) + tuple(amp)
			print "%5d" % self.step,
			for j in range(dof.n):
				print " %10.5f %5.3f" % (vals[j],accepts[j]),
			print "%12.3f" % L
			sys.stdout.flush()
		return dof.unzip(x)
	def draw(self, verbose=False, fullmetro=False):
		if fullmetro:
			with bench.mark("draw_metro"):
				self.off, self.beam, self.pflux = self.draw_metro_all(verbose=verbose)
		else:
			with bench.mark("draw_amps"):
				self.pflux = self.draw_pflux()
			with bench.mark("draw_pos_beam"):
				self.off, self.beam = self.draw_pos_beam(verbose=verbose)
		sigma, phi = self.beam[:2], self.beam[2]
		self.step += 1
		return self.off, np.concatenate([sigma,[phi]]), pflux2amp(self.pflux,self.beam)

#def beam_prior(sigma, phi):
#	"""We want a uniform prior in log(sigma) and phi."""
#	# First do the beam to ibeam part
#	c,s = np.cos(phi), np.sin(phi)
#	b2, b3 = sigma**-2, sigma**-3
#	db = b2[0]-b2[1]
#	J_ib_b = np.array([
#			[ -2*c*c*b3[0], +2*s*s*b3[1], -2*s*c*db ],
#			[ +2*s*s*b3[0], -2*c*c*b3[1], +2*c*s*db ],
#			[ -2*c*s*b3[0], +2*s*c*b3[1], (c*c-s*s)*db ]])
#	det_J_ib_b = np.log(np.abs(np.linalg.det(J_ib_b)))
#	# Then do the log(sigma) to sigma part. This one is diagonal
#	det_J_ls_s = np.log(np.abs(1/np.product(sigma)))
#	# Combine into a full prior
#	return -det_J_ib_b+det_J_ls_s

def make_maps(tod, data, pos, ncomp, radius, resolution):
	tod = tod.copy()
	nsrc= len(pos)
	# Set up pixels
	n   = int(np.round(2*radius/resolution))
	boxes = np.array([[p-radius,p+radius] for p in pos])
	# Set up output maps
	rhs  = np.zeros([nsrc,ncomp,n,n],dtype=dtype)
	div  = np.zeros([ncomp,nsrc,ncomp,n,n],dtype=dtype)
	# Build rhs
	ptsrc_data.nmat_mwhite(tod, data)
	ptsrc_data.pmat_thumbs(-1, tod, rhs, data.point, data.phase, boxes)
	# Build div
	for c in range(ncomp):
		idiv = div[0].copy(); idiv[:,c] = 1
		wtod = data.tod.astype(dtype,copy=True); wtod[...] = 0
		ptsrc_data.pmat_thumbs( 1, wtod, idiv, data.point, data.phase, boxes)
		ptsrc_data.nmat_mwhite(wtod, data, 0.0)
		ptsrc_data.pmat_thumbs(-1, wtod, div[c], data.point, data.phase, boxes)
	div = np.rollaxis(div,1)
	bin = rhs/div[:,0] # Fixme: only works for ncomp == 1
	return bin, rhs, div

# Process all the tods, one by one
for ind in range(comm.rank, ntod, comm.size):
	fname = filelist[ind]
	try:
		id = fname[fname.rindex("/")+1:]
	except ValueError:
		id = fname
	L.debug("Processing %s" % id)
	# Make our per-tod output dir
	tdir = args.odir + "/" + id
	utils.mkdir(tdir)

	d = ptsrc_data.read_srcscan(fname)

	# Validate the noise model
	vars, nvars = ptsrc_data.measure_mwhite(d.tod, d)
	ivars = np.sum(nvars,0)/np.sum(vars,0)
	# I don't trust the supplied noise model. 
	d.ivars = ivars
	#print d.ivars/ivars
	#1/0

	# Discard sources that aren't sufficiently hit
	srcmask = d.offsets[:,-1]-d.offsets[:,0] > args.minrange
	# We don't like detectors the noise properties vary too much.
	detmask = np.zeros(len(ivars),dtype=bool)
	for di, (dvar,ndvar) in enumerate(zip(vars.T,nvars.T)):
		dhit = ndvar > 20
		dvar, ndvar = dvar[dhit], ndvar[dhit]
		mean_variance = np.sum(dvar)/np.sum(ndvar)
		individual_variances = dvar/ndvar
		# It is dangerous if the actual variance in a segment of the tod
		# is much higher than what we think it is.
		detmask[di] = np.max(individual_variances/mean_variance) < 3
		#print "%5d %12.5f %12.5f %12.5f" % (di, mvar*d.ivars[di], std*d.ivars[di], np.max(dvar/ndvar)*d.ivars[di])
		#detmask[di] = std < mvar * 0.1
	hit_srcs = np.where(srcmask)[0]
	hit_dets = np.where(detmask)[0]

	d = d[hit_srcs,hit_dets]
	my_nsrc, my_ndet = len(hit_srcs), len(hit_dets)
	my_pos = pos[hit_srcs]

	#nvars, vars = nvars[hit_srcs][:,hit_dets], vars[hit_srcs][:,hit_dets]
	#ivars_src = nvars/vars
	#for j in range(my_ndet):
	#	for i in range(my_nsrc):
	#		if d.offsets[i,-1]-d.offsets[i,0] <= args.minrange: continue
	#		if np.isfinite(ivars_src[i,j]):
	#			print "%4d %4d %12.5f %15.7e %15.7e %5d" % (i,j,(ivars_src[i,j]/d.ivars[j])**-1,(ivars_src[i,j])**-1,(d.ivars[j])**-1,nvars[i,j])

	# Make minimaps of our data, for visual inspection
	dmaps, drhs, ddiv = make_maps(d.tod.astype(dtype), d, my_pos, ncomp, args.radius*m2r, args.resolution*m2r)
	with h5py.File(tdir + "/data.hdf", "w") as hfile:
		hfile["data"] = dmaps
		hfile["rhs"]  = drhs
		hfile["div"]  = ddiv

	# Start off with zero shift and a standard beam
	off0  = np.array([0.0,0.0])
	amp0  = np.zeros([my_nsrc,ncomp])
	doff  = 0.20*m2r
	if args.freeze_beam != 0:
		beam0 = np.array([args.freeze_beam*b2r,args.freeze_beam*b2r,0.0])
		dbeam = np.array([0.00*b2r,0.00*b2r,00*d2r])
	else:
		dbeam = np.array([0.20*b2r,0.20*b2r,40*d2r])
	damp  = amp0 + 1500
	# And sample from this data set
	sampler = SrcSampler(d, my_pos, off0, beam0, amp0, doff, dbeam, damp)
	offsets = np.zeros([args.nsamp, 2])
	beams   = np.zeros([args.nsamp, 3])
	amps    = np.zeros([args.nsamp,my_nsrc,ncomp])
	for i in range(-args.burnin, args.nsamp):
		for j in range(args.thin):
			offset, beam, amp = sampler.draw(verbose=verbosity>1, fullmetro=False)
		if i >= 0: 
			offsets[i] = offset
			beams[i]   = beam
			amps[i]    = amp
			if i % args.dump == 0:
				params = sampler.get_params(offset, beam, amp2pflux(amp,beam))
				model  = sampler.get_model(params)
				mmaps, mrhs, mdiv = make_maps(model, d, my_pos, ncomp, args.radius*m2r, args.resolution*m2r)
				with h5py.File(tdir + "/model%04d.hdf" % i, "w") as hfile:
					hfile["data"] = mmaps
					hfile["rhs"]  = mrhs
					hfile["div"]  = mdiv
	# And save the results
	with h5py.File(tdir + "/params.hdf", "w") as hfile:
		hfile["offsets"] = offsets
		hfile["beams"] = beams
		hfile["amps"] = amps
		hfile["srcs"] = hit_srcs
