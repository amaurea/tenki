"""Unlike srclik_join, this program keeps the global parameters fixed,
and samples each TOD independently. The parameters are offset[{dec,ra}],
beam[3], amps[nsrc]. The amplitudes have a gaussian conditional distribution,
and can be samples directly. So we will use Gibbs sampling. That should make
this very fast.

Most of the point sources are very faint, with S/N < 0.1. We still sample them
individually per tod. The result will be a large number of noise-dominated samples.
These can then later be binned across tods in a different program to look for
source variability and gain variability."""

import numpy as np, os, mpi4py.MPI, time, h5py, sys, warnings, psutil, bunch
from scipy import ndimage
from scipy.ndimage import filters
from enlib import utils, config, ptsrc_data, log, bench, cg, array_ops, enmap
from enlib.degrees_of_freedom import DOF, Arg

#pids = [os.getpid()]
#while pids[-1] != 1:
#	pids.append(psutil.Process(pids[-1]).ppid())
#print "pidmap: %d -> %s" % (mpi4py.MPI.COMM_WORLD.rank, str(pids))

warnings.filterwarnings("ignore")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("filelist")
parser.add_argument("srcs")
parser.add_argument("odir")
parser.add_argument("--ncomp", type=int, default=1)
parser.add_argument("--nsamp", type=int, default=100)
parser.add_argument("--burnin",type=int, default=100)
parser.add_argument("--thin", type=int, default=5)
parser.add_argument("--minrange", type=int, default=0x100)
parser.add_argument("-R", "--radius", type=float, default=5.0)
parser.add_argument("-r", "--resolution", type=float, default=0.25)
parser.add_argument("-d", "--dump", type=int, default=100)
parser.add_argument("-c", action="store_true")
parser.add_argument("--freeze-beam", type=float, default=0)
parser.add_argument("--use-posmap", action="store_true")
parser.add_argument("--ml-start", action="store_true")
parser.add_argument("--nchain", type=int, default=3)
parser.add_argument("--nbasis", type=int, default=-50)
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("-b", "--brute", action="store_true")
args = parser.parse_args()

if args.seed > 0: np.random.seed(args.seed)

verbosity = config.get("verbosity")

comm  = mpi4py.MPI.COMM_WORLD
ncomp = args.ncomp
dtype = np.float64
d2r   = np.pi/180
m2r   = np.pi/180/60
b2r   = np.pi/180/60/(8*np.log(2))**0.5

# prior on beam
beam_min = 0.7*b2r
beam_max = 2.0*b2r
beam_ratio_max = 3.0
# prior on position
pos_deviation_max = 3*m2r

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
if args.freeze_beam: bfid = args.freeze_beam * b2r

# BEAMS
# We will represent the beam as sigma[2], phi
def amp2pflux(amp, beam):
	return amp.copy()
	#return amp*np.product(beam[:2])**0.5
def pflux2amp(pflux, beam):
	return pflux.copy()
	#return pflux/np.product(beam[:2])**0.5

def calc_fid_model(d, srcs):
	model = d.tod.astype(dtype)
	params = np.zeros([len(srcs),2+3+ncomp],dtype=dtype)
	bfid = srcs[0,19]*b2r
	params[:,:2] = srcs[:,[3,5]]*d2r
	params[:,2:-3] = srcs[:,7:7+2*ncomp:2]
	params[:,-3:] = utils.compress_beam(np.array([bfid,bfid]),0.0)[None,:]
	ptsrc_data.pmat_model(model, params, d)
	return model

def estimate_SN(d, srcs):
	# Estimate signal-to-noise in this TOD by generating a fiducial model
	model = calc_fid_model(d, srcs)
	nmodel = model.copy()
	ptsrc_data.nmat_basis(nmodel, d)
	return np.sum(model*nmodel)

class Adist:
	def __init__(self, A, b, dof):
		self.A = A
		self.b = b
		self.dof = dof
		self.Aih = array_ops.eigpow(A, -0.5)
		self.Ah  = array_ops.eigpow(A,  0.5)
		self.Ai  = array_ops.eigpow(A, -1.0)
		_,self.ldet = np.linalg.slogdet(self.Ai)
		self.x = self.Ai.dot(b)
	@property
	def a_ml(self): return self.dof.unzip(self.x)[0]
	def lik(self, amps):
		y = self.dof.zip(amps)
		d = y-self.x
		return 0.5*np.sum(d*self.A.dot(d)) + 0.5*self.ldet + 0.5*self.dof.n*np.log(2*np.pi)
	def draw_r(self): return np.random.standard_normal(self.dof.n)
	def r_to_a(self, r): return self.dof.unzip(self.x + self.Aih.dot(r))[0]
	def a_to_r(self, a): return self.Ah.dot(self.dof.zip(a)-self.x)
	def draw(self): return self.r_to_a(self.draw_r())

# Define our sampling scheme
class SrcSampler:
	def __init__(self, data, pos, off0, beam0, amp0, doff, dbeam, damp, use_posmap=False, ml_start=False):
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
		self.use_posmap = use_posmap
		self.L = np.inf
		self.adist = None
		if use_posmap:
			self.poslik, self.posbox = self.likmap_pos()
			self.Lbox_old = np.inf
		if ml_start:
			#print pflux2amp(self.pflux,self.beam)
			#self.off = np.array([0.0,-0.6])*m2r
			#pflux = self.draw_pflux(ml=True)
			#print pflux2amp(pflux,self.beam)
			#model = self.get_model(self.get_params(self.off, self.beam, pflux))
			#bin, _, _ = make_maps(model, data, pos, ncomp, args.radius*m2r, args.resolution*m2r)
			#with h5py.File("foomodel.hdf","w") as hfile:
			#	hfile["data"] = bin
			# Brute force grid search for ML point
			liks, boxes = self.likmap_pos(n=15, ml=True)
			#with h5py.File("foolik.hdf","w") as hfile:
			#	hfile["liks"] = liks
			#	hfile["boxes"] = boxes
			#	hfile["amps"] = amps
			#sys.exit(0)
			liks, boxes = liks.reshape(-1), boxes.reshape(-1,2,2)
			i = np.argmin(liks)
			self.off = np.mean(boxes[i],0)

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
	def prior(self, off, beam, pflux):
		sigma, phi = beam[:2],beam[2]
		if not np.all(np.isfinite(sigma)): return np.inf
		if np.max(sigma) > beam_max: return np.inf
		if np.min(sigma) < beam_min: return np.inf
		if np.max(sigma)/np.min(sigma) > beam_ratio_max: return np.inf
		if np.sum(off**2)**0.5 > pos_deviation_max: return np.inf
		return 0
	def lik(self, off, beam, pflux):
		prior  = self.prior(off, beam, pflux)
		if not np.isfinite(prior): return prior
		params = self.get_params(off, beam, pflux).astype(dtype)
		tod    = self.data.tod.astype(dtype)
		prob  = 0.5*np.sum(ptsrc_data.chisq_by_range(tod, params, self.data))
		return prob
	def likmap_pos(self, n=20, r=pos_deviation_max, ml=False):
		liks  = np.zeros([n,n])
		boxes = np.zeros([n,n,2,2])
		for yi, y in enumerate(np.linspace(-r,r,n,True)):
			for xi, x in enumerate(np.linspace(-r,r,n,True)):
				pos = np.array([y,x])
				boxes[yi,xi] = [[y-r/2,x-r/2],[y+r/2,x+r/2]]
				pflux = self.pflux
				if ml:
					offbak = self.off
					self.off = pos
					pflux = self.draw_pflux(ml=True)
					self.off = offbak
				liks[yi,xi]  = self.lik(pos, self.beam, pflux)
		return liks, boxes
	def draw_pflux(self, off=None, beam=None, ml=False):
		if beam is None: beam = self.beam
		adist = self.get_adist(off, beam)
		amps = adist.a_ml if ml else adist.draw()
		return amp2pflux(amps, beam)
	def get_adist(self, off=None, beam=None, dummy=False):
		# Maximum likelihood amplitude given by P'N"Pa = P'N"d.
		# But we want to draw a sample from the distribution.
		# Mean value given by the above. Variance given by (P'N"P)",
		# So to sample, must add (P'N"P)**0.5 r to rhs
		if off is None: off = self.off
		if beam is None: beam = self.beam
		params = self.get_params(off, beam, self.pflux).astype(dtype)
		if dummy:
			dof = DOF(Arg(default=params[:,2:-3]))
			return Adist(np.eye(dof.n), dof.zip(params[:,2:-3]), dof)
		# rhs = P'N"d
		tod    = self.data.tod.astype(dtype, copy=True)
		ptsrc_data.nmat_basis(tod, self.data)
		ptsrc_data.pmat_model(tod, params, self.data, dir=-1)
		rhs    = params[:,2:-3].copy()
		dof    = DOF(Arg(default=rhs))
		# Set up functional form of A
		def Afun(x):
			p = params.copy()
			p[:,2:-3], = dof.unzip(x)
			ptsrc_data.pmat_model(tod, p, self.data, dir=+1)
			ptsrc_data.nmat_basis(tod, self.data)
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
		return Adist(A, dof.zip(rhs), dof)
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
			print "%12.8f" % L
			sys.stdout.flush()
		return dof.unzip(x)
	def draw_posmap(self, nstep=1, verbose=False):
		L = self.lik(self.off, self.beam, self.pflux)
		def draw_posbox():
			lflat, bflat = self.poslik.reshape(-1), self.posbox.reshape(-1,2,2)
			nbox = lflat.size
			probs = np.exp(-0.5*(lflat-np.min(lflat)))
			probs /= np.sum(probs)
			pcum = np.cumsum(probs)
			i = np.searchsorted(pcum, np.random.uniform())
			return lflat[i], bflat[i]
		Lbox, box = draw_posbox()
		# Then draw the actual position using metropolis. Remember to
		# compensate for the asymmetric step!
		off = self.off.copy()
		for i in range(nstep):
			off_new = box[0] + (box[1]-box[0])*np.random.uniform(size=2)
			L_new = self.lik(off_new, self.beam, self.pflux)
			if np.random.uniform() < np.exp(L-L_new)*np.exp(self.Lbox_old-Lbox):
				off, L = off_new, L_new
		self.Lbox_old = Lbox
		return off
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
	def draw_pos_beam_r(self, nstep=1, verbose=False, boost=1, adjust=False):
		# Draw in parameters (theta=[pos,beam,...],r), where
		# a = a_ml + Aih r and r is normal distributed. Done
		# wia a two-step gibbs scheme, where
		#  P(theta|r,d) = L(theta,a=a_ml+Aih.dot(r)) (metropolis)
		#  P(r|theta,d) = N(0,1)
		off, beam, pflux = self.off.copy(), self.beam.copy(), self.pflux.copy()
		adist = self.adist if self.adist is not None else self.get_adist(off, beam)
		r = adist.a_to_r(pflux2amp(pflux,beam))
		L = self.L
		if self.metro_nsamp is None:
			self.metro_nsamp = 0
			self.metro_naccept = 0
		for i in range(nstep):
			# Draw new off and beam
			off_new  = off  + np.random.standard_normal(off.size)*self.doff*boost
			beam_new = beam + np.random.standard_normal(beam.size)*self.dbeam
			# Get amp distribution for the new position
			adist_new = self.get_adist(off_new, beam_new)
			a_new = adist_new.r_to_a(r)
			pflux_new = amp2pflux(a_new, beam_new)
			L_new = self.lik(off_new, beam_new, pflux_new)
			if np.random.uniform() < np.exp(L-L_new):
				off, beam, pflux, adist, L = off_new, beam_new, pflux_new, adist_new, L_new
				self.metro_naccept += 1
			self.metro_nsamp += 1
		# Then draw new r
		r = adist.draw_r()
		pflux = amp2pflux(adist.r_to_a(r),beam)
		L = self.lik(off,beam,pflux)
		if verbose:
			print "%5d %5.3f" % (self.step,float(self.metro_naccept)/self.metro_nsamp),
			print " %10.5f %10.5f %10.5f %10.5f %10.5f" % (tuple(off/m2r)+tuple(beam[:2]/b2r)+(beam[2]/d2r,)),
			print "%12.3f" % L
		self.off, self.beam, self.pflux, self.adist = off, beam, pflux, adist
		self.L  = L
		if adjust and self.metro_nsamp > 20:
			# Adjust step size based on accept rate so far
			goal  = 0.20
			ratio = float(self.metro_naccept)/self.metro_nsamp/goal
			factor= ratio
			print "Adjusted steps by", factor
			self.doff *= factor
			self.dbeam *= factor
			self.metro_naccept, self.metro_nsamp = 0,0
		return self.off, self.beam, self.pflux
	def draw_metro_joint(self, nstep=1, verbose=False):
		# Step simultaneously in off, beam and amps, but try to do so intelligently.
		# Step in off and beam as usual. Then draw amps from P(amps|off,beam).
		# So g(off2,beam2,amps2|off1,beam1,amps1) = h(off2,beam2|off1,beam1)*
		# P(amps2|off2,beam2). The overall acceptence rate will then be
		#  P(o2,b2,a2)*g(o1,b1,a1|o2,b2,a2)   P(o2,b2,a2)   P(a1|o1,b1)
		#  -------------------------------- = ----------- * -----------
		#  P(o1,b1,a1)*g(o2,b2,a2|o1,b1,a1)   P(o1,b1,a1)   P(a2|o2,b2)
		# The latter conditionals are gaussian and fast to evaluate
		# (we just need the cov and ml point, which are calculated anyway.
		# cache them from the last step, and no extra evaluation is needed).
		# With this scheme, no special pflux stuff is needed either. Let's
		# hope it works.
		off, beam, pflux = self.off.copy(), self.beam.copy(), self.pflux.copy()
		L = self.L
		adist = self.adist if self.adist is not None else self.get_adist(off, beam)
		a_lik = adist.lik(pflux2amp(pflux,beam))
		if self.metro_nsamp is None:
			self.metro_nsamp = 0
			self.metro_naccept = 0
		for i in range(nstep):
			# Draw new off and beam
			off_new  = off  + np.random.standard_normal(off.size)*self.doff
			beam_new = beam + np.random.standard_normal(beam.size)*self.dbeam
			# Get amp distribution for the new position
			adist_new = self.get_adist(off_new, beam_new)
			#a_new = adist_new.draw()
			a_new = adist_new.a_ml
			pflux_new = amp2pflux(a_new, beam_new)
			# Evaluate likelihood at new position
			L_new = self.lik(off_new, beam_new, pflux_new)
			a_lik_new = adist_new.lik(a_new)
			# And accept or reject
			print "%15.5f %15.5f %15.5f %15.5f %15.5f" % (L, L_new, a_lik, a_lik_new, L-L_new+a_lik_new-a_lik)
			if np.random.uniform() < np.exp(L-L_new)*np.exp(a_lik_new-a_lik):
				off, beam, pflux = off_new, beam_new, pflux_new
				L, a_lik = L_new, a_lik_new
				adist = adist_new
				self.metro_naccept += 1
			self.metro_nsamp += 1
		if verbose:
			print "%5d %5.3f" % (self.step,float(self.metro_naccept)/self.metro_nsamp),
			print " %10.5f %10.5f %10.5f %10.5f %10.5f" % (tuple(off/m2r)+tuple(beam/b2r)),
			print "%12.3f" % L
		self.off, self.beam, self.pflux = off, beam, pflux
		self.L, self.adist = L, adist
		return self.off, self.beam, self.pflux

	def draw(self, verbose=False, mode="joint", adjust=False):
		if mode == "joint":
			with bench.mark("draw_metro_joint"):
				self.off, self.beam, self.pflux = self.draw_metro_joint(verbose=verbose)
		elif mode == "brute":
			with bench.mark("draw_metro_full"):
				self.off, self.beam, self.pflux = self.draw_metro_all(verbose=verbose)
		elif mode == "hybrid":
			with bench.mark("draw_pos_beam"):
				self.off, self.beam = self.draw_pos_beam(verbose=verbose)
			with bench.mark("draw_amps"):
				self.pflux = self.draw_pflux()
		elif mode == "gibbs":
			with bench.mark("gibbs"):
				boost = 1+10*(self.step%5==0)
				self.off, self.beam, self.pflux = self.draw_pos_beam_r(verbose=verbose, boost=boost, adjust=adjust)
		else:
			raise ValueError("Unrecognized mode '%s'" % mode)
		sigma, phi = self.beam[:2], self.beam[2]
		self.step += 1
		return self.off, np.concatenate([sigma,[phi]]), pflux2amp(self.pflux,self.beam), self.lik(self.off, self.beam, self.pflux)
	def brute_force_grid(self, shape, box, levels=3, nsub=(2,2), threshold=1e-4, verbose=False):
		# P(p) = sum(P(p,a),a) = q sum(norm(a,ca),a)
		# P(p,a0) = q norm(a0,ca) = q/normint(ca) => q = normint(ca)*P(p,a0)
		# P(p) = P(p,a0) * normint(ca) * sum(exp(-(a-a0)ca"(a-a0)),a)/normint(ca)
		#      = P(p,a0) * normint(ca)
		# sum(P(p)) = 1
		#
		# So evaluate p in grid. For each position, build amplitude distribution.
		# Store P(p)[ny,nx], a0(p)[ny,nx,namp] and ca(p)[ny,nx,namp,namp]
		# Build initial cells
		ashape = self.pflux.shape
		def build_cell(box):
			pmid  = np.mean(box,0)
			parea = np.product(box[1]-box[0])
			adist = self.get_adist(pmid, self.beam, dummy=False)
			a0 = adist.a_ml
			ca = adist.Ai.reshape(ashape+ashape)
			# Given the a distribution, the posterior is given by
			# logP = 0.5*log(|A|) + 0.5*a_ml'cov(a)"a_ml + const
			logPa0 = 0.5*np.sum(adist.x*adist.A.dot(adist.x))
			logArea= 0.5*np.linalg.slogdet(adist.Ai)[1]
			# logArea creates a quite strong bias towards poorly determined
			# regions. Leaving it out is equivalent to imposing a prior that
			# P(offset) should be uniform in the absence of data. I think this
			# is well-defined, but it will need to be discussed in a paper.
			logP   = logPa0 - self.prior(pmid, self.beam, amp2pflux(a0,self.beam))

			#logPa0 = -self.lik(pmid, self.beam, amp2pflux(a0, self.beam))
			#logArea= -0.5*np.linalg.slogdet(2*np.pi*adist.Ai)[1]
			#logP = logPa0 + logArea

			#logPa0_dummy = -self.lik(pmid, self.beam, amp2pflux(adist_dummy.a_ml, self.beam))
			if verbose:
				print "%12.4f %12.4f %15.4f %15.4f %15.4f %15.2f" % (tuple(pmid/m2r)+(logP,logPa0,logArea,a0[0,0]))
			return bunch.Bunch(box=box, a0=a0, ca=ca, logP=logP, logArea=logArea, logPa0=logPa0, area=parea)
		def norm_cells(cells):
			logP = np.array([c.logP for c in cells])
			area = np.array([c.area for c in cells])
			P = np.exp(logP-np.max(logP))*area
			P /= np.sum(P)
			for ci, c in enumerate(cells): c.P = P[ci]
		# Build the initial level of cells
		def build_cell_grid(shape, box):
			cells = []
			b0 = box[0]; db = (box[1]-box[0])/np.array(shape)
			for iy in range(shape[0]):
				for ix in range(shape[1]):
					cbox = np.array([b0+db*[iy,ix],b0+db*[iy+1,ix+1]])
					cells.append(build_cell(cbox))
			norm_cells(cells)
			return cells
		def refine_cells(cells, nsub, threshold, nmax=1000):
			if len(cells) > nmax: return cells
			res = []
			# These are interesting in themselves
			selected = [cell.P > threshold for cell in cells]
			# Also get their neighbors (O(N^2))
			def isneigh(box1,box2,tol=0.1):
				dists = np.abs(box1[:,0]-box2[::-1,0])
				sizes = np.maximum(np.abs(box1[1]-box1[0]),np.abs(box2[1]-box2[0]))
				return np.all(dists < sizes*tol)
			neighsel = [any([isneigh(cell.box,other.box) for oi,other in enumerate(cells) if selected[oi]]) for cell in cells]
			for i in range(len(selected)): selected[i] = selected[i] or neighsel[i]
			for ci, cell in enumerate(cells):
				if selected[ci]:
					# Refine this cell
					res += build_cell_grid(nsub, cell.box)
				else:
					res.append(cell)
			norm_cells(res)
			return res
		def flatten_cells(cells):
			n = len(cells)
			boxes = np.array([c.box for c in cells])
			a0s = np.array([c.a0 for c in cells])
			cas = np.array([c.ca for c in cells])
			Ps = np.array([c.P for c in cells])
			logPs = np.array([c.logP for c in cells])
			logAreas = np.array([c.logArea for c in cells])
			logPa0s = np.array([c.logPa0 for c in cells])
			return bunch.Bunch(box=boxes,a0=a0s,ca=cas,P=Ps, logP=logPs, logArea=logAreas, logPa0=logPa0s)
		cells = build_cell_grid(shape, box)
		for level in range(levels):
			cells = refine_cells(cells, nsub, threshold)
		cells = flatten_cells(cells)
		#areas = np.product(cells.box[:,1]-cells.box[:,0],1)
		#density = cells.P/areas
		#density/=np.max(density)
		#for p, P, d in zip(np.mean(cells.box,1), cells.P, density):
		#	print "%8.4f %8.4f %8.6f %8.6f" % (tuple(p/m2r)+(P,d))
		return cells

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
	ptsrc_data.nmat_basis(tod, data)
	ptsrc_data.pmat_thumbs(-1, tod, rhs, data.point, data.phase, boxes)
	# Build div
	for c in range(ncomp):
		idiv = div[0].copy(); idiv[:,c] = 1
		wtod = data.tod.astype(dtype,copy=True); wtod[...] = 0
		ptsrc_data.pmat_thumbs( 1, wtod, idiv, data.point, data.phase, boxes)
		ptsrc_data.nmat_basis(wtod, data, white=True)
		ptsrc_data.pmat_thumbs(-1, wtod, div[c], data.point, data.phase, boxes)
	div = np.rollaxis(div,1)
	bin = rhs/div[:,0] # Fixme: only works for ncomp == 1
	return bin, rhs, div
def stack_maps(rhs, div, amps=None):
	if amps is None: amps = np.zeros(len(rhs))+1
	ar = (rhs.T*amps.T).T; ad = (div.T*amps.T**2).T
	sr = np.sum(ar,0)
	sd = np.sum(ad,0)
	return sr/sd[0]*amps[0]

def chisqmap(tod, data, pos, ncomp, radius, resolution):
	ntod = tod.copy()
	# Compute chisq per sample
	ptsrc_data.nmat_basis(ntod, data)
	tchisq = ntod*tod
	# Set up pixels
	n   = int(np.round(2*radius/resolution))
	boxes = np.array([[p-radius,p+radius] for p in pos])
	# Set up output maps
	chisqs = np.zeros([nsrc,ncomp,n,n],dtype=dtype)
	# project
	ptsrc_data.pmat_thumbs(-1, tchisq, chisqs, data.point, data.phase, boxes)
	return chisqs

def dump_maps(ofile, tod):
	dmaps, drhs, ddiv = make_maps(tod.astype(dtype), d, my_pos, ncomp, args.radius*m2r, args.resolution*m2r)
	dstack = stack_maps(drhs, ddiv, my_srcs[:,7])
	with h5py.File(ofile, "w") as hfile:
		hfile["data"] = dmaps
		hfile["rhs"]  = drhs
		hfile["div"]  = ddiv
		hfile["stack"] = dstack

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
	if args.c and os.path.isfile(tdir + "/params.hdf"):
		continue

	d   = ptsrc_data.read_srcscan(fname)
	d.Q = ptsrc_data.build_noise_basis(d,args.nbasis)

	# Override noise model - the one from the files
	# doesn't seem to be reliable enough.
	vars, nvars = ptsrc_data.measure_basis(d.tod, d)
	ivars = np.sum(nvars,0)/np.sum(vars,0)
	d.ivars = ivars

	# Discard sources that aren't sufficiently hit
	srcmask = d.offsets[:,-1]-d.offsets[:,0] > args.minrange
	if np.sum(srcmask) == 0:
		L.debug("Too few sources in %s, skipping" % id)
		continue

	# We don't like detectors where the noise properties vary too much.
	detmask = np.zeros(len(ivars),dtype=bool)
	for di, (dvar,ndvar) in enumerate(zip(vars.T,nvars.T)):
		dhit = ndvar > 20
		if np.sum(dhit) == 0:
			# oops, rejected everything
			detmask[di] = False
			continue
		dvar, ndvar = dvar[dhit], ndvar[dhit]
		mean_variance = np.sum(dvar)/np.sum(ndvar)
		individual_variances = dvar/ndvar
		# It is dangerous if the actual variance in a segment of the tod
		# is much higher than what we think it is.
		detmask[di] = np.max(individual_variances/mean_variance) < 3
	hit_srcs = np.where(srcmask)[0]
	hit_dets = np.where(detmask)[0]

	print "FIXME"
	hit_srcs = hit_srcs[:1]

	if len(hit_srcs) == 0 or len(hit_dets) == 0:
		L.debug("All data rejected in %s, skipping" % id)
		continue

	print "FIXME"
	d.ivars[...]=1

	d = d[hit_srcs,hit_dets]
	d.Q = ptsrc_data.build_noise_basis(d,args.nbasis)
	my_nsrc, my_ndet = len(hit_srcs), len(hit_dets)
	my_pos = pos[hit_srcs]
	my_srcs= srcs[hit_srcs]

	SN = estimate_SN(d, my_srcs)
	print "SN**0.5", SN**0.5

	# Replace data with simulation
	print "FIXME"
	amp0  = my_srcs[:,range(7,7+2*ncomp,2)]
	off0  = np.array([0,0])
	beam0 = np.array([bfid,bfid,0])
	sampler = SrcSampler(d, my_pos, off0, beam0, amp0, off0*0, beam0*0, amp0*0)
	model = sampler.get_model(sampler.get_params(sampler.off, sampler.beam, sampler.pflux))
	noisesim = np.random.standard_normal(model.shape).astype(model.dtype)
	print d.offsets.shape
	ri2d = {ri:di for so in d.offsets for di in range(so.size-1) for ri in d.rangesets[so[di]:so[di+1]]}
	for ri, r in enumerate(d.ranges):
		noisesim[r[0]:r[1]] *= d.ivars[ri2d[ri]]**-0.5
	d.tod = noisesim#+model

	# Make minimaps of our data, for visual inspection
	dump_maps(tdir + "/data.hdf", d.tod)
	# Make map for fiducial model too
	model = calc_fid_model(d, my_srcs)
	dump_maps(tdir + "/model_fid.hdf", model)
	# And a full simulation
	noisesim = np.random.standard_normal(model.shape).astype(model.dtype)
	ri2d = {ri:di for so in d.offsets for di in range(so.size-1) for ri in d.rangesets[so[di]:so[di+1]]}
	for ri, r in enumerate(d.ranges):
		noisesim[r[0]:r[1]] *= d.ivars[ri2d[ri]]**-0.5
	dump_maps(tdir + "/noise.hdf", noisesim)
	## Output filtered data vs. sim
	#ftod = d.tod.astype(dtype); ptsrc_data.nmat_basis(ftod, d)
	#fmodel = model.copy(); ptsrc_data.nmat_basis(fmodel, d)
	#with h5py.File(tdir + "/tods.hdf", "w") as hfile:
	#	hfile["data"] = ftod
	#	hfile["model"] = fmodel

	if args.brute:
		# In this mode we don't output samples. Instead we output an explicit likelihood
		# model
		amp0  = my_srcs[:,range(7,7+2*ncomp,2)]
		off0  = np.array([0,0])
		beam0 = np.array([bfid,bfid,0])
		sampler = SrcSampler(d, my_pos, off0, beam0, amp0, off0*0, beam0*0, amp0*0)
		cells = sampler.brute_force_grid((10,10),np.array([[-1,-1],[1,1]])*pos_deviation_max, verbose=verbosity>1,levels=0)
		print cells.P/np.max(cells.P)
		with h5py.File(tdir + "/posterior.hdf", "w") as hfile:
			hfile["off"]  = cells.box
			hfile["a0"]   = cells.a0
			hfile["ca"]   = cells.ca
			hfile["P"]    = cells.P
			hfile["logP"] = cells.logP
			hfile["logPa0"] = cells.logPa0
			hfile["logArea"] = cells.logArea
			hfile["srcs"] = hit_srcs
			hfile["SN"]   = SN
			hfile["beam"] = sampler.beam
		# Find maxima
		print "maxlik", np.max(cells.logPa0)
		fullbox = np.array([np.min(cells.box[:,0],0),np.max(cells.box[:,1],0)])
		shape   = (500,500)
		wcs     = enmap.create_wcs(shape, fullbox, "car")
		pmap    = enmap.zeros(shape, wcs)
		imap    = enmap.zeros(shape, wcs, dtype=int)
		for i,(b,logp) in enumerate(zip(cells.box,cells.logP)):
			pmap.submap(b,inclusive="True")[...] = logp-np.max(cells.logP)
			imap.submap(b,inclusive="True")[...] = i
		maxvals = filters.maximum_filter(pmap, 50)
		maxmask = (pmap == maxvals)*(maxvals > -2)
		maxinds = imap[np.where(maxmask)]
		maxvals = pmap[np.where(maxmask)]
		# Sort by value
		sortinds = np.argsort(maxvals)
		maxinds = np.unique(maxinds[sortinds])
		print maxinds

		# Dump chisqmaps
		chisqs = chisqmap(d.tod.astype(dtype), d, my_pos, ncomp, args.radius*m2r, args.resolution*m2r)
		with h5py.File(tdir + "/data_chisq.hdf", "w") as hfile:
			hfile["data"] = chisqs
			hfile["stack"] = np.sum(chisqs,0)

		# And plot them
		for i, maxind in enumerate(maxinds):
			model = sampler.get_model(sampler.get_params(np.mean(cells.box[maxind],0), sampler.beam, amp2pflux(cells.a0[maxind],sampler.beam)))
			dump_maps(tdir + "/maxlik%d.hdf" % i, model)
			dump_maps(tdir + "/residual%d.hdf" % i, d.tod-model)
			chisqs = chisqmap((d.tod-model).astype(dtype), d, my_pos, ncomp, args.radius*m2r, args.resolution*m2r)
			with h5py.File(tdir + "/maxlik%d_chisq.hdf" % i, "w") as hfile:
				hfile["data"] = chisqs
				hfile["stack"] = np.sum(chisqs,0)
		print "SN**0.5", SN**0.5

	else:
		offsets = np.zeros([args.nchain,args.nsamp, 2])
		beams   = np.zeros([args.nchain,args.nsamp, 3])
		amps    = np.zeros([args.nchain,args.nsamp,my_nsrc,ncomp])
		liks    = np.zeros([args.nchain,args.nsamp])
		for chain in range(args.nchain):
			# Start each chain with a random position to probe initial
			# condition dependence.
			off0  = np.random.standard_normal(2)*pos_deviation_max/4
			amp0  = my_srcs[:,range(7,7+2*ncomp,2)]
			doff  = 0.10*m2r
			if args.freeze_beam != 0:
				beam0 = np.array([args.freeze_beam*b2r,args.freeze_beam*b2r,0.0])
				dbeam = np.array([0.00*b2r,0.00*b2r,00*d2r])
			else:
				beam0 = np.array([bfid,bfid,0.0])
				dbeam = np.array([0.10*b2r,0.10*b2r,10*d2r])
			damp  = amp0*0 + 1500
			# And sample from this data set
			sampler = SrcSampler(d, my_pos, off0, beam0, amp0, doff, dbeam, damp, use_posmap=args.use_posmap, ml_start=args.ml_start)

			#liks, boxes = sampler.likmap_pos(n=40, ml=True)
			#with h5py.File("foolik.hdf","w") as hfile:
			#	hfile["liks"] = liks
			#	hfile["boxes"] = boxes
			#sys.exit(0)

			for i in range(-args.burnin, args.nsamp):
				for j in range(args.thin):
					if i == 0 and j == 0: adjust = True
					elif i < 0: adjust = (-i)%20 == 0 and j == 0
					else: adjust = False
					offset, beam, amp, lik = sampler.draw(verbose=verbosity>1, mode="gibbs", adjust=adjust)
				if i >= 0: 
					offsets[chain,i] = offset
					beams[chain,i]   = beam
					amps[chain,i]    = amp
					liks[chain,i]    = lik
					if i % args.dump == 0:
						params = sampler.get_params(offset, beam, amp2pflux(amp,beam))
						model  = sampler.get_model(params)
						#mmaps, mrhs, mdiv = make_maps(model, d, my_pos, ncomp, args.radius*m2r, args.resolution*m2r)
						#with h5py.File(tdir + "/model%d_%04d.hdf" % (chain,i), "w") as hfile:
						#	hfile["data"] = mmaps
						#	hfile["rhs"]  = mrhs
						#	hfile["div"]  = mdiv
		# And save the results
		with h5py.File(tdir + "/params.hdf", "w") as hfile:
			hfile["offsets"] = offsets
			hfile["beams"] = beams
			hfile["amps"] = amps
			hfile["srcs"] = hit_srcs
			hfile["liks"] = liks
			hfile["SN"] = SN
