"""The old srclik_tod sampled offset[2], beam[3], amps[nsrc,ncomp]. Sampling all this led to
over-fitting, where offset would be unrealistically certain despite all the amplitudes being
individually very uncertain. To get around this problem, I here split the amplitudes into
two sets: strong and weak. The strong ones are those that can be detected individually, and so
constrain more than they overfit. The weak ones are the rest. I then sample as before over
offset, beam and strong with the weak fixed at fiducial values, and then sample
strong and weak based on offset and beam."""

import numpy as np, argparse, warnings, mpi4py.MPI, copy, h5py, os, bunch
from enlib import utils, ptsrc_data, log, bench, cg, array_ops, enmap, errors, pointsrcs
from enlib.degrees_of_freedom import DOF, Arg
from scipy.special import erf
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("filelist")
parser.add_argument("srcs")
parser.add_argument("beam")
parser.add_argument("odir")
parser.add_argument("-v", "--verbosity", type=int, default=1)
parser.add_argument("--ncomp", type=int, default=1)
parser.add_argument("--nsamp", type=int, default=500)
parser.add_argument("--burnin",type=int, default=100)
parser.add_argument("--thin", type=int, default=3)
parser.add_argument("--nbasis", type=int, default=-30)
parser.add_argument("--nadapt", type=int, default=10)
parser.add_argument("--strong", type=float, default=3.0)
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("-c", action="store_true")
parser.add_argument("-g", "--grid", type=int, default=0)
parser.add_argument("--minrange", type=int, default=0x40)
parser.add_argument("--allow-clusters", action="store_true")
parser.add_argument("--sample-beam", action="store_true")
parser.add_argument("--no-adaptive", action="store_true")
#parser.add_argument("--nchain", type=int, default=3)
parser.add_argument("-R", "--radius", type=float, default=5.0)
parser.add_argument("-r", "--resolution", type=float, default=0.25)
parser.add_argument("-m", "--map", action="store_true")
parser.add_argument("-F", "--fitmap", action="store_true")
parser.add_argument("--oldformat", action="store_true")
parser.add_argument("--fitmaps", action="store_true")
args = parser.parse_args()

if args.seed: np.random.seed(args.seed)

comm  = mpi4py.MPI.COMM_WORLD
ncomp = args.ncomp
dtype = np.float64
d2r   = np.pi/180
m2r   = np.pi/180/60
b2r   = np.pi/180/60/(8*np.log(2))**0.5

try:
	# Beam in format [r,val], where r is equispaced starting at 0, in units of arcmin
	# and val has a max value of 1
	b = np.loadtxt(args.beam)
	beam = bunch.Bunch(profile=b[:,1], rmax=b[1,0]*len(b)*utils.arcmin)
except IOError:
	# Assume beam is gaussian
	b = float(args.beam)*utils.arcmin*utils.fwhm
	r    = np.linspace(0,10,1000)*b
	beam = bunch.Bunch(profile=np.exp(-0.5*(r/b)**2), rmax=10*b)
	beam_global = b
if not args.oldformat:
	beam_global = 1.0
	print "Using old model"
else:
	print "Using new model"
# prior on beam deformations
beam_rel_min = 0.5
beam_rel_max = 2.0
beam_ratio_max = 3.0
# prior on position
pos_rel_max = 5*m2r

log_level = log.verbosity2level(args.verbosity)
L = log.init(level=log_level, rank=comm.rank, shared=False)
bench.stats.info = [("time","%6.2f","%6.3f",1e-3),("cpu","%6.2f","%6.3f",1e-3),("mem","%6.2f","%6.2f",2.0**30),("leak","%6.2f","%6.3f",2.0**30)]

filelist = utils.read_lines(args.filelist)

srcs = pointsrcs.read(args.srcs)
posi, ampi, beami = [0,1], [2,3,4], [5,6,7]
nsrc = len(srcs)

utils.mkdir(args.odir)

if args.oldformat:
	def apply_model(tod, pflat, d, dir=1):
		ptsrc_data.pmat_model(tod, pflat, d, dir=dir)
else:
	def apply_model(tod, pflat, d, dir=1):
		ptsrc_data.pmat_beam_foff(tod, pflat, beam, d, dir=dir)

class Parameters:
	def __init__(self, pos_fid, beam_fid, amp_fid, pos_rel=None, beam_rel=None, amp_rel=None, strong=None):
		self.pos_fid   = np.array(pos_fid,dtype=float)
		self.beam_fid  = np.array(beam_fid,dtype=float)
		self.amp_fid   = np.array(amp_fid,dtype=float)
		self.pos_rel   = np.array([0,0] if pos_rel is None else pos_rel,dtype=float)
		self.beam_rel  = np.array([1,1,0] if beam_rel is None else beam_rel,dtype=float)
		self.amp_rel   = np.zeros(self.amp_fid.shape,dtype=float) if amp_rel is None else np.array(amp_rel,dtype=float)
		self.groups    = [range(self.amp_fid.shape[0])]
		self.strong    = np.full(self.amp_fid.shape, True, dtype=bool) if strong is None else np.array(strong,dtype=bool)
		self.nsrc, self.ncomp = self.amp_fid.shape
	@property
	def flat(self):
		if args.oldformat:
			params = np.zeros([self.nsrc,8],dtype=dtype)
			params[:,:2] = self.pos_fid + self.pos_rel
		else:
			params = np.zeros([self.nsrc,10],dtype=dtype)
			params[:,:2] = self.pos_fid
			params[:,8:] = self.pos_rel
		params[:,2:2+self.ncomp] = self.amp_fid + self.amp_rel
		for i in range(self.nsrc):
			bf = utils.compress_beam(self.beam_fid[i,:2],self.beam_fid[i,2])
			br = utils.compress_beam(self.beam_rel[:2], self.beam_rel[2])
			params[:,5:8] = utils.combine_beams([bf,br])
		return params
	@property
	def area(self):
		"""Compute the beam area for each source. This will need to change if we change
		the beam model from the naive stretch model used here."""
		return np.product(self.beam_fid[:,:2],1)*np.product(self.beam_rel[:2])
	def copy(self): return copy.deepcopy(self)

class AmpDist:
	def __init__(self, icov, rhs, dof):
		self.Ai = icov
		self.b   = rhs
		self.dof = dof
		if icov.size > 0:
			self.A   = array_ops.eigpow(icov, -1.0)
			self.Aih = array_ops.eigpow(icov,  0.5)
			self.Ah  = array_ops.eigpow(icov, -0.5)
			_,self.ldet = np.linalg.slogdet(self.Ai)
			self.x = self.A.dot(rhs)
		else:
			self.A, self.Aih, self.Ah = [icov.copy()]*3
			self.x, self.ldet = rhs.copy(), 0
	@property
	def a(self): return self.dof.unzip(self.x)[0]
	def draw_r(self): return np.random.standard_normal(self.dof.n)
	def r_to_a(self, r): return self.dof.unzip(self.x + self.Ah.dot(r))[0]
	def a_to_r(self, a): return self.Aih.dot(self.dof.zip(a)-self.x)
	def draw(self): return self.r_to_a(self.draw_r())

def validate_srcscan(d, srcs):
	"""Generate a noise model for d and reduce it to a relevant set of sources and detectors.
	Returns the new d and source indices."""
	if not args.no_adaptive:
		d.Q = ptsrc_data.build_noise_basis_adaptive(d,nmax=args.nadapt)
	else:
		d.Q = ptsrc_data.build_noise_basis(d,args.nbasis)
	# Override noise model - the one from the files
	# doesn't seem to be reliable enough.
	vars, nvars = ptsrc_data.measure_basis(d.tod, d)
	ivars = np.sum(nvars,0)/np.sum(vars,0)
	d.ivars = ivars
	# Discard sources that aren't sufficiently hit. This is
	# a limit on the number of ranges for each source, not the
	# number of samples!
	srcmask = np.sum(d.offsets[:,:,1]-d.offsets[:,:,0],1) > args.minrange
	# Discard clusters, as they aren't point-like
	if not args.allow_clusters:
		srcmask *= srcs[:,ampi[0]] > 0
	if np.sum(srcmask) == 0:
		raise errors.DataError("Too few sources")
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
	# Reduce to relevant sources and dets, and update noise model
	d = d[hit_srcs,hit_dets]
	if not args.no_adaptive:
		d.Q = ptsrc_data.build_noise_basis_adaptive(d,nmax=args.nadapt)
	else:
		d.Q = ptsrc_data.build_noise_basis(d,args.nbasis)
	d.tod = d.tod.astype(dtype)
	return d, hit_srcs

def independent_groups(d):
	# Two sources are independent if they don't share any ranges
	nsrc, ndet = d.shape
	groups = []
	done = np.zeros(nsrc,dtype=bool)
	for si in range(nsrc):
		if done[si]: continue
		group = [si]
		range_hit = np.zeros(d.ranges.shape[0],dtype=bool)
		# Mark all the ranges the current source hits
		for off in d.offsets[si]:
			range_hit[d.rangesets[off[0]:off[1]]] = True
		# For all other sources, check if we overlap or not
		for si2 in range(si+1,nsrc):
			if done[si2]: continue
			overlap = False
			for off in d.offsets[si2]:
				if np.any(range_hit[d.rangesets[off[0]:off[1]]]): overlap = True
			if overlap: group.append(si2)
		for s in group: done[s] = True
		groups.append(group)
	return groups

def groups_to_dof(groups, dof):
	"""Given a set of independet source groups and a DOF object, output
	independent groups in terms of those degrees of freedom."""
	dof2comp, = dof.unzip(np.arange(1,dof.n+1))
	ncomp = dof2comp.shape[1]
	comps = range(ncomp)
	res = [[dof2comp[e,c]-1 for e in g for c in comps if dof2comp[e,c]>0] for g in groups]
	return [e for e in res if len(e) > 0]

def estimate_SN(d, fparams, src_groups):
	gmax = max([len(g) for g in src_groups])
	nsrc, ncomp = fparams[:,2:-3].shape
	SN = np.zeros([nsrc,ncomp])
	for i in range(gmax):
		for c in range(ncomp):
			flat = fparams.copy()
			flat[:,2:-3] = 0
			for g in src_groups:
				if len(g) <= i: continue
				flat[g[i],c+2] = fparams[g[i],c+2]
			# Do all the compatible sources in parallel
			mtod = d.tod.copy()
			apply_model(mtod, flat, d)
			ntod = mtod.copy()
			ptsrc_data.nmat_basis(ntod, d)
			# And then extract S/N for each of them
			for g in src_groups:
				if len(g) <= i: continue
				si = g[i]
				my_sn = 0
				for off in d.offsets[si]:
					for ri in d.rangesets[off[0]:off[1]]:
						r = d.ranges[ri]
						my_sn += np.sum(ntod[r[0]:r[1]]*mtod[r[0]:r[1]])
				SN[si,c] = my_sn
	return SN

def calc_amp_dist(tod, d, params, mask=None):
	if mask is None: mask = params.strong
	if np.sum(mask) == 0: return AmpDist(np.zeros([0,0]), np.zeros([0]), DOF(Arg(mask=mask)))
	# rhs = P'N"d
	tod = tod.astype(dtype, copy=True)
	pflat = params.flat.copy()
	ptsrc_data.nmat_basis(tod, d)
	apply_model(tod, pflat, d, dir=-1)
	rhs    = pflat[:,2:-3].copy()
	dof    = DOF(Arg(mask=mask))
	# Set up functional form of icov
	def icov_fun(x):
		p = pflat.copy()
		p[:,2:-3], = dof.unzip(x)
		apply_model(tod, p, d, dir=+1)
		ptsrc_data.nmat_basis(tod, d)
		apply_model(tod, p, d, dir=-1)
		return dof.zip(p[:,2:-3])
	# Build A matrix in parallel. When using more than
	# one component, the ndof will be twice the number of sources, so
	# groups must be modified
	dgroups = groups_to_dof(params.groups, dof)
	icov = np.zeros([dof.n,dof.n])
	nmax = max([len(g) for g in dgroups])
	for i in range(nmax):
		# Loop through the elements of the uncorrelated groups in parallel
		u = np.zeros(dof.n)
		u[[g[i] for g in dgroups if len(g) > i]] = 1
		icov_u = icov_fun(u)
		# Extract result into full A
		for g in dgroups:
			if len(g) > i:
				icov[g[i],g] = icov_u[g]
	return AmpDist(icov, dof.zip(rhs), dof)

def subtract_model(tod, d, fparams):
	mtod = tod.astype(dtype,copy=True)
	p = fparams.copy()
	apply_model(mtod, p, d)
	return tod-mtod

def calc_posterior(tod, d, fparams):
	wtod = subtract_model(tod, d, fparams)
	ntod = wtod.copy()
	ptsrc_data.nmat_basis(ntod, d)
	return -0.5*np.sum(ntod*wtod)

def calc_marginal_amps_strong(d, p):
	# The marginal probability -2 log P(pos|beam,aw) = (d-Pw aw)'N"(d-Pw aw) - as' As" as
	# where as = (Ps'N"Ps)"Ps'N"(d-Pw aw). First compute as and As.
	p_weak = p.flat; p_weak[:,2:-3][p.strong] = 0
	tod_rest = subtract_model(d.tod, d, p_weak)
	# calc_amp_dist only uses strong dof by default
	adist_strong = calc_amp_dist(tod_rest, d, p)
	x_s, Ai_s = adist_strong.x, adist_strong.Ai
	P_s = 0.5*np.sum(x_s*Ai_s.dot(x_s))
	# The remainder is tod_rest'N"tod_rest
	ntod_rest = tod_rest.copy()
	ptsrc_data.nmat_basis(ntod_rest, d)
	P_w = -0.5*np.sum(tod_rest*ntod_rest)
	return P_s, P_w, adist_strong

def grid_pos(d, params, box=np.array([[-1,-1],[1,1]])*pos_rel_max, shape=(10,10)):
	# Build the pos grid
	p = params.copy()
	shape, wcs = enmap.geometry(pos=box, shape=shape, proj="car")
	probs = enmap.zeros(shape, wcs)
	pos_rel = probs.posmap()
	best = -np.inf
	for iy in range(shape[0]):
		for ix in range(shape[1]):
			p.pos_rel = pos_rel[:,iy,ix]
			P_s, P_w, adist_strong = calc_marginal_amps_strong(d, p)
			probs[iy,ix] = P_s + P_w
			best = max(best,probs[iy,ix])
			print "%4d %4d %6.2f %6.2f %9.3f %9.3f %9.3f %s" % ((iy,ix)+tuple(pos_rel[:,iy,ix]/m2r)+(probs[iy,ix],P_s,P_w) + ("*" if probs[iy,ix]>=best else "",))
	return probs

class HybridSampler:
	def __init__(self, d, params, dpos, dbeam, nstep=1, verbose=False, prior=lambda p: 0, dist=np.random.standard_normal):
		self.d = d
		self.p = params.copy()
		self.dpos = dpos
		self.dbeam = dbeam
		self.nstep = nstep
		self.logP = -np.inf
		self.adist_strong = None
		self.verbose = verbose
		self.ntry = 0
		self.naccept = 0
		self.prior = prior
		self.dist = dist
	def draw_pos_beam(self):
		"""Draw a sample of the relative offsets, beam and strong amplitude
		distribution given the data and weak amplitudes, which are kept constant. Returns
		an updated params object as well as an AmpDist object."""
		p_new = self.p.copy()
		first_step = np.isinf(self.logP)
		if not first_step:
			p_new.pos_rel  += self.dist(len(self.dpos))*self.dpos
			p_new.beam_rel += self.dist(len(self.dbeam))*self.dbeam
		# Adjust amps to preserve flux
		area_ratio = p_new.area/self.p.area
		p_new.amp_rel = (p_new.amp_fid+p_new.amp_rel)/area_ratio[:,None] - p_new.amp_fid
		P_s, P_w, adist_strong = calc_marginal_amps_strong(self.d, p_new)
		logP_new = self.prior(p_new, adist_strong)
		logP_new += P_s + P_w
		if np.random.uniform() < np.exp(logP_new-self.logP):
			self.p = p_new
			self.logP = logP_new
			self.adist_strong = adist_strong
			self.naccept += 1
		self.ntry += 1
		if self.verbose:
			print "%6d %5.3f %8.3f %8.3f %8.4f %8.4f %8.2f" % ((self.ntry, float(self.naccept)/self.ntry) + 
					tuple(self.p.pos_rel/m2r) + tuple(self.p.beam_rel[:2]) + (self.p.beam_rel[2]/d2r,))
		return self.p, self.adist_strong
	def draw_amps(self):
		"""Draw a sample from the distribution of strong and weak amplitudes
		given the position and beam. The internal amplitudes are not updated.
		The strong ones would be ignored in draw_pos_beam anyway, since it
		marginalizes over them. The weak ones are kept fixed internally to
		decrease the number of degrees of freedom."""
		p_new = self.p.copy()
		all   = np.full(p_new.strong.shape, True, dtype=bool)
		adist = calc_amp_dist(self.d.tod, d, p_new, mask=all)
		p_new.amp_rel = adist.draw() - p_new.amp_fid
		return p_new
	def draw(self):
		for i in range(self.nstep):
			self.draw_pos_beam()
		return self.draw_amps()
	def adjust(self, goal=0.25, nmin=10):
		"""Adjust step size based on current accept ratio."""
		if self.ntry < nmin: return
		current_ratio = float(self.naccept)/self.ntry
		factor = min(4.0,max(0.25,current_ratio/goal))
		L.debug("Adjusted steps by %f" % factor)
		self.dpos  *= factor
		self.dbeam *= factor
		self.naccept, self.ntry = 0,0

def make_maps(tod, data, pos, ncomp, radius, resolution):
	tod = tod.copy()
	pos = np.array(pos)
	# Handle angle wrapping
	pos = utils.rewind(pos, data.point[0,-2:])
	nsrc= len(pos)
	dbox= np.array([[-1,-1],[1,1]])*radius
	shape, wcs = enmap.geometry(pos=dbox, res=resolution)
	# Set up pixels
	n   = int(np.round(2*radius/resolution))
	boxes = np.array([[p-radius,p+radius] for p in pos])
	# Set up output maps
	rhs  = enmap.zeros((nsrc,ncomp)      +shape, wcs, dtype=dtype)
	div  = enmap.zeros((ncomp,nsrc,ncomp)+shape, wcs, dtype=dtype)
	# Build rhs
	ptsrc_data.nmat_basis(tod, data)
	ptsrc_data.pmat_thumbs(-1, tod, rhs, data.point[:,-2:], data.phase, boxes)
	# Build div
	for c in range(ncomp):
		idiv = div[0].copy(); idiv[:,c] = 1
		wtod = data.tod.astype(dtype,copy=True); wtod[...] = 0
		ptsrc_data.pmat_thumbs( 1, wtod, idiv, data.point[:,-2:], data.phase, boxes)
		ptsrc_data.nmat_basis(wtod, data, white=True)
		ptsrc_data.pmat_thumbs(-1, wtod, div[c], data.point[:,-2:], data.phase, boxes)
	div = np.rollaxis(div,1)
	mask = div[:,0] != 0
	bin = rhs.copy()
	bin[mask] = rhs[mask]/div[:,0][mask] # Fixme: only works for ncomp == 1
	return bin, rhs, div
def stack_maps(rhs, div, amps=None):
	if amps is None: amps = np.zeros(len(rhs))+1
	ar = (rhs.T*amps.T).T; ad = (div.T*amps.T**2).T
	sr = np.sum(ar,0)
	sd = np.sum(ad,0)
	return sr/sd[0]*amps[0]
def dump_maps(ofile, tod, data, pos, amp, rad=args.radius*m2r, res=args.resolution*m2r):
	ncomp = amp.shape[1]
	dmaps, drhs, ddiv = make_maps(tod.astype(dtype), data, pos, ncomp, rad, res)
	dstack = stack_maps(drhs, ddiv, amp[:,0])
	dmaps = np.concatenate((dmaps,[dstack]),0)
	dmaps = enmap.samewcs(dmaps, ddiv)
	enmap.write_map(ofile, dmaps[::-1])
	#with h5py.File(ofile, "w") as hfile:
	#	hfile["data"] = dmaps
	#	hfile["rhs"]  = drhs
	#	hfile["div"]  = ddiv
	#	hfile["stack"] = dstack

# Utility functions end

# Process all the tods, one by one
for ind in range(comm.rank, len(filelist), comm.size):
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

	d = ptsrc_data.read_srcscan(fname)
	if not args.oldformat:
		assert d.ys is not None, "Source scan in old format, but expected new"

	try:
		d, hit_srcs = validate_srcscan(d, srcs)
	except errors.DataError as e:
		L.debug("%s in %s, skipping" % (e.message, id))
		continue
	my_srcs = srcs[hit_srcs]
	nsrc = len(my_srcs)

	# Extract the mean dec,ra of this observation. This will be weighted
	# towards the exposed point sources, and so will only be approximate.
	mean_point = np.mean(d.point,0)

	pos_fid, amp_fid = my_srcs[:,posi], my_srcs[:,ampi[:ncomp]]
	#beam_fid = np.hstack([my_srcs[:,beami[:2]], my_srcs[:,beami[2:]]])
	beam_fid = np.hstack([np.full((nsrc,2),beam_global),np.zeros((nsrc,1))])
	params = Parameters(pos_fid, beam_fid, amp_fid)
	params.groups = independent_groups(d)
	# Decide which sources are strong enough to include in the full sampling
	SN = estimate_SN(d, params.flat, params.groups)**0.5
	params.strong = mask=SN >= args.strong
	desc = "SN: %.1f :" % np.sum(SN**2)**0.5
	order= np.argsort(np.sum(SN**2,1))[::-1]
	for i in order:
		desc += " %s%d:%.2f" % ("s" if np.any(params.strong[i]) else "", i, np.sum(SN[i]**2)**0.5)
	L.info(desc)
	for i in np.where(np.any(params.strong,1))[0]:
		L.debug("src %3d: SN %3.1f amp %5.0f pos %7.2f %7.2f" % ((i, np.sum(SN[i]**2)**0.5, params.amp_fid[i,0]) + tuple(params.pos_fid[i]/d2r)))

	dpos  = np.array([0.1,0.1])*m2r
	dbeam = np.array([0.1,0.1,20*d2r])
	# Disable beam sampling
	if not args.sample_beam:
		dbeam *= 0
	def prior(p, adist_strong):
		if np.sum(p.pos_rel**2)**0.5 > pos_rel_max: return -np.inf
		if np.any(p.beam_rel[:2] < beam_rel_min): return -np.inf
		if np.any(p.beam_rel[:2] > beam_rel_max): return -np.inf
		# We are supposed to be able to detect strong sources at high S/N, so
		# we should be able to require that their amplitudes are correct without
		# truncating the distribution. So I want to multiply by the fraction of
		# the probability volume that's actually allowed.
		# int_{-inf}^0 exp(-0.5(a-aml)...) daml / int_{-inf}^{inf} ... daml
		# Assuming the as are independent (good enough for a prior like this)
		# the normalized integral is just f(am) = prod 0.5*(1+erf(-1/2**0.5 A**0.5 am))
		# Since we are dealing with logarithms, what we need to return is
		# log(1-f(am)).
		# Some amplitudes are actually supposed to be negative, so invert those
		res = 0
		if adist_strong.x.size > 0:
			a_ml  = adist_strong.x
			a_fid = adist_strong.dof.zip(p.amp_fid)
			aset = a_ml * np.sign(a_fid)
			Aset = np.diag(adist_strong.A)
			def f(a,A): return 0.5*(1+erf(-2**-0.5 * A**-0.5*a))
			res = np.log(1-np.product([f(a,A) for a,A in zip(aset,Aset)]))
		return res

	if args.map:
		L.info("Building source map")
		dump_maps(tdir + "/map.hdf", d.tod, d, pos_fid, amp_fid)

	if args.grid:
		L.info("Building pos grid")
		g = np.abs(args.grid)
		p = params.copy()
		if args.grid < 0:
			p.strong[:] = False
		grid = grid_pos(d, p, shape=(g,g))
		grid -= np.max(grid)
		maxpos = grid.pix2sky(np.unravel_index(np.argmax(grid),grid.shape))
		print np.max(grid), maxpos*180*60/np.pi
		if np.sum(maxpos**2)**0.5 <= pos_rel_max*2/3:
			params.pos_rel[...] = maxpos
		enmap.write_map(tdir + "/grid.hdf", grid)

	L.info("Drawing %d samples" % args.nsamp)
	sampler = HybridSampler(d, params, dpos, dbeam, nstep=args.thin, prior=prior, dist=np.random.standard_cauchy)

	pos_rel  = np.zeros([args.nsamp,2])
	beam_rel = np.zeros([args.nsamp,3])
	amp = np.zeros((args.nsamp,)+params.amp_fid.shape)
	for i in range(-args.burnin, args.nsamp):
		if i <= 0 and (-i)%25 == 0: sampler.adjust()
		p = sampler.draw()
		L.debug("%6d %5.3f %8.3f %8.3f %8.4f %8.4f %8.2f" % ((i+1, float(sampler.naccept)/sampler.ntry) + 
				tuple(p.pos_rel/m2r) + tuple(p.beam_rel[:2]) + (p.beam_rel[2]/d2r%180,)))
		if i >= 0:
			pos_rel[i] = p.pos_rel
			beam_rel[i] = p.beam_rel
			amp[i] = p.amp_fid + p.amp_rel
		if args.fitmaps:
			tod = subtract_model(d.tod, d, p.flat)
			dump_maps(tdir + "/fitmap%05d.hdf" % i, tod, d, pos_fid, amp_fid)
	if args.fitmap:
		tod = subtract_model(d.tod, d, p.flat)
		dump_maps(tdir + "/fitmap.hdf", tod, d, pos_fid, amp_fid)

	with h5py.File(tdir + "/params.hdf", "w") as hfile:
		hfile["pos_rel"] = pos_rel
		hfile["beam_rel"] = beam_rel
		hfile["amp"] = amp
		hfile["srcs"] = hit_srcs
		hfile["SN"] = SN
		hfile["id"] = id
		hfile["point"] = mean_point

	#probs = grid_pos(d, params, shape=(40,40))
	#probs -= np.max(probs)
	#enmap.write_map(args.odir + "/grid.fits", probs)
