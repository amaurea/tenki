import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("src_fits")
parser.add_argument("base_model")
parser.add_argument("odir")
parser.add_argument("-T", "--maxdur", type=float, default=2.0, help="Split if longer than this in hours")
parser.add_argument(      "--nmin",   type=int,   default=6,   help="Skip if fewer than this many sources")
parser.add_argument("-R", "--robust", type=int,   default=1,   help="Nonzero to fit robustly, which is the default. 3x slower.")
parser.add_argument("-F", "--fit",    type=str,   default="enc_offset_az,enc_offset_el,base_tilt_cos,base_tilt_sin")
parser.add_argument(      "--detoff", type=str,   default="lat_offs.txt")
parser.add_argument("-G", "--group",  type=str,   default="obs", help="What to fit jointly. 'obs' to fit all data in obs together. 'obs,wafer' to fit each wafer separately per obs. 'obs,tube': same, but by tube. --fit-offs probably won't make sense if you fit wafers separately.")
parser.add_argument("-f", "--fit-offs", type=int, default=1)
parser.add_argument("-v", "--verbose", action="count", default=1)
parser.add_argument("-q", "--quiet",   action="count", default=0)
args = parser.parse_args()
import numpy as np
from pixell import utils, bunch, coordsys, colors, bench
from scipy import ndimage, optimize

DEG = utils.degree
AMIN= DEG/60

# Radial-roll model
# --------------
# Residuals from fitting the standard v2 model + wafer offsets across the
# whole focal plane # show a strange pattern where
#  Δhor = Δhor0 * (1-(r/r0)**2) * (roll-roll0)
# with unknowns Δhor0[2], r0 and roll0. I would like to fit these jointly
# with the wafer offsets and the v2 parameters, instead of having those as
# ad-hoc steps added at the end.
#
# Parameters can have three statues: fixed, global and per-obs.
# Wafer offsets are a bit awkward because there's a variable number
# of them, up to 7*3 = 21. Might be easiest to just have them all
# be present all the time, but fix the unused ones.
# We would then have a set of parameters woff_i1_ws0_xi, woff_i1_ws0_eta, etc.
#
# The parameter specification would then be [name, mode, val], which we can
# represent as a numpy structured array. val would be the global value, or
# the one we get after evaluating for a single obs. The full state would be
# bigger and would need a different repsentation, e.g. with zip/unzip.

# Doing the fitting
# -----------------
# The data consists of individual offset measurements, each of which
# belongs to a time-group (where parameters are assumed to be constant)
# and an area in the focal-plane (e.g. a wafer or a tube).
#
# More generally, we can have a labeling function lab(measurement) → label,
# where label is an N-tuple. Entry i in the tuple corresponds to something
# a parameter could depend on. For example, for a model with parameters that can be
# static, position-dependent or time-dependent, so 3 types of parameters,
# the label could be (0, "i1:ws0", 10), where the first entry
# is an arbitrary number representing the single static category,
# the 2nd corresponds to the position type, and the 3rd to the time-group
# (10th time group in this case).
#
# Parameters would then *not* have e.g. the wafer baked into their names like I
# considered earlier. Instead, they would be something like
# * name=encoder_az_offset, fit=True,  depend=2 (time-dependent)
# * name=encoder_el_offset, fit=True,  depend=2 (time-dependent)
# * name=base_tilt_sin,     fit=False, depend=2 (theoretically time-dependent, but kept fixed at initial value here)
# * name=offset_xi,         fit=True,  depend=1 (position-dependent)
# * name=offset_eta,        fit=True,  depend=1 (position-dependent)
# * name=r0,                fit=True,  depend=0 (global)
#
# What about parameters that depend on multiple things, e.g. both position and time?
# That's easily handled in the lab function. Just make an extra
# category for that, and return e.g. "i1:ws0,10". So we only need to concern
# ourselves with a single dependency here.
#
# Should fit be a separate field, or merged with depend, e.g. negative depend?
# The latter would make it easier to specify the model from the command line,
# but the model needs a full implementation in python anyway, so that's not
# really relevant.
#
# We can see the pointing model as something that maps
#  (baz,bel,broll) → (az,el,roll)
# where the latter are the actual wafer quantities. The model has parameters,
# and may also depend on other quantities from the catalog, but these are extra.
# These will be plain numpy arrays [ndata,{az,el,roll}]
#
# So to fit, we need to specify:
# * data, a numpy structured array with arbitrary dtype. Contains both what what we
#   want to predict (e.g. pointing offsets) and things the model could depend on.
# * label(data) → keys, keys = (key0, key1, ... , keyN)
# * eval(pfit, dof, cat, icoord) → ocoord. Produce the model wafer pointing in horizontal
#   coordinages given a parameter fit, the catalog and raw boresight coordinates.
# * inverse(pfit, dof, cat, ocoord) → icoord. Recover raw boresight coordinates from
#   pointing-corrected wafer coordinates, a pointing model and a catalog. This is
#   how we get the icoords we need in eval. Initially, the wafer coordinates are
#   constructed from cat.ra, etc. Then icoord is recovered with inverse and a fiducial
#   model, which should match the one used in depth1_pointing.py.
#   A complication is that we know icoord.roll from the beginning, but not ocoord.roll.
#   This makes inverse a bit hacky.
# * fit(params, dof, cat, icoord, gcoord, dcoord) → pfit. paramas differs from pfit by simply having one
#   entry per parameter. Or should this be a pfit too? What if we need a non-static
#   starting guess? gcoords = the goal coordinates, what we want to match.
#   dcoord = uncertainty in gcoord. Used in likelihood.
#
# The inputs concept is useful. This, and data, should be model-independent quantities
# that we can use together with multiple models.
# inputs contain at least the boresight coordinates for the observation. That's not
# directly
#
# Let's encapsulate the model-specific stuff in a Model class.

# Since we handle time-dependence via generic labeling, how can we record
# the relevant time-ranges for each time-group in the outputs? In general,
# each label-class needs not just some symbolic label, but actual values.
# One solution would be to let the label itself be that value. E.g.
# instead of having label return ("i1:ws0", 10) for the 10th time-range,
# it could return ("i1:ws0", (ctime_min, ctime_max)). Alternatively,
# label could return number tuples, and then a separate list for the values.
# That would reduce duplication.

# Need to decide on the representation of the parameters. I think there are
# three different concepts here:
# 1. The parameter definition pinfo. One row per parameter, but no values.
# 2. The parameters we work with in functions like eval, inverse etc.
#    Structured 1D recarray with length ncat. Dtype will be built from pinfo.
# 3. The degrees of freedom. plain 1d array
# Useful to have an object that takes care of doing these translations.
# Need pinfo and labels to build. Should the model objects store this inside?
# If so, they will need a cat to be built, and would call label internally.
# Alternatively, the user builds it by calling label manually. Would let
# user use the same model object on different subsets of data, but cumbersome,
# and how often would I need that?

# How to do the actual fit?
# 1. A gibbs-like approach where paramaeters are split into into groups that
#    consist of chunks that can be fit independently, and then iterate over
#    fits of such groups. E.g. if static parameters and wafer parameters are
#    held fixed, then each time-bin can be fit by itself, and if time and
#    static are held fixed, then eact wafer can be fit by itself. At the
#    abstract level I'm working at here, this gets cumbersome to implement,
#    but all the information needed is encoded in pinfo.depend.
# 2. Use Newton-CG, which I think should handle the high dimensionality of the problem.
#    This requires the implementation of derivatives, which is also cumbersome.

# name = name of parameter, e.g. "encoder_az_offset"
# type = "fixed", "area" or "time"
# Could add a unit field here, if I want different units on disk, or for display
dtype_pinfo = [("name", "U25"), ("depend", "i"), ("fit","i"), ("default", "d"), ("scale", "d")]

# How should I handle things that aren't fit? These should not
# enter into the degrees of freedom, but they should still be given
# a value when expanded.
# 1. add a default value field to pinfo. Won't work because e.g.
#    wafer offsets will have separate values per wafer.
# 2. pass in a set of default values. These would be in expanded form.
#    No seprate lookup structures needed.
# 3. Make a separate Pzipper for default values. Could use this to
#    to generate the epxanded form needed in #2.
# Going with 2+3

# We fit for wafer coordinates, but in the actual mapmaking we will be dealing
# with detectors, not wafers, so what we need is a per-wafer correction to apply
# per wafer. I want to be able to write out something useful with dof.write, so
# maybe the params themselves would contain offsets, not absolute coordinates
# But I do need the abolute coordinates in the model. How about this?
#  waf_pos_xi, waf_pos_eta are the fidicual pos, and are not fit
#  waf_off_xi, waf_off_eta are the offsets we will fit, and are relative to the above
# Then the mapmaker only needs to care about waf_off_*.

class Pzipper:
	dtype_annot = [("name", "U32"), ("type", "U32"), ("bin", "U100"), ("val", "d")]
	def __init__(self, pinfo, labels, defaults=[]):
		self.pinfo    = pinfo
		self.labels   = labels
		if   defaults is None: defaults = []
		elif isinstance(defaults, np.ndarray): defaults = [defaults]
		# Pre-pend defaults from pinfo
		defaults = [pinfo2defaults(pinfo)]+defaults
		self.defaults = defaults
		# Build a dtype for pfull
		self.dtype    = [(p["name"], "d") for p in pinfo]
		self.labels   = labels
		# Find how many degrees of freedom there are per label-type
		lname, bins, bnames = labels
		lnum = [len(bname) for bname in bnames]
		# Build the mapping between x and pfull
		ndof = 0
		map = []
		for p in pinfo:
			name, dep, fit, scale = p["name"], p["depend"], p["fit"], p["scale"]
			if not fit: continue
			# How many degrees of freedom we contribute with
			pnum  = lnum [dep]
			# Which degree of freedom each entry in pfull[name]
			# is associated with. Many pfull entries can map to
			# a single degree of freedom, but a degree of freedom
			# will map to at least one pfull entry.
			pbins = bins[dep]
			# name:  name of parameter
			# dep:   what parameter depends on. Index into labels
			# pbins: which bin each cat entry falls into for this parameter
			# xinds: which indices into the x vector we map to.
			# scale: divide by this when going to x
			map.append((name, dep, pbins, np.arange(pnum)+ndof, scale))
			ndof += pnum
		self.map   = map
		self.ndof  = ndof
		self.ntype = len(lname)
		self.nfull = len(bins[0])
	def default(self):
		# This one is straightforward, since it's a broadcasting operation
		pfull = np.zeros(self.nfull, self.dtype).view(np.recarray)
		# Initialize with defaults, if available.
		# It's done like this to allow us to use the result of a model
		# with fewer parameters than ours as defaults. The try/except
		# also allows us to ignore any spurious parameters in the input
		for default in self.defaults:
			for name in default.dtype.names:
				try: pfull[name] = default[name]
				except ValueError: pass
		return pfull
	def unzip(self, x):
		pfull = self.default()
		for name, dep, pbins, xinds, scale in self.map:
			pfull[name] = x[xinds][pbins]*scale
		return pfull
	def adjoint_unzip(self, pfull):
		"""The adjoint of the unzip operation. Unlike zip, this accumulates and
		applies the scaling again. This operation probably isn't needed"""
		x = np.zeros(self.ndof)
		for name, dep, pbins, xinds, scale in self.map:
			x[xinds] = np.bincount(pbins, pfull[name])*scale
		return x
	def zip(self, pfull):
		"""Extract our degrees of freedom from the expanded parameters.
		One degree of freedom can map to many expanded parameters, in which
		case the average will be returned. But usually this would just be the
		average of the same, duplicated value anyway."""
		x = np.zeros(self.ndof)
		for name, dep, pbins, xinds, scale in self.map:
			# pbins won't have any holes, so don't need minlength
			#print("name", name)
			#print("dep", dep)
			#print("pbins", pbins)
			#print("xinds", xinds)
			#print("pfull", pfull)
			x[xinds] = np.bincount(pbins, pfull[name])/np.bincount(pbins)/scale
		return x
	def split(self):
		"""Return a list of smaller Pzippers, each of which deals with only
		a subset of degrees of freedom and the smallest amount of data needed
		for these. E.g. one that only deals with static parameter,
		one for each wafer and one for each time-chunk. This is useful for
		a Gibbs-like solution schemes, since optimizing the full parameter
		set at once can be slow, especially without derivatives."""
		# For each bin for each type we will build:
		# * inds of relevant subset of cat/pfull
		# * sub-labels (trivial with those inds)
		# * inds of sub-x into full-x. x has order [params][bins]

		# Full x order, where A, B, C are params, with B and C being in the same class,
		# and the number is the bin
		#  A1 A2 A3
		#  B1 B2 B3 B4
		#  C1 C2 C3 C4
		# Sub-xs should be
		#  A1, A2, A3, B1 C1, B2 C2, B3 C3, B4 C4
		# So to get the mapping, I just need to know how many bins there are for each type,
		# and how many parameters share that type
		splits     = []
		default    = self.default()
		# First build the xinds mapping, which requires parameter-first looping instead
		# of class-first looping
		nper       = [len(bnames) for bnames in self.labels[2]]
		sub_xindss = [[[] for bi in range(nper[ti])] for ti in range(self.ntype)]
		i_full     = 0
		for p in self.pinfo:
			name, ti, fit = p["name"], p["depend"], p["fit"]
			if not fit: continue
			for bi in range(nper[ti]):
				sub_xindss[ti][bi].append(i_full)
				i_full += 1
		# Then infer inds and sub-labels and build the new Pzippers
		for ti in range(self.ntype):
			# Will only fit parameters in this group
			sub_pinfo = self.pinfo.copy()
			sub_pinfo.fit[sub_pinfo.depend != ti] = False
			# Get the groups for this type. E.g. for the time-dependency,
			# this would loop through the individual time-bins
			lname, bins, bnames = [a[ti] for a in self.labels]
			ubins, order, edges = utils.find_equal_groups_fast(bins)
			# There's one entry in x for each group per parameter that group is relevant for
			for bi, bin in enumerate(ubins):
				sub_name    = "%s[%s]" % (lname, bnames[bi])
				# This is the inds we want
				sub_inds    = order[edges[bi]:edges[bi+1]]
				# Each subset will still list all dep-types, but only entry [ti] will
				# be used in practice.
				sub_lnames  = self.labels[0]
				sub_lbins   = [np.zeros(len(sub_inds),int) for i in range(self.ntype)]
				sub_bnames  = [bnames[:0] for i in range(self.ntype)]
				sub_bnames[ti] = bnames[bi:bi+1]
				# Each subset will only have a single type and single bin
				sub_labels  = [sub_lnames, sub_lbins, sub_bnames]
				sub_xinds   = np.array(sub_xindss[ti][bi],dtype=int)
				sub_dof     = Pzipper(sub_pinfo, sub_labels, defaults=default[sub_inds])
				splits.append(bunch.Bunch(inds=sub_inds, xinds=sub_xinds, dof=sub_dof, name=sub_name))
		# Phew!
		return splits
	def annot(self, x):
		"""Given a plain degrees of freedom vector x, returns a structured
		array suitable for printing, with labels for each entry. This will
		still have the same length as x, so anything not fit will not be
		included. If you want to include any fixed priors too, then
		the easiest way is probably to make a Model with inverse fit flags
		and then annotating that."""
		xannot = np.zeros(len(x), self.dtype_annot).view(np.recarray)
		for name, dep, pbins, xinds, scale in self.map:
			xannot.name[xinds] = name
			xannot.type[xinds] = self.labels[0][dep]
			xannot.bin [xinds] = self.labels[2][dep]
			xannot.val [xinds] = x[xinds]*scale
		return xannot
	def parse(self, xannot):
		"""Create an x from an xannot read from disk. Unlike just
		doing xannot.val, this is robust to reordering. It's a bit
		inefficient, but this function won't be called much anyway."""
		targ  = self.annot(xannot.val)
		ikeys = [" ".join((row["name"], row["type"], row["bin"])) for row in xannot]
		okeys = [" ".join((row["name"], row["type"], row["bin"])) for row in targ]
		order = utils.find(ikeys, okeys, default=-1)
		x     = np.zeros(len(xannot))
		for i, (name, dep, pbins, xinds, scale) in enumerate(self.map):
			inds = order[xinds]
			vals = np.where(inds >= 0, xannot.val[inds], self.pinfo.default[i])
			x[xinds] = vals/scale
		return x
	def read(self, fname, full=False):
		"""Read a parameter file like what write produces, returning an x.
		Note that only parameters that can be represented in this Pzipper's
		active degrees of freedom will be included! Therefore you should
		usually call this function with a model.dof_full, not just a plain model.dof"""
		xannot = np.loadtxt(args.base_model, dtype=Pzipper.dtype_annot).view(np.recarray)
		x      = self.parse(xannot)
		if full: return self.unzip(x)
		else: return x
	def write(self, fname, x, full=False, header=""):
		if full: x = self.zip(x)
		xannot   = self.annot(x)
		lens     = np.max([[len(row[a]) for a in ["name", "type", "bin"]] for row in xannot],0)
		lens     = tuple(map(int, lens))
		fmt      = "%%-%ds %%-%ds %%-%ds %%15.7e" % lens
		np.savetxt(fname, xannot, fmt=fmt, header=header)

def get_pinfo(pinfo0, fit=None):
	pinfo = pinfo0.copy()
	if   fit is None: return pinfo
	if isinstance(fit, str): fit = [fit]
	kmap = {p["name"]:i for i,p in enumerate(pinfo)}
	for name in fit:
		if name == "*": pinfo.fit[:] = True
		else: pinfo.fit[kmap[name]] = True
	return pinfo

class Model:
	def __init__(self, cat, dof=None, dof_full=None, fit=None, defaults=[]):
		self.cat = cat
		labels   = None
		# These are the standard degrees of freedom that we will fit
		if dof is None:
			pinfo = get_pinfo(self.pinfo0, fit=fit)
			if labels is None: labels = self.label(cat)
			dof   = Pzipper(pinfo, labels, defaults=defaults)
		self.dof = dof
		# These are the same, except everything that can be fit is included.
		# This is mainly useful when reading in parameters
		if dof_full is None:
			pinfo_full = get_pinfo(self.pinfo0, fit="*")
			if labels is None: labels = self.label(cat)
			dof_full = Pzipper(pinfo_full, labels, defaults=defaults)
		self.dof_full = dof_full
	def variant(self, cat, dof): return self.__class__(cat, dof=dof)
	@property
	def pinfo(self): return self.dof.pinfo
	def label(self, cat):
		"""Labels are (name[ntype], bin[ntype][nrow], bname[ntype][nval])
		* Name[type] is the name a given type of label, e.g. "static", "wafer", "time" etc.
		* bin [type] is which bin within the type each entry in cat falls into. E.g.
		  if there are 50 time-splits, then each entry in cat would be assigned some number
		  between 0 and 49 inclusive, with all numbers in that range being used at least once.
		* bname[type] is a string identifying each bin, e.g. for the time-split, it could be
			length 50 with entries like "1777800000:177790000", and for the wafer split it
			could have entries like "i1:ws0"."""
		name  = ("static",)
		bin   = (np.zeros(len(cat),int),)
		bname = (np.array(["-"]),)
		return name, bin, bname
	def split(self):
		splits = self.dof.split()
		for split in splits:
			split.model = self.variant(self.cat[split.inds], split.dof)
		return splits
	def eval(self, pfull, icoord): raise NotImplementedError
	def inverse(self, pfull, ocoord, niter=10):
		"""Calculate icoord = [baz,bel,broll].T given ocoord[az,el,psi] such that
		eval(icoord) = ocoord. baz, bel and psi are unknown. broll is taken from cat.
		Returns icoord, ocoord, where the inferred psi can be read out from the ocoord."""
		# Initial guess
		ocoord = ocoord.copy()
		icoord = ocoord.copy()
		for it in range(niter):
			icoord = self._approx_inverse(pfull, ocoord, icoord)
			ocoord[:,2] = self.eval(pfull, icoord)[:,2]
		return icoord, ocoord
	def fit(self, icoord, tcoord, dcoord, verbose=0, prefix=[]):
		"""Fit a model such that input coordinates icoord are transformed into
		target coordinates tcoord with as small error as possible"""
		if self.dof.ndof == 0: return
		pfull = self.dof.default()
		x     = self.dof.zip(pfull)
		step  = [0]
		def calc_chisq(x):
			pfull  = self.dof.unzip(x)
			mcoord = self.eval(pfull, icoord)
			# only ra,dec parts used in fit, not psi
			chisq  = np.sum((mcoord[:,:2]-tcoord[:,:2])**2/dcoord[:,:2]**2)
			step[0] += 1
			if verbose > 0:
				print("%s%4d %15.7e " % (" ".join(prefix), step[0], chisq) + " ".join(["%12.5e" % val for val in x][:8]))
			return chisq
		x = optimize.fmin_powell(calc_chisq, x, disp=False)
		return x
	def fit_gibbs(self, icoord, tcoord, dcoord, niter=3, verbose=0, prefix=[]):
		# If needed, set our initial condition by writing to self.dof.defaults.
		# Yes, awkward.
		pfull  = self.dof.default()
		x      = self.dof.zip(pfull)
		splits = self.split()
		for it in range(niter):
			for si, split in enumerate(splits):
				if verbose > 0:
					print("%s %3d/%d %3d/%d %s" % (" ".join(prefix), it+1, niter, si+1, len(splits), split.name))
				inds, xinds, model = split.inds, split.xinds, split.model
				# The model/dof/dof-internals split is awkward. Anyway, we pass in
				# both the initial condition and state of the parameters we're not
				# fitting in this split by updating dof.defaults, 
				model.dof.defaults = [pfull[inds]]
				#print("A", split.name)
				#print("inds", inds)
				#print("defaults")
				#print(model.dof.defaults)
				x[xinds] = model.fit(icoord[inds], tcoord[inds], dcoord[inds], verbose=verbose-1, prefix=prefix+[split.name,"%02d"%it])
				# Need to keep x and pfull in sync. Could do this with a full
				# unzip, but a partial unzip should be equivalent and faster
				pfull[inds] = model.dof.unzip(x[xinds])
			if verbose > 0:
				mcoord = self.eval(pfull, icoord)
				chisq  = np.sum((mcoord[:,:2]-tcoord[:,:2])**2/dcoord[:,:2]**2)
				print("%s%4d %15.7e" % (" ".join(prefix), it+1, chisq))
		return x
	def fit_robust(self, icoord, tcoord, dcoord, niter=3, etol=3, errs=[0.3*utils.arcmin, 0.1*utils.arcmin, 0], verbose=0, prefix=[]):
		penalty = np.zeros(len(icoord))
		for it, err in enumerate(errs):
			dcoord_eff = (dcoord**2 + err**2 + penalty[:,None])**0.5
			x = self.fit_gibbs(icoord, tcoord, dcoord_eff, niter=niter, verbose=verbose, prefix=prefix + ["rit %d/%d"%(it+1,len(errs))])
			if it < len(errs)-1:
				pfull  = self.dof.unzip(x)
				mcoord = self.eval(pfull, icoord)
				excess = np.mean(utils.rewind(tcoord[:,:2]-mcoord[:,:2])**2,1) - np.mean(dcoord_eff**2,1)
				penalty += np.maximum(0, excess - etol**2*np.mean(dcoord_eff**2,1))
		return x, dcoord_eff
	def _approx_inverse(self, pfull, ocoord, icoord):
		"""Given pfull, ocoord and icoord such that eval(pfull, icoord) ≈ ocoord,
		return a new icoord that results in an even closer approximation."""
		raise NotImplementedError
	pinfo0 = np.zeros(0, dtype=dtype_pinfo).view(np.recarray)

class ModelStaticV2(Model):
	name = "lat_v2"
	def __init__(self, cat, dof=None, fit=None, defaults=[]):
		Model.__init__(self, cat, dof=dof, fit=fit, defaults=defaults)
	def label(self, data):
		toks = parse_names(data.name)
		ustatic, istatic = np.array(["-"]), np.zeros(len(data),int)
		uwafer,  iwafer  = np.unique(toks.wafer, return_inverse=True)
		name  = ("static", "wafer")
		inds  = (istatic, iwafer)
		bname = (ustatic, uwafer)
		return name, inds, bname
	def eval(self, pfull, icoord):
		baz, bel, broll = icoord.T
		q_base, q_lonlat, q_middle, q_det = build_v2_quats(pfull, baz, bel, broll)
		coords = coordsys.Coords(q=q_base * q_lonlat * q_middle * q_det)
		ocoord = np.array(restore_el(bel, coords.az, coords.el, coords.roll)).T
		return ocoord
	def _approx_inverse(self, pfull, ocoord, icoord):
		"""Estimate icoord such that eval(icoord) approximates ocoord, given
		an initial guess for icoord."""
		# Having an initial guess let's us use that for the nonlinear parts, making
		# the operation linear
		oaz,  oel,  oroll = ocoord.T
		baz0, bel0, roll0 = icoord.T
		q_base, q_lonlat, q_middle, q_det = build_v2_quats(pfull, baz0, bel0, roll0)
		# We start from q_tot. This is the final pointing in eval, so our starting point here
		q_tot = coordsys.Coords(az=oaz, el=oel, roll=oroll).q
		# Recover q_lonlat, which contains our boresight az and el, but *not* roll!
		q_lonlat = 1/q_base * q_tot / q_det / q_middle
		c_lonlat = coordsys.Coords(q=q_lonlat)
		baz, bel, dummy = restore_el(bel0, c_lonlat.az, c_lonlat.el, c_lonlat.roll)
		# Quaternion part done. Do the inverse sag. Can solve this part exactly, but
		# more consistent with the rest to just use our starting guess
		Δel  = (bel0+pfull.enc_offset_el) - pfull.el_sag_pivot
		bel -= Δel*pfull.el_sag_lin + Δel**2 * pfull.el_sag_quad
		baz -= pfull.enc_offset_az
		bel -= pfull.enc_offset_el
		return np.array([baz, bel, roll0]).T
	# Here 0 = static, 1 = pos-dependent
	pinfo0 = np.array([
		("waf_pos_xi",          1, False,0,       1*AMIN),
		("waf_pos_eta",         1, False,0,       1*AMIN),
		("waf_off_xi",          1, True, 0,       1*AMIN),
		("waf_off_eta",         1, True, 0,       1*AMIN),
		("enc_offset_az",       0, True, 0,       1*DEG),
		("enc_offset_el",       0, True, 0,       1*DEG),
		("enc_offset_cr",       0, True, 0,       1*DEG),
		("base_tilt_sin",       0, True, 0,       1*DEG),
		("base_tilt_cos",       0, True, 0,       1*DEG),
		("el_axis_center_xi0",  0, True, 0,       1*DEG),
		("el_axis_center_eta0", 0, True, 0,       1*DEG),
		("mir_center_xi0",      0, True, 0,       1*DEG),
		("mir_center_eta0",     0, True, 0,       1*DEG),
		("cr_center_xi0",       0, True, 0,       1*DEG),
		("cr_center_eta0",      0, True, 0,       1*DEG),
		("el_sag_pivot",        0, True, 90*DEG,  1*DEG),
		("el_sag_lin",          0, True, 0,       0.01),
		("el_sag_quad",         0, True, 0,       0.01),
	], dtype=dtype_pinfo).view(np.recarray)

class ModelDynamicV2(ModelStaticV2):
	name = "lat_v2"
	def __init__(self, cat, dof=None, fit=None, defaults=[], maxdur=2.0*utils.hour):
		self.maxdur = maxdur
		ModelStaticV2.__init__(self, cat, dof=dof, fit=fit, defaults=defaults)
	def label(self, data):
		toks = parse_names(data.name)
		uwafer,  iwafer  = np.unique(toks.wafer, return_inverse=True)
		# For time, we want to split by obs (really depth1-id), but split if too long
		# split by time and pattern
		utpat,   itpat   = build_tpat(data, tol=self.tol, maxdur=self.maxdur)
		name  = ("tpat", "wafer")
		inds  = (itpat, iwafer)
		bname = (utpat, uwafer)
		return name, inds, bname
	pinfo0 = np.array([
		("enc_offset_az",       0, True,    0,       1*DEG),
		("enc_offset_el",       0, True,    0,       1*DEG),
		("enc_offset_cr",       0, True,    0,       1*DEG),
		("base_tilt_sin",       0, False,   0,       1*DEG),
		("base_tilt_cos",       0, False,   0,       1*DEG),
		("el_axis_center_xi0",  0, False,   0,       1*DEG),
		("el_axis_center_eta0", 0, False,   0,       1*DEG),
		("mir_center_xi0",      0, False,   0,       1*DEG),
		("mir_center_eta0",     0, False,   0,       1*DEG),
		("cr_center_xi0",       0, False,   0,       1*DEG),
		("cr_center_eta0",      0, False,   0,       1*DEG),
		("el_sag_pivot",        0, False,   90*DEG,  1*DEG),
		("el_sag_lin",          0, False,   0,       0.01),
		("el_sag_quad",         0, False,   0,       0.01),
		("waf_pos_xi",          1, False,   0,       1*AMIN),
		("waf_pos_eta",         1, False,   0,       1*AMIN),
		("waf_off_xi",          1, True,    0,       1*AMIN),
		("waf_off_eta",         1, True,    0,       1*AMIN),
	], dtype=dtype_pinfo).view(np.recarray)

class ModelRadRollHor(Model):
	name = "rad_roll_hor"
	def __init__(self, cat, dof=None, fit=None, defaults=[], maxdur=2.0*utils.hour, tol=1*utils.degree):
		self.maxdur = maxdur
		self.tol    = tol
		Model.__init__(self, cat, dof=dof, fit=fit, defaults=defaults)
	def variant(self, cat, dof):
		return ModelRadRollHor(cat, dof=dof, maxdur=self.maxdur)
	def label(self, data):
		toks = parse_names(data.name)
		ustatic, istatic = np.array(["-"]), np.zeros(len(data),int)
		uwafer,  iwafer  = np.unique(toks.wafer, return_inverse=True)
		# split by time and pattern
		utpat,   itpat   = build_tpat(data, tol=self.tol, maxdur=self.maxdur)
		name  = ("tpat", "wafer", "static")
		inds  = (itpat, iwafer, istatic)
		bname = (utpat, uwafer, ustatic)
		return name, inds, bname
	def eval(self, pfull, icoord):
		# This model is identical to the static one, except we apply an r-dependent,
		# roll-dependent displacement of az,el at the end
		baz, bel, broll = icoord.T
		q_base, q_lonlat, q_middle, q_det = build_v2_quats(pfull, baz, bel, broll)
		coords = coordsys.Coords(q=q_base * q_lonlat * q_middle * q_det)
		# Get the radius in the focal plane
		r2    = pfull.waf_pos_xi**2 + pfull.waf_pos_eta**2
		scale = (1-r2/pfull.roff_r0**2) * (broll-pfull.roff_roll0)
		coords.az += pfull.roff_offset_az * scale
		coords.el += pfull.roff_offset_el * scale
		ocoord = np.array(restore_el(bel, coords.az, coords.el, coords.roll)).T
		return ocoord
	# Here 2 = static, 1 = wafer-dependent and 0 = time-dependent
	pinfo0 = np.array([
		("enc_offset_az",       0, True,    0,       1*DEG),
		("enc_offset_el",       0, True,    0,       1*DEG),
		("enc_offset_cr",       0, True,    0,       1*DEG),
		("base_tilt_sin",       0, False,   0,       1*DEG),
		("base_tilt_cos",       0, False,   0,       1*DEG),
		("el_axis_center_xi0",  0, False,   0,       1*DEG),
		("el_axis_center_eta0", 0, False,   0,       1*DEG),
		("mir_center_xi0",      0, False,   0,       1*DEG),
		("mir_center_eta0",     0, False,   0,       1*DEG),
		("cr_center_xi0",       0, False,   0,       1*DEG),
		("cr_center_eta0",      0, False,   0,       1*DEG),
		("el_sag_pivot",        0, False,   90*DEG,  1*DEG),
		("el_sag_lin",          0, False,   0,       0.01),
		("el_sag_quad",         0, False,   0,       0.01),
		("waf_pos_xi",          1, False,   0,       1*AMIN),
		("waf_pos_eta",         1, False,   0,       1*AMIN),
		("waf_off_xi",          1, True,    0,       1*AMIN),
		("waf_off_eta",         1, True,    0,       1*AMIN),
		("roff_r0",             2, True,    1*DEG, 0.1*DEG),
		("roff_roll0",          2, False,   0,       1*DEG),
		("roff_offset_az",      2, True,    0,       1*DEG),
		("roff_offset_el",      2, True,    0,       1*DEG),
	], dtype=dtype_pinfo).view(np.recarray)

class ModelRadRollArc(Model):
	name = "arc"
	def __init__(self, cat, dof=None, fit=None, defaults=[], maxdur=2.0*utils.hour, tol=1*utils.degree):
		self.maxdur = maxdur
		self.tol    = tol
		Model.__init__(self, cat, dof=dof, fit=fit, defaults=defaults)
	def variant(self, cat, dof):
		return ModelRadRollArc(cat, dof=dof, maxdur=self.maxdur)
	def label(self, data):
		toks = parse_names(data.name)
		ustatic, istatic = np.array(["-"]), np.zeros(len(data),int)
		uwafer,  iwafer  = np.unique(toks.wafer, return_inverse=True)
		utpat,   itpat   = build_tpat(data, tol=self.tol, maxdur=self.maxdur)
		name  = ("tpat", "wafer", "static")
		inds  = (itpat, iwafer, istatic)
		bname = (utpat, uwafer, ustatic)
		return name, inds, bname
	def eval(self, pfull, icoord):
		# This model is identical to the dynamic v2 one, except we
		# apply an r-dependent, roll-dependent offset of xieta
		baz, bel, broll = icoord.T
		# Calcluate the xi eta offsets
		r2    = pfull.waf_pos_xi**2 + pfull.waf_pos_eta**2
		scale = pfull.arc_amp * (1-r2/pfull.arc_r0**2)
		ang   = broll-pfull.arc_roll0
		dxi   = scale * (np.cos(ang)-1)
		deta  = scale * np.sin(ang)
		# Apply it
		pfull_off = pfull.copy()
		pfull_off.waf_off_xi  += dxi
		pfull_off.waf_off_eta += deta
		# The rest goes as normal
		q_base, q_lonlat, q_middle, q_det = build_v2_quats(pfull_off, baz, bel, broll)
		coords = coordsys.Coords(q=q_base * q_lonlat * q_middle * q_det)
		ocoord = np.array(restore_el(bel, coords.az, coords.el, coords.roll)).T
		return ocoord
	# Here 2 = static, 1 = wafer-dependent and 0 = time-scanpat
	# In this model, most of the v2 static parameters are static
	# and not fit. Might want to change that in variant models in the
	# future, but for now this seems to perform well
	pinfo0 = np.array([
		("enc_offset_az",       0, True,    0,       1*DEG),
		("enc_offset_el",       0, True,    0,       1*DEG),
		("enc_offset_cr",       0, True,    0,       1*DEG),
		("base_tilt_sin",       2, False,   0,       1*DEG),
		("base_tilt_cos",       2, False,   0,       1*DEG),
		("el_axis_center_xi0",  2, False,   0,       1*DEG),
		("el_axis_center_eta0", 2, False,   0,       1*DEG),
		("mir_center_xi0",      2, False,   0,       1*DEG),
		("mir_center_eta0",     2, False,   0,       1*DEG),
		("cr_center_xi0",       2, False,   0,       1*DEG),
		("cr_center_eta0",      2, False,   0,       1*DEG),
		("el_sag_pivot",        2, False,   90*DEG,  1*DEG),
		("el_sag_lin",          2, False,   0,       0.01),
		("el_sag_quad",         2, False,   0,       0.01),
		("waf_pos_xi",          1, False,   0,       1*AMIN),
		("waf_pos_eta",         1, False,   0,       1*AMIN),
		("waf_off_xi",          1, True,    0,       1*AMIN),
		("waf_off_eta",         1, True,    0,       1*AMIN),
		("arc_amp",             2, True,    0,       1*AMIN),
		("arc_r0",              2, True,    1*DEG, 0.1*DEG),
		("arc_roll0",           2, True,    0,       1*DEG),
	], dtype=dtype_pinfo).view(np.recarray)

def build_v2_quats(pfull, baz, bel, roll):
	"""Build the quaternion building blocks used in the static v2 model.
	pfull must broadcast with baz, bel and roll."""
	corot = bel - roll - 60*utils.degree
	# Apply offsets
	az     = baz   + pfull.enc_offset_az
	el     = bel   + pfull.enc_offset_el
	corot  = corot + pfull.enc_offset_cr
	# El sag
	Δel    = el     - pfull.el_sag_pivot
	el    += Δel    * pfull.el_sag_lin
	el    += Δel**2 * pfull.el_sag_quad
	q_lonlat     = coordsys.Coords(az=az, el=el).q
	q_mir_center = 1/coordsys.rotation_xieta(pfull.mir_center_xi0, pfull.mir_center_eta0)
	q_el_roll    = coordsys.euler(2, el - 60*utils.degree)
	q_el_axis_center = 1/coordsys.rotation_xieta(pfull.el_axis_center_xi0, pfull.el_axis_center_eta0)
	q_cr_roll    = coordsys.euler(2, -corot)
	q_cr_center  = 1/coordsys.rotation_xieta(pfull.cr_center_xi0, pfull.cr_center_eta0)
	q_middle     = q_mir_center * q_el_roll * q_el_axis_center * q_cr_roll * q_cr_center
	# Base tilt
	phi = np.arctan2(pfull.base_tilt_sin, pfull.base_tilt_cos)
	amp = (pfull.base_tilt_sin**2 + pfull.base_tilt_cos**2)**0.5
	q_base = coordsys.euler(2,phi) * coordsys.euler(1, amp) * coordsys.euler(2, -phi)
	# Detectors
	q_det  = coordsys.rotation_xieta(
		pfull.waf_pos_xi  + pfull.waf_off_xi,
		pfull.waf_pos_eta + pfull.waf_off_eta,
	)
	return q_base, q_lonlat, q_middle, q_det

def azel2xieta_v2(pfull, coord, icoord):
	"""Calculate the xi,eta focal plane coordinates given horizontal coordinates coords and
	boresight coordinates icoords, assuming a v2 static model. If you only need focal
	plane *offsets*, then it should be fine to use this for other similar models too."""
	q_base, q_lonlat, q_middle, q_det = build_v2_quats(pfull, icoord[:,0], icoord[:,1], icoord[:,2])
	q_tot = coordsys.Coords(az=coord[:,0], el=coord[:,1], roll=coord[:,2]).q
	# q_tot = q_base * q_lonlat * q_middle * q_det. Want to infer q_det instead of using the
	# fiducial one
	q_odet = (1/q_middle) * (1/q_lonlat) * (1/q_base) * q_tot
	return np.array(coordsys.decompose_xieta(q_odet)).T

def build_tpat(data, tol=1*utils.degree, maxdur=2*utils.hour):
	"""Build the tpat = time+scanpat split"""
	# First split by time and pattern
	toks = parse_names(data.name)
	itime, tmins, tmaxs = time_split(toks.obs, data.ctime, maxdur=maxdur)
	ipat, azs,els,rolls = pattern_split(data.pat_baz, data.pat_bel, data.roll, tol=tol)
	# Then combine
	itpat, iinds  = utils.label_multi([itime,ipat], return_index=True)
	tmins, tmaxs  = [a[itime[iinds]] for a in [tmins, tmaxs]]
	azs,els,rolls = [a[ipat [iinds]] for a in [azs,els,rolls]]
	# Try to keep name reasonably compact
	names = np.array(["t:%.0f:+%.0f,az:%.0f,el:%.0f,roll:%.0f" % (
		t1,t2-t1,az,el,roll) for t1,t2,az,el,roll in
		zip(tmins,tmaxs,azs/utils.degree,els/utils.degree,rolls/utils.degree)])
	return names, itpat

def pinfo2defaults(pinfo):
	dtype    = [(p["name"], "d") for p in pinfo]
	default  = np.zeros(1, dtype)
	for p in pinfo:
		default[p["name"]] = p["default"]
	return default

#def dump_resid(fname, icoord, ocoord, mcoord, dcoord):
#	np.savetxt(fname, np.concatenate([icoord,ocoord,mcoord,dcoord],1)/utils.degree, fmt="%10.3f")

def parse_names(names):
	patch = []
	wafer = []
	band  = []
	obs   = []
	for name in names:
		toks = name.split("_")
		patch.append(toks[0])
		wafer.append(":".join(toks[1:-3]))
		band.append(toks[-3])
		obs.append(toks[-2])
	return bunch.Bunch(patch=np.array(patch), wafer=np.array(wafer),
		band=np.array(band), obs=np.array(obs))

def time_split(ilabels, ctime, maxdur=np.inf):
	names, labels = np.unique(ilabels, return_inverse=True)
	# Split each name-group by maxdur
	allofthem = np.arange(len(names))
	tmins  = ndimage.minimum(ctime, labels, allofthem)
	tmaxs  = ndimage.maximum(ctime, labels, allofthem)
	# avoid division by zero for groups with just one obs in them
	tmaxs  = np.maximum(tmaxs, tmins+1)
	nsplit = utils.floor((tmaxs-tmins)/maxdur)+1
	durs   = (tmaxs-tmins)/nsplit
	tid    = utils.floor((ctime-tmins[labels])/durs[labels])
	# Avoid length-1 group at the end
	tid    = np.minimum(tid, nsplit[labels]-1)
	olabels, nlabel = utils.label_multi([tid, ilabels], return_nlabel=True)
	allofthem = np.arange(nlabel)
	tmins  = ndimage.minimum(ctime, olabels, allofthem)
	tmaxs  = ndimage.maximum(ctime, olabels, allofthem)
	return olabels, tmins, tmaxs

def pattern_split(baz, bel, roll, tol=1*utils.degree):
	labels, nlabel = multi_split([baz,bel,roll],[tol,tol,tol])
	allofthem = np.arange(nlabel)
	oaz, oel, oroll = [ndimage.mean(a, labels, allofthem) for a in [baz,bel,roll]]
	return labels, oaz, oel, oroll

def multi_split(valss, tols):
	labelss = [utils.label_similar_groups_fast(vals, tol) for vals,tol in zip(valss,tols)]
	return utils.label_multi(labelss, return_nlabel=True)

def restore_el(el0, az, el, roll):
	over  = el0 > np.pi/2
	oaz   = utils.rewind(np.where(over, az+np.pi, az))
	oel   = np.where(over, np.pi-el, el)
	oroll = utils.rewind(np.where(over, roll+np.pi, roll))
	return oaz, oel, oroll

fit_dtype = [
	("ctime","d"),("ra","d"),("dec","d"),("snr","d"),("flux","d"),("dflux","d"),
	("Δra","d"), ("σra","d"), ("Δdec","d"), ("σdec","d"),
	("az","d"), ("el","d"),
	("Δaz","d"), ("σaz","d"), ("Δel","d"), ("σel","d"),
	("pat_baz","d"), ("pat_waz","d"), ("pat_bel","d"), ("roll","d"),
	("sid","i"), ("ref_ra","d"), ("ref_dec","d"), ("ref_flux","d"),
	("name","U50"),
]
deg_fields = ["ra", "dec", "az", "el", "pat_baz", "pat_waz", "pat_bel", "roll", "ref_ra", "ref_dec"]
arcmin_fields = ["Δra", "σra", "Δdec", "σdec", "Δaz", "σaz", "Δel", "σel"]

def read_fit_cat(fname):
	cat = np.loadtxt(fname, dtype=fit_dtype).view(np.recarray)
	for field in deg_fields: cat[field] *= utils.degree
	for field in arcmin_fields: cat[field] *= utils.arcmin
	return cat

def calc_resid(pfull, icoord, tcoord, mcoord):
	# Residuals
	resid_hor    = tcoord[:,:2] - mcoord[:,:2]
	resid_xieta  = (azel2xieta_v2(pfull, tcoord, icoord)-azel2xieta_v2(pfull, mcoord, icoord))[:,:2]
	return resid_hor, resid_xieta

def calc_avg_arr(labels, weights, arr):
	rhs = utils.bincount(labels, weights*arr.T)
	div = np.bincount(labels, weights)
	return (rhs/div).T

def calc_avg_struct(labels, weights, struct):
	div  = np.bincount(labels, weights)
	nlab = len(div)
	imax = np.array(ndimage.maximum_position(weights, labels, np.arange(nlab))).reshape(-1,1)[:,0]
	res  = np.zeros(len(div), dtype=struct.dtype)
	for name, subd in struct.dtype.fields.items():
		if np.issubdtype(subd, np.floating):
			# Can use normal averaging
			res[name] = np.bincount(labels, weights*struct[name])/div
		else:
			# Just return the highest-weight entry
			res[name] = struct[name][imax]
	return res

def dump_resid(fname, cat, pfull, resid_hor, resid_xieta):
	n = len(cat)
	# First convert cat back to file units
	cat = cat.copy()
	for field in deg_fields: cat[field] /= utils.degree
	for field in arcmin_fields: cat[field] /= utils.arcmin
	# Residuals
	resid_hor    = resid_hor/utils.arcmin
	resid_xieta  = resid_xieta/utils.arcmin
	# Plain loop for writing
	with open(fname, "w") as ofile:
		for i in range(n):
			ctime, ra, dec, snr, flux, dflux, dra, posacc, ddec, posacc, az, el, daz, posacc, del_, posacc, obs_az, obs_waz, obs_el, obs_roll, refi, ref_ra, ref_dec, ref_flux, base = cat[i]
			res_az, res_el  = resid_hor[i]
			res_xi, res_eta = resid_xieta[i]
			parts = []
			# First reproduce the input catalog. 1-10
			parts.append("%10.0f  %8.3f %7.3f %7.2f %7.1f %7.1f  %6.3f %6.3f %6.3f %6.3f" % (
					ctime, ra, dec, snr, flux, dflux, dra, posacc, ddec, posacc))
			# Part 2. Horizontal version: az, el, Δaz+err, Δel+err. 11-16
			parts.append("  %8.3f %7.3f  %6.3f %6.3f %6.3f %6.3f" % (
					az, el, daz, posacc, del_, posacc))
			# Part 3. Boresight: baz waz bel roll. 17-20
			parts.append("  %7.2f %7.2f %6.2f %7.2f" % (
					obs_az, obs_waz, obs_el, obs_roll))
			# Part 4. Reference: ref_id, ref_ra, ref_dec, ref_flux. 21-24
			parts.append("  %5d %8.3f %7.3f %7.1f" % (
					refi, ref_ra, ref_dec, ref_flux))
			# Part 5: name. 25
			parts.append("  %s" % base)
			# Add the wafer coordinates. These are taken from the fit, but since they're just used
			# as anchor points for offset plots, the details don't matter. 26-27
			parts.append("  %6.3f %6.3f" % (pfull["waf_pos_xi"][i]/utils.degree, pfull["waf_pos_eta"][i]/utils.degree))
			# Then add the residuals. 28-31
			parts.append("  %6.3f %6.3f  %6.3f %6.3f" % (
				res_az, res_el, res_xi, res_eta))
			msg = "".join(parts)
			ofile.write(msg + "\n")

verbosity  = args.verbose - args.quiet
utils.mkdir(args.odir)

# Read our catalog
cat        = read_fit_cat(args.src_fits)
# Set up the v2 model we're starting from
base_model = ModelStaticV2(cat)
base_pfull = base_model.dof_full.read(args.base_model, full=True)

# Recover our input coordinates. We also get a new ocoord because we're
# inferring what the psi angle should be to get the known roll angle
ocoord  = np.array([cat.az, cat.el, cat.roll]).T
dcoord  = np.array([cat.σaz, cat.σel, cat.σaz]).T
icoord, ocoord = base_model.inverse(base_pfull, ocoord)

# Set up the model we want to fit
model   = ModelRadRollArc(cat, defaults=[base_pfull])
#model   = ModelDynamicV2(cat, defaults=[base_pfull])
# Set up the target coordinates we want to match
tcoord  = np.array([cat.az-cat.Δaz, cat.el-cat.Δel, cat.az*0]).T
#x       = model.fit(icoord, tcoord, dcoord, verbose=2, prefix="test")
#x       = model.fit_gibbs(icoord, tcoord, dcoord, verbose=verbosity)
x, dcoord = model.fit_robust(icoord, tcoord, dcoord, verbose=verbosity)
# Write out our final fit parameters
pfull  = model.dof.unzip(x)
model.dof_full.write(os.path.join(args.odir, "params.txt"), pfull, full=True, header="model: %s" % model.name)
# Dump the residuals. For each point we want:
# * All the info in cat
# * resid az,el
# * resid xi,eta
# * wafer xi,eta
mcoord  = model.eval(pfull, icoord)
resid_hor, resid_xieta = calc_resid(pfull, icoord, tcoord, mcoord)
dump_resid(os.path.join(args.odir, "resid_per.txt"), cat, pfull, resid_hor, resid_xieta)
# Average over bins
labels  = utils.label_multi(model.dof.labels[1])
weights = np.sum(dcoord[:,:2]**2,1)**-1
avg_cat   = calc_avg_struct(labels, weights, cat)
avg_pfull = calc_avg_struct(labels, weights, pfull)
# Update uncertainty
posacc    = (np.bincount(labels, weights)/2)**-0.5
for name in ["σra", "Δdec", "σdec", "Δaz", "σaz", "Δel", "σel"]:
	avg_cat[name] = posacc
avg_resid_hor   = calc_avg_arr(labels, weights, resid_hor)
avg_resid_xieta = calc_avg_arr(labels, weights, resid_xieta)
dump_resid(os.path.join(args.odir, "resid_avg.txt"), avg_cat, avg_pfull, avg_resid_hor, avg_resid_xieta)
