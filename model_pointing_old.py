import argparse
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
parser.add_argument(      "--fit-only", type=str, default=None)
args = parser.parse_args()
import numpy as np
from pixell import utils, bunch, coordsys, colors, bench
from scipy import ndimage, optimize

# Differs from model_pointing.py by supporting joint fits across wafers

fit_dtype = [
	("ctime","d"),("ra","d"),("dec","d"),("snr","d"),("flux","d"),("dflux","d"),
	("Δra","d"), ("σra","d"), ("Δdec","d"), ("σdec","d"),
	("az","d"), ("el","d"),
	("Δaz","d"), ("σaz","d"), ("Δel","d"), ("σel","d"),
	("baz","d"), ("waz","d"), ("bel","d"), ("roll","d"),
	("sid","i"), ("ref_ra","d"), ("ref_dec","d"), ("ref_flux","d"),
	("name","U50"),
]
deg_fields = ["ra", "dec", "az", "el", "baz", "waz", "bel", "roll", "ref_ra", "ref_dec"]
arcmin_fields = ["Δra", "σra", "Δdec", "σdec", "Δaz", "σaz", "Δel", "σel"]

def read_fit_cat(fname):
	cat = np.loadtxt(fname, dtype=fit_dtype).view(np.recarray)
	for field in deg_fields: cat[field] *= utils.degree
	for field in arcmin_fields: cat[field] *= utils.arcmin
	return cat

def read_detoff(fname_or_literal):
	try:
		xi, eta = [float(w) for w in fname_or_literal.split(",")]
		return {"*": (xi*utils.degree, eta*utils.degree)}
	except ValueError:
		offs = {}
		with open(fname_or_literal, "r") as ifile:
			for line in ifile:
				toks = line.split()
				if len(toks) == 1: continue
				offs[toks[0]] = (float(toks[1])*utils.degree, float(toks[2])*utils.degree)
		return offs

def name2wafer(name):
	# This works both for wafer and tube grouping
	return ":".join(name.split("_")[1:-3])

def get_detoff(offs, wafer):
	if wafer in offs: return offs[wafer]
	if "*"   in offs: return offs["*"]
	else: raise KeyError("no detoffs for '%s')" % wafer)

#def get_detoff(offs, name):
#	key = name2wafer(name)
#	if key in offs: return offs[key]
#	if "*" in offs: return offs["*"]
#	else: raise KeyError("no detoffs for '%s')" % key)

def cat_wafer_inds(cat):
	# This is a bit inefficient, but straightforward
	cwafers = np.array([name2wafer(name) for name in cat.name])
	wafers, order, edges = utils.find_equal_groups_fast(cwafers)
	labels = np.zeros(len(cat),int)
	for wi in range(len(wafers)):
		labels[order[edges[wi]:edges[wi+1]]] = wi
	return bunch.Bunch(wafers=wafers, order=order, edges=edges, labels=labels)

def add_cat_detoff(cat, offs, winds):
	# Make an output catalog with xi,eta columns and copy over
	dtype = cat.dtype.descr + [("xi","d"),("eta","d")]
	ocat = np.zeros(len(cat), dtype=dtype).view(np.recarray)
	for field in cat.dtype.names:
		ocat[field] = cat[field]
	# then add the offsets
	for gi, wafer in enumerate(winds.wafers):
		inds = winds.order[winds.edges[gi]:winds.edges[gi+1]]
		ocat.xi[inds], ocat.eta[inds] = get_detoff(offs, wafer)
	return ocat

#def add_cat_detoff(cat, offs):
#	dtype = cat.dtype.descr + [("xi","d"),("eta","d")]
#	ocat = np.zeros(len(cat), dtype=dtype).view(np.recarray)
#	for field in cat.dtype.names:
#		ocat[field] = cat[field]
#	# den add the offsets
#	unames, order, edges = utils.find_equal_groups_fast(cat.name)
#	for gi, name in enumerate(unames):
#		inds = order[edges[gi]:edges[gi+1]]
#		ocat.xi[inds], ocat.eta[inds] = get_detoff(offs, name)
#	return ocat

def name2tag(names, group_by="obs"):
	fields = np.array([name.split("_") for name in names]).T
	# Find out which columns we want
	fmap   = {"obs":(-2,), "tube":(-5,), "wafer":(-5,-4), "wtype":(-4,), "band":(-3,)}
	cols   = []
	for fname in group_by.split(","):
		cols += fmap[fname]
	# Extract those
	res = fields[cols[0]]
	for col in cols[1:]:
		res = np.strings.add(np.strings.add(res, "_"), fields[col])
	return res

def split_cat(cat, maxdur=np.inf, group_by="obs"):
	# Group by name
	tags = name2tag(cat.name, group_by=group_by)
	names, labels = np.unique(tags, return_inverse=True)
	# Split each name-group by maxdur
	allofthem = np.arange(len(names))
	tmins  = ndimage.minimum(cat.ctime, labels, allofthem)
	tmaxs  = ndimage.maximum(cat.ctime, labels, allofthem)
	nsplit = utils.floor((tmaxs-tmins)/maxdur)+1
	dts    = (tmaxs-tmins)/nsplit
	tid    = utils.floor((cat.ctime-tmins[labels])/dts[labels])
	# Avoid length-1 group at the end
	tid    = np.minimum(tid, nsplit[labels]-1)
	labels, inds = utils.label_multi([tid, tags], return_index=True)
	# Reformat for output. Useful to have in similar format as
	# find_equal_groups_fast
	ulabels, order, edges = utils.find_equal_groups_fast(labels)
	# Get the start and end time for each of the final groups
	allofthem = np.arange(len(ulabels))
	tmins  = ndimage.minimum(cat.ctime, labels, allofthem)
	tmaxs  = ndimage.maximum(cat.ctime, labels, allofthem)
	return bunch.Bunch(gnames=tags[inds], gsubs=tid[inds], order=order,
		edges=edges, t1s=tmins, t2s=tmaxs)

# Model fitting:
#  * obs_az,  obs_el  = model0(baz, bel, roll) => baz, bel, roll = model0"(obs_az, obs_el)
#  * true_az, true_el = model (baz, bel, roll)
#  * Fit model until it matches true_az, true_el
#  * If we have the tube/wafer reprsentative offsets, then that would enter into model0 and model
#    as constants. Should write it in, but allow them to be zero
#  * Sadly, model is non-linear, so the inverse model0" will itself require a solution process

def restore_el(el0, az, el, roll):
	over  = el0 > np.pi/2
	oaz   = utils.rewind(np.where(over, az+np.pi, az))
	oel   = np.where(over, np.pi-el, el)
	oroll = utils.rewind(np.where(over, roll+np.pi, roll))
	return oaz, oel, oroll

class PointingModel:
	def __init__(self, params, roll=0, det_xieta=None):
		self.params    = params.copy()
		self.roll      = roll
		self.det_xieta = np.array(det_xieta) if det_xieta is not None else np.zeros(2)
	def build_quats(self, baz, bel, roll):
		model = self.params
		corot = bel - roll - 60*utils.degree
		# Apply offsets
		az     = baz   + model.enc_offset_az
		el     = bel   + model.enc_offset_el
		corot  = corot + model.enc_offset_cr
		# El sag
		Δel    = el     - model.el_sag_pivot
		el    += Δel    * model.el_sag_lin
		el    += Δel**2 * model.el_sag_quad
		q_lonlat     = coordsys.Coords(az=az, el=el).q
		q_mir_center = 1/coordsys.rotation_xieta(model.mir_center_xi0, model.mir_center_eta0)
		q_el_roll    = coordsys.euler(2, el - 60*utils.degree)
		q_el_axis_center = 1/coordsys.rotation_xieta(model.el_axis_center_xi0, model.el_axis_center_eta0)
		q_cr_roll    = coordsys.euler(2, -corot)
		q_cr_center  = 1/coordsys.rotation_xieta(model.cr_center_xi0, model.cr_center_eta0)
		q_middle     = q_mir_center * q_el_roll * q_el_axis_center * q_cr_roll * q_cr_center
		# Base tilt
		phi = np.arctan2(model.base_tilt_sin, model.base_tilt_cos)
		amp = (model.base_tilt_sin**2 + model.base_tilt_cos**2)**0.5
		q_base = coordsys.euler(2,phi) * coordsys.euler(1, amp) * coordsys.euler(2, -phi)
		# Detectors
		q_det  = coordsys.rotation_xieta(*self.det_xieta)
		return q_base, q_lonlat, q_middle, q_det
	def apply(self, bazel):
		q_base, q_lonlat, q_middle, q_det = self.build_quats(bazel[0], bazel[1], self.roll)
		coords = coordsys.Coords(q=q_base * q_lonlat * q_middle * q_det)
		return np.array(restore_el(bazel[1], coords.az, coords.el, coords.roll)[:2])
	def _approx_inverse(self, azel, bazel_guess, oroll_guess):
		# Our approximate inverse assumes it knows baz, bel, making the whole
		# operation linear. It then applies this operation to the actual azel.
		# The result will be the next bazel guess
		q_base, q_lonlat, q_middle, q_det = self.build_quats(bazel_guess[0], bazel_guess[1], self.roll)
		# Reconstruct q_lonlat
		q_tot = coordsys.Coords(az=azel[0], el=azel[1], roll=oroll_guess).q
		q_lonlat = 1/q_base * q_tot / q_det / q_middle
		c_lonlat = coordsys.Coords(q=q_lonlat)
		baz, bel, broll = restore_el(bazel_guess[1], c_lonlat.az, c_lonlat.el, c_lonlat.roll)
		# Quaternion part done. Do the inverse sag. Can solve this part exactly, but
		# more consistent with the rest to just use bazel_guess
		Δel  = (bazel_guess[1]+self.params.enc_offset_el) - self.params.el_sag_pivot
		bel -= Δel*self.params.el_sag_lin + Δel**2 * self.params.el_sag_quad
		baz -= self.params.enc_offset_az
		bel -= self.params.enc_offset_el
		return np.array([baz, bel])
	def inverse(self, azel, niter=10, return_oroll=False):
		bazel = azel
		oroll = self.roll
		for it in range(niter):
			bazel  = self._approx_inverse(azel, bazel, oroll)
			# Get a new oroll guess
			q_base, q_lonlat, q_middle, q_det = self.build_quats(bazel[0], bazel[1], self.roll)
			oroll = coordsys.Coords(q=q_base * q_lonlat * q_middle * q_det).roll
			oaz, oel, oroll = restore_el(azel[1], bazel[0], bazel[1], oroll)
		if return_oroll: return bazel, oroll
		else: return bazel
	def copy(self): return PointingModel(params=self.params, roll=self.roll, det_xieta=self.det_xieta)
	def update(self, dict):
		self.params.update(dict)
		return self
	# The ones below here are for exploring coordinate systems where the pointing residuals
	# could be more naturally explained
	def azel2xieta(self, azel, bazel, oroll):
		q_base, q_lonlat, q_middle, q_det = self.build_quats(bazel[0], bazel[1], self.roll)
		q_tot = coordsys.Coords(az=azel[0], el=azel[1], roll=oroll).q
		# q_tot = q_base * q_lonlat * q_middle * q_det. Want to infer q_det instead of using the
		# fiducial one
		q_odet = (1/q_middle) * (1/q_lonlat) * (1/q_base) * q_tot
		return np.array(coordsys.decompose_xieta(q_odet))
	def azel2postroll(self, azel, bazel, oroll):
		q_base, q_lonlat, q_middle, q_det = self.build_quats(bazel[0], bazel[1], self.roll)
		q_tot  = coordsys.Coords(az=azel[0], el=azel[1], roll=oroll).q
		q_post = (1/q_lonlat) * (1/q_base) * q_tot
		return np.array(coordsys.decompose_xieta(q_post))

def debug(cat, base, det_xieta=None):
	baz, bel, roll = utils.rewind(np.array([190, 120, 185])*utils.degree)
	oroll = (3.269316324510058-180)*utils.degree
	#baz, bel, roll = np.array([10, 60, 5])*utils.degree
	model = PointingModel(base, roll=roll, det_xieta=(cat.xi, cat.eta))
	az, el = model.apply([baz, bel])
	#baz2, bel2 = model.inverse([az,el])
	baz2, bel2 = model._approx_inverse([az,el], [baz,bel], oroll)
	print("%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (baz/utils.degree, baz2/utils.degree, (baz2-baz)/utils.degree, bel/utils.degree, bel2/utils.degree, (bel2-bel)/utils.degree))
	baz3, bel3 = model.inverse([az,el])
	print("%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" % (baz/utils.degree, baz3/utils.degree, (baz3-baz)/utils.degree, bel/utils.degree, bel3/utils.degree, (bel3-bel)/utils.degree))

# Uncertainty estimate. If we assume gaussian, then we have:
#  chisq    = (d-Pa)'N"(d-Pa), so
#  chisq_,ij = (P'N"P)_ij. For comparison, the cov(â) = (P'N"P)", so
#  chisq_,ij = icov(â)_ij

def _fit_model(cat, base, params=None, scale=1e-4, errs=None, estimate_dx=True, xieta_offset=None, xoff=0):
	if params is None: params = ["enc_offset_az", "enc_offset_el", "base_tilt_cos", "base_tilt_sin"]
	azel_obs  = np.array([cat.az,  cat.el])
	Δazel     = np.array([cat.Δaz, cat.Δel])
	σazel     = np.array([cat.σaz, cat.σel]) if errs is None else errs
	azel_true = azel_obs - Δazel
	# Unapply current model
	model0    = PointingModel(base, roll=cat.roll, det_xieta=(cat.xi, cat.eta))
	bazel, oroll = model0.inverse(azel_obs, return_oroll=True)
	# Now that we've calculated bazel and oroll, we can safely apply any xieta offset.
	# If we did it before this, it would mostly cancel instead of representing a pointing
	# error term
	if xieta_offset is not None:
		model0.det_xieta[0] += xieta_offset[0]
		model0.det_xieta[1] += xieta_offset[1]
	# Solve for the model parametrs that makes model.apply(bazel) match azel_true
	def zip(model): return np.array([model.params[name] for name in params])/scale
	def unzip(x): return model0.copy().update({name:x[i]*scale for i,name in enumerate(params)})
	def calc_chisq(x):
		model = unzip(x)
		azel  = model.apply(bazel)
		resid = azel-azel_true
		chisq = np.sum((resid/σazel)**2)
		#print("%15.7e %15.7e %15.7e %15.7e  %15.7e" % (x[0],x[1],x[2],x[3],chisq))
		return chisq
	def calc_ddchisq(x, δ):
		# Calculate the diagonal of the hessian of calc_chisq, using
		# step size δ, which should be small compared to the uncertainty,
		# but not too small. It's in the same, scaled units as x
		# Uses 2*ndof+1 function evaluations (so typically 9), which is much
		# less than the more robust and accurate scipy.differentiate.hessian uses,
		# which would tpyically be thousands
		ddchisq = x*0
		δ       = x*0+δ
		chisq   = calc_chisq(x)
		for i, δval in enumerate(δ):
			u = utils.uvec(len(x), i)
			v1 = calc_chisq(x+u*δval)
			v2 = calc_chisq(x-u*δval)
			ddchisq[i] = (v1 - 2*chisq + v2)/δval**2
		return ddchisq
	x0    = zip(model0)
	if x0.size > 0: x = optimize.fmin_powell(calc_chisq, x0, disp=False)
	else:           x = x0.copy()
	# debug: artificial x offset
	x    += xoff/scale
	model = unzip(x)
	azel  = model.apply(bazel)
	resid = azel-azel_true
	chisq = calc_chisq(x)
	# Get offsets in focal plane coordinates too
	Δxieta = model.azel2xieta(azel, bazel, oroll)[:2] - model.azel2xieta(azel_true, bazel, oroll)[:2]
	#Δpost  = model.azel2postroll(azel, bazel, oroll)[:2] - model.azel2postroll(azel_true, bazel, oroll)[:2]
	# Get a weighted average quantities
	weight= np.sum(σazel**-2,0)
	def avg(a): return np.sum(a*weight)/np.sum(weight)
	avg_t    = avg(cat.ctime)
	avg_az   = avg(utils.rewind(cat.az))
	avg_el   = avg(cat.el)
	avg_roll = avg(cat.roll)
	# Value and uncertainty for the parts of the model we fit. The
	# values are redundant, since they're already inside the model we return
	xout  = x*scale
	if estimate_dx:
		ivar  = calc_ddchisq(x, 0.01*utils.arcmin/scale)/scale**2
		dxout = ivar**-0.5
	else: dxout = x*0
	res   = bunch.Bunch(model=model, resid=resid, chisq=chisq, n=len(cat), x=xout, dx=dxout,
		params=params, t=avg_t, az=avg_az, el=avg_el, roll=avg_roll, Δxieta=Δxieta)#, Δpost=Δpost)
	return res

def fit_model(cat, base, params=None, scale=1e-4,
		errs=[0.3*utils.arcmin,0.1*utils.arcmin,0], etol=3, xieta_offset=None, xoff=0, robust=False):
	if not robust:
		return _fit_model(cat, base, params=params, scale=scale, errs=None, estimate_dx=True, xieta_offset=xieta_offset, xoff=xoff)
	else:
		# First do a fit with bounded accuracy
		σazel   = np.array([cat.σaz, cat.σel])
		penalty = np.zeros(len(cat))
		for i, err in enumerate(errs):
			last  = i==len(errs)-1
			σwork = (σazel**2 + err**2 + penalty)**0.5
			fit   = _fit_model(cat, base, params=params, scale=scale, errs=σwork, estimate_dx=last, xieta_offset=xieta_offset, xoff=xoff)
			# Which have poor fits?
			excess= np.mean(fit.resid**2,0) - np.mean(σwork**2,0)
			penalty += np.maximum(0, excess - etol**2*np.mean(σazel**2,0))
		return fit

def wafer_avg(labels, vals, minlength=0):
	weight = (cat.σaz**2+cat.σel**2)**-1
	avgs   = utils.bincount(labels, vals*weight, minlength=minlength)
	div    = np.bincount(labels, weight, minlength=minlength)
	with utils.nowarn():
		avgs /= div
		utils.remove_nan(avgs)
	return avgs

def fit_det_offs(ginfo, cat, base, winds, params=None, scale=1e-4,
		errs=[0.3*utils.arcmin,0.1*utils.arcmin,0], etol=3, niter=2, robust=False, verbose=False):
	nwafer = len(winds.wafers)
	xieta_offset = np.zeros((2,nwafer))
	for it in range(niter):
		# Fit each group
		Δxieta = np.zeros((2,len(cat)))
		for gi, (gname, gsub) in enumerate(zip(ginfo.gnames, ginfo.gsubs)):
			name  = "%s_%02d" % (gname, gsub)
			inds  = ginfo.order[ginfo.edges[gi]:ginfo.edges[gi+1]]
			gcat  = cat[inds]
			nsrc  = len(gcat)
			t1, t2= ginfo.t1s[gi], ginfo.t2s[gi]
			if nsrc < args.nmin: continue
			# offsets[:,winds.labels] looks up the wafer offsets for each entry in cat
			fit = fit_model(gcat, base, params=params, robust=robust, xieta_offset=xieta_offset[:,winds.labels[inds]])
			Δxieta[:,inds] = fit.Δxieta
			if verbose:
				msg = format_fit(fit, name, [t1,t2])
				print("%d/%d %5d/%d %s" % (it+1, niter, gi+1, len(ginfo.gnames), msg))
		# We now have the individual offsets. Average per wafer
		xieta_offset += wafer_avg(winds.labels, -Δxieta, minlength=nwafer)
	return xieta_offset

#def wafer_avg(labels, vals):
#	weight = (cat.σaz**2+cat.σel**2)**-1
#	avgs   = utils.bincount(labels, vals*weight)/np.bincount(labels, weight)
#	return avgs[:,labels]
#
#def fit_det_offs(ginfo, cat, base, params=None, scale=1e-4,
#		errs=[0.3*utils.arcmin,0.1*utils.arcmin,0], etol=3, niter=2, robust=False, verbose=False):
#	xieta_offset = np.zeros((2,len(cat)))
#	wlabels = utils.label_multi([cat.xi, cat.eta])
#	for it in range(niter):
#		# Fit each group
#		Δxieta = np.zeros((2,len(cat)))
#		for gi, (gname, gsub) in enumerate(zip(ginfo.gnames, ginfo.gsubs)):
#			name  = "%s_%02d" % (gname, gsub)
#			inds  = ginfo.order[ginfo.edges[gi]:ginfo.edges[gi+1]]
#			gcat  = cat[inds]
#			nsrc  = len(gcat)
#			t1, t2= ginfo.t1s[gi], ginfo.t2s[gi]
#			if nsrc < args.nmin: continue
#			fit = fit_model(gcat, base, params=params, robust=robust, xieta_offset=xieta_offset[:,inds])
#			Δxieta[:,inds] = fit.Δxieta
#			if verbose:
#				msg = format_fit(fit, name, [t1,t2])
#				print("%d/%d %5d/%d %s" % (it+1, niter, gi+1, len(ginfo.gnames), msg))
#		# We now have the individual offsets. Average per wafer
#		xieta_offset += wafer_avg(wlabels, -Δxieta)
#	return xieta_offset

# Dumps the residual per individual observation
def dump_resid(fname, cat, fit):
	out = np.array([
		cat.ctime, cat.az/utils.degree, cat.el/utils.degree, cat.roll/utils.degree, cat.snr,
		cat.Δaz/utils.arcmin, cat.σaz/utils.arcmin,
		cat.Δel/utils.arcmin, cat.σel/utils.arcmin,
		cat.Δra/utils.arcmin, cat.σra/utils.arcmin,
		cat.Δdec/utils.arcmin,cat.σdec/utils.arcmin,
		cat.sid, cat.ra/utils.degree, cat.dec/utils.degree,
		cat.ref_flux, cat.flux,
		fit.resid[0]/utils.arcmin, fit.resid[1]/utils.arcmin]).T
	np.savetxt(fname, out, fmt="%10.0f  %7.2f %6.2f %7.2f %7.2f  %6.3f %6.3f %6.3f %6.3f  %6.3f %6.3f %6.3f %6.3f  %5d %7.2f %7.2f %7.1f %7.1f  %6.3f %6.3f")

# Dump the average offset and residual per entry in the
# focal plane
def dump_fplane(fname, cat, fit):
	# First group by xieta
	groups = utils.find_equal_groups(np.array([cat.xi, cat.eta]).T)
	with open(fname, "w") as ofile:
		# Will output one line in the file for each of these groups.
		for gi, group in enumerate(groups):
			gcat    = cat[group]
			# If this is too vulnerable to outliers, then should increase error bar
			# at some earlier point, not in this function
			weight  = (gcat.σaz**2+gcat.σel**2)**-1
			wtot    = np.sum(weight)
			σ       = wtot**-0.5
			Δaz0    = np.sum(gcat.Δaz*weight)/wtot
			Δel0    = np.sum(gcat.Δel*weight)/wtot
			Δaz,Δel = np.sum(fit.resid[:,group]*weight,-1)/wtot
			ctime   = np.sum(gcat.ctime*weight)/wtot
			baz     = np.sum(gcat.baz*weight)/wtot
			bel     = np.sum(gcat.bel*weight)/wtot
			roll    = np.sum(gcat.roll*weight)/wtot
			Δxi,Δeta= np.sum(fit.Δxieta[:,group]*weight,-1)/wtot
			#Δpost   = np.sum(fit.Δpost[:,group]*weight,-1)/wtot
			msg     = "%6.3f %6.3f %6.3f %6.3f  %6.3f %6.3f  %6.3f %6.3f  %6.3f %6.3f   %10.0f %8.2f %8.2f %8.2f" % ( # %6.3f %6.3f" % (
				Δaz0/utils.arcmin, σ/utils.arcmin,
				Δel0/utils.arcmin, σ/utils.arcmin,
				Δaz/utils.arcmin, Δel/utils.arcmin,
				gcat.xi[0]/utils.degree, gcat.eta[0]/utils.degree,
				Δxi/utils.arcmin, Δeta/utils.arcmin,
				ctime, baz/utils.degree, bel/utils.degree, roll/utils.degree,
				#Δpost[0]/utils.arcmin, Δpost[1]/utils.arcmin,
				)
			ofile.write(msg + "\n")

param_order = ["enc_offset_az", "enc_offset_el", "enc_offset_cr", "base_tilt_sin", "base_tilt_cos", "el_axis_center_xi0", "el_axis_center_eta0", "mir_center_xi0", "mir_center_eta0", "cr_center_xi0", "cr_center_eta0", "el_sag_pivot", "el_sag_lin", "el_sag_quad"]

def format_fit(fit, name, trange):
	desc   = "%s %10.0f %10.0f %10.0f %6.1f %6.1f %6.1f %3d %6.3f : " % (
		name, fit.t, trange[0], trange[1], fit.az/utils.degree, fit.el/utils.degree, fit.roll/utils.degree,
		fit.n, fit.chisq/fit.n)
	# The fit parameters
	for pname in param_order:
		v = fit.model.params[pname]
		try: σ = fit.dx[fit.params.index(pname)]
		except ValueError: σ = 0
		desc += " %9.6f %8.6f" % (v/utils.degree, σ/utils.degree)
	return desc

def format_header():
	header = "# %s %-10s %-10s %-10s %-6s %-6s %-6s %-3s %-6s : " % ("name", "tavg", "t1", "t2", "az", "el", "roll", "n", "chisq")
	for param in param_order:
		header += " %s err_%s" % (param, param)
	header += "\n"
	return header

def fit_only_hack(cat, fit_only):
	sel = np.full(len(cat), False)
	for tok in fit_only.split(","):
		sel |= np.char.count(cat.name, tok) > 0
	cat.σaz[sel] *= 1e-2
	cat.σel[sel] *= 1e-2

params = args.fit.split(",") if args.fit != "none" else []
cat    = read_fit_cat(args.src_fits)
if args.fit_only: fit_only_hack(cat, args.fit_only)
# add xieta wafer offsets per entry
winds  = cat_wafer_inds(cat)
detoff = read_detoff(args.detoff)
cat    = add_cat_detoff(cat, detoff, winds)

base  = bunch.read(args.base_model)
ginfo = split_cat(cat, maxdur=args.maxdur*utils.hour, group_by=args.group)

utils.mkdir(args.odir)

if args.fit_offs:
	xieta_offset = fit_det_offs(ginfo, cat, base, winds, params=params, robust=args.robust, verbose=True)
	with open(args.odir + "/wafer_offs.txt", "w") as ofile:
		for wafer, (xi,eta) in zip(winds.wafers, xieta_offset.T):
			ofile.write("%-6s %8.5f %8.5f\n" % (wafer, xi/utils.arcmin, eta/utils.arcmin))
else: xieta_offset = np.zeros((2,len(cat)))

xoff = np.zeros(len(params))
#xoff[0] += 1*utils.arcmin

with open(args.odir + "/model.txt", "w") as ofile:
	ofile.write(format_header())
	for gi, (gname, gsub) in enumerate(zip(ginfo.gnames, ginfo.gsubs)):
		name  = "%s_%02d" % (gname, gsub)
		inds  = ginfo.order[ginfo.edges[gi]:ginfo.edges[gi+1]]
		gcat  = cat[inds]
		nsrc  = len(gcat)
		t1, t2= ginfo.t1s[gi], ginfo.t2s[gi]
		if nsrc < args.nmin:
			print("%5d/%d %s skip:nsrc=%d" % (gi+1, len(ginfo.gnames), name, nsrc))
			continue
		fit = fit_model(gcat, base, params=params, robust=args.robust,
			xieta_offset=xieta_offset[:,winds.labels[inds]], xoff=xoff)
		dump_resid("%s/resid_%s.txt" % (args.odir, name), gcat, fit)
		dump_fplane("%s/fplane_%s.txt" % (args.odir, name), gcat, fit)
		msg = format_fit(fit, name, [t1,t2])
		print("%5d/%d %s" % (gi+1, len(ginfo.gnames), msg))
		ofile.write(msg + "\n")
