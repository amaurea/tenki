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
args = parser.parse_args()
import numpy as np
from pixell import utils, bunch, coordsys, colors, bench
from scipy import ndimage, optimize

#fit_dtype = [
#	("ctime","d"),("az","d"),("el","d"),("roll","d"),("snr","d"),
#	("Δaz","d"), ("σaz","d"), ("Δel","d"), ("σel","d"),
#	("Δra","d"), ("σra","d"), ("Δdec","d"), ("σdec","d"),
#	("sid","i"), ("ra","d"), ("dec", "d"), ("ref_flux","d"),
#	("flux","d"), ("name","U50"),
#]
#deg_fields = ["az", "el", "roll", "ra", "dec"]
#arcmin_fields = ["Δaz", "σaz", "Δel", "σel", "Δra", "σra", "Δdec", "σdec"]

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

def get_detoff(offs, name):
	key = ":".join(name.split("_")[1:-4])
	if key in offs: return offs[key]
	if "*" in offs: return offs["*"]
	else: raise KeyError("no detoffs for '%s')" % key)

def split_cat(cat, maxdur=np.inf):
	# Group by name
	names, labels = np.unique(cat.name, return_inverse=True)
	# Split each name-group by maxdur
	allofthem = np.arange(len(names))
	tmins  = ndimage.minimum(cat.ctime, labels, allofthem)
	tmaxs  = ndimage.maximum(cat.ctime, labels, allofthem)
	nsplit = utils.floor((tmaxs-tmins)/maxdur)+1
	dts    = (tmaxs-tmins)/nsplit
	tid    = utils.floor((cat.ctime-tmins[labels])/dts[labels])
	# Avoid length-1 group at the end
	tid    = np.minimum(tid, nsplit[labels]-1)
	labels, inds = utils.label_multi([tid, cat.name], return_index=True)
	# Reformat for output. Useful to have in similar format as
	# find_equal_groups_fast
	_, order, edges = utils.find_equal_groups_fast(labels)
	# Get the start and end time for each of the final groups
	allofthem = np.arange(len(labels))
	tmins  = ndimage.minimum(cat.ctime, labels, allofthem)
	tmaxs  = ndimage.maximum(cat.ctime, labels, allofthem)
	return bunch.Bunch(gnames=cat.name[inds], gsubs=tid[inds], order=order,
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
		self.det_xieta = det_xieta if det_xieta is not None else (0,0)
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
	def inverse(self, azel, niter=10):
		# FIXME: apply and inverse must handle el>90°. If not, we end up with crazy fits
		bazel = azel
		oroll = self.roll
		for it in range(niter):
			bazel  = self._approx_inverse(azel, bazel, oroll)
			# Get a new oroll guess
			q_base, q_lonlat, q_middle, q_det = self.build_quats(bazel[0], bazel[1], self.roll)
			oroll = coordsys.Coords(q=q_base * q_lonlat * q_middle * q_det).roll
			oaz, oel, oroll = restore_el(azel[1], bazel[0], bazel[1], oroll)
		return bazel
	def copy(self): return PointingModel(params=self.params, roll=self.roll, det_xieta=self.det_xieta)
	def update(self, dict):
		self.params.update(dict)
		return self

def debug(cat, base, det_xieta=None):
	baz, bel, roll = utils.rewind(np.array([190, 120, 185])*utils.degree)
	oroll = (3.269316324510058-180)*utils.degree
	#baz, bel, roll = np.array([10, 60, 5])*utils.degree
	model = PointingModel(base, roll=roll, det_xieta=det_xieta)
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

def fit_model(cat, base, params=None, scale=1e-4, errs=None, estimate_dx=True, det_xieta=None):
	if params is None: params = ["enc_offset_az", "enc_offset_el", "base_tilt_cos", "base_tilt_sin"]
	azel_obs  = np.array([cat.az,  cat.el])
	Δazel     = np.array([cat.Δaz, cat.Δel])
	σazel     = np.array([cat.σaz, cat.σel]) if errs is None else errs
	azel_true = azel_obs - Δazel
	# Unapply current model
	model0    = PointingModel(base, roll=cat.roll, det_xieta=det_xieta)
	bazel     = model0.inverse(azel_obs)
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
	x     = optimize.fmin_powell(calc_chisq, x0, disp=False)
	model = unzip(x)
	azel  = model.apply(bazel)
	resid = azel-azel_true
	chisq = calc_chisq(x)
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
		params=params, t=avg_t, az=avg_az, el=avg_el, roll=avg_roll)
	return res

def fit_model_robust(cat, base, params=None, scale=1e-4,
		errs=[0.3*utils.arcmin,0.1*utils.arcmin,0], etol=3, det_xieta=None):
	# First do a fit with bounded accuracy
	σazel   = np.array([cat.σaz, cat.σel])
	penalty = np.zeros(len(cat))
	for i, err in enumerate(errs):
		last  = i==len(errs)-1
		σwork = (σazel**2 + err**2 + penalty)**0.5
		fit   = fit_model(cat, base, params=params, scale=scale, errs=σwork, estimate_dx=last, det_xieta=det_xieta)
		# Which have poor fits?
		excess= np.mean(fit.resid**2,0) - np.mean(σwork**2,0)
		penalty += np.maximum(0, excess - etol**2*np.mean(σazel**2,0))
	return fit

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

params= args.fit.split(",")
icat = read_fit_cat(args.src_fits)
base = bunch.read(args.base_model)
ginfo = split_cat(icat, maxdur=args.maxdur*utils.hour)

# This should be per-tube/per-wafer, not just an overall number
detoff = read_detoff(args.detoff)
utils.mkdir(args.odir)

# TODO: The pointing model has an el-sag term that requires the
# full, 0-180° range of el, but we don't have that available in our
# input fits. Should fix that.

# TODO: check if wafer offsets matter. They're currently ignored, but
# maybe they're responsible for the wafer inconsistency I see in the
# fits.

with open(args.odir + "/model.txt", "w") as ofile:
	ofile.write(format_header())
	for gi, (gname, gsub) in enumerate(zip(ginfo.gnames, ginfo.gsubs)):
		name  = "%s_%02d" % (gname, gsub)
		inds  = ginfo.order[ginfo.edges[gi]:ginfo.edges[gi+1]]
		gcat  = icat[inds]
		nsrc  = len(gcat)
		t1, t2= ginfo.t1s[gi], ginfo.t2s[gi]
		try:
			det_xieta = get_detoff(detoff, name)[::-1]
		except KeyError as e:
			print("%5d/%d %s skip:%s" % (gi+1, len(ginfo.gnames), name, str(e)))
			continue
		if nsrc < args.nmin:
			print("%5d/%d %s skip:nsrc=%d" % (gi+1, len(ginfo.gnames), name, nsrc))
			continue
		if args.robust: fit = fit_model_robust(gcat, base, params=params, det_xieta=det_xieta)
		else:           fit = fit_model(gcat, base, params=params, det_xieta=det_xieta)
		dump_resid("%s/resid_%s.txt" % (args.odir, name), gcat, fit)
		msg = format_fit(fit, name, [t1,t2])
		print("%5d/%d %s" % (gi+1, len(ginfo.gnames), msg))
		ofile.write(msg + "\n")
