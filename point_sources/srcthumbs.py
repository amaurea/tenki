from __future__ import division, print_function
import sys
help_general = """Usage:
  srcthumbs build srcs sel odir

    For each contiguous scan (typically one hour), build a data file containing
    thumbnail maps and ivars for each source that it hits, along with the time
    each source is hit and how the focalplane x,y axes map to the sky locally.
    Also builds a fourier-space noise model for the thumbs. Different arrays
    and frequencies are analysed separately. The source id and flux is also
    recorded, along with its fiducial pixel coordinates in its thumbnail, and
    the beam.

  srcthumbs analyse ifiles odir
    Reads in the files produced by srcthumbs build and fits the flux and
    position offset for each point source. Different freqs of the same array
    can be analysed jointly. An observing period can also be split into sub-periods
    (e.g. 10 minutes, like a TOD) for higher time resolution at the cost of S/N.
    Outputs these in text format.

  srcthumbs model gfit_pos odir
    Split the data points output from "analyse" into segments with the same scanning patterns and
    not too large gaps in time. Remove any outliers, and model the points as a slowly changing
    function in time and a gradient in azimuth.

  srcthumbs remodel imodel gfits_pos odir
    Read in a model and some new data points, and spit out a new model that's the
    old model, but modified by those data points. The use case for this is when we have
    an array with only low-S/N measurements and so can't build a decent model on our own,
    and instead base it on the model from a better array.

  srcthumbs calfile sel modelfiles odir
    Given the model files and a tod selector, output the new pointing model

"""

import numpy as np
from pixell import utils

if len(sys.argv) < 2:
	sys.stderr.write(help_general + "\n")
	sys.exit(1)
mode = sys.argv[1]

def build_groups(vals, tols):
	"""Split into groups with edges vals[n,m] change by more than tols[n]"""
	vals, tols = np.asarray(vals).T, np.asarray(tols)
	diffs = vals[1:]-vals[:-1]
	jumps = np.where(np.any(np.abs(diffs)>tols,1))[0]
	edges = np.concatenate([[0],jumps+1,[len(data)]])
	groups= np.array([edges[:-1],edges[1:]]).T
	return groups

def stepfuns(vals, inds, n):
	vals  = np.asarray(vals)
	dvals = vals[1:]-vals[:-1]
	work  = np.zeros(n, vals.dtype)
	work[inds[0]]  = vals[0]
	work[inds[1:]] = dvals
	return np.cumsum(work)

def split_long_groups(groups, t, maxdur=5000):
	"""Split long groups so that no groups are longer than maxdur.
	The implementation may look weird, but it avoids looping in python."""
	edges = groups[:,0]
	# Make time count from beginning
	t     = t-t[0]
	# Figure out how many pieces to split each group into
	durs  = np.maximum(1,np.concatenate([t[edges[1:]]-t[edges[:-1]-1], [t[-1]-t[edges[-1]]]]))
	nsplit= np.maximum(1,np.ceil(durs/maxdur).astype(int))
	# Make the times relative to each segment
	trel  = t-stepfuns(t[edges], edges, t.size)
	# Scale each into split units
	srel  = trel * stepfuns(nsplit/durs, edges, t.size)
	# Make counts relative to each sub-split. The new edges are where
	# the resulting function falls.
	srel %= 1
	mask  = np.concatenate([[True],srel[1:]-srel[:-1]<0])
	# The prodcedure above loses len-1 groups. Handle by forcing an edge at orig edge pos
	mask[edges] = True
	oedges  = np.concatenate([np.where(mask)[0],[t.size]])
	ogroups = np.array([oedges[:-1],oedges[1:]]).T
	return ogroups

def classify_outliers(offs, doffs, nsigma=10, nmedian=5):
	offs = np.array(offs)
	doffs= np.array(doffs)
	# Median filter each offs
	for i, off in enumerate(offs):
		offs[i] -= ndimage.median_filter(off, nmedian, mode="reflect")
		# Estimate uncertainty of median-subtracted quantity
		dref = ndimage.median_filter(doffs[i], nmedian, mode="reflect")
		dmed = dref * (np.pi*(2*nmedian+1)/(4*nmedian**2))**0.5
		doffs[i] = (doffs[i]**2 + dmed**2)**0.5
	bad = np.any(np.abs(offs) > nsigma*doffs,0)
	# If everything is bad, then make the best with what we have
	if np.all(bad): bad[:] = False
	return bad

def fit_model_poly(pos, dpos, t, az):
	# Make time and az easier to work with
	az  = utils.unwind(az, 360)
	t0  = np.mean(utils.minmax(t))
	az0 = np.mean(utils.minmax(az))
	dt  = t-t0
	daz = az-az0
	# Build the basis
	B_full = np.array([daz*0+1, daz, dt, dt**2, dt**3]) # [na,:]
	n_full = len(B_full)
	# Decide how many degrees of freedom we can use in the fit
	dur    = t[-1]-t[0]
	npoint = len(t)
	ndof   = min(npoint, 2+(dur>500)+(dur>3000)+(dur>7000))
	B      = B_full[:ndof]
	rhs = np.einsum("ai,yi->ya", B, pos/dpos**2) # [{y,x},na]
	iA  = np.einsum("ai,yi,bi->yab", B, 1/dpos**2, B) # [{y,x},na,na]
	a   = np.linalg.solve(iA, rhs) # [{y,x},na]
	da  = np.einsum("yaa->ya",np.linalg.inv(iA))**0.5
	# Pad back to the full shape
	def expand(v, n):
		res = np.zeros((v.shape[0],)+(n,)*(v.ndim-1),v.dtype)
		res[(slice(None),)+tuple([slice(0,m) for m in v.shape[1:]])] = v
		return res
	a_full  = expand( a, n_full)
	da_full = expand(da, n_full)
	iA_full = expand(iA, n_full)
	# Get the model, residual and chisq
	model = np.einsum("ai,ya->yi", B, a)
	resid = pos - model
	chisq = np.sum((model-resid)**2)
	return bunch.Bunch(pos=pos, dpos=dpos, t0=t0, az0=az0, dt=dt, daz=daz, ndof=ndof,
			B=B_full, a=a_full, da=da_full, iA=iA_full, model=model, resid=resid, chisq=chisq, neff=len(t)-ndof)

def fit_model_piecewise(pos, dpos, t, az, az0=None, ndof="auto", dof_penalty=500, azslope=True):
	# Fit a series of linear segments that share the same az gradient.
	nsamp    = len(t)
	nt       = len(np.unique(t))
	if ndof == "auto":
		if nt == 1:
			return fit_model_piecewise(pos, dpos, t, az, ndof=1, azslope=False)
		else:
			dur  = max(t[-1]-t[0],300)
			nmax = min(nt-1, 2+min(utils.nint(dur/800), nt//4))
			fits = [fit_model_piecewise(pos, dpos, t, az, az0=az0, ndof=i, azslope=azslope) for i in range(1, nmax+1) for azslope in [False,True]]
			#if t[0] > 1555412544.0:
			#	print([fit.chisq/fit.neff for fit in fits])
			# TODO: finish ndof penalty
			best = np.argmin([fit.chisq/fit.neff+(i/dur)*dof_penalty for i, fit in enumerate(fits)])
			return fits[best]
	ndof     = min(ndof, nt)
	t_range  = utils.minmax(t);  t0  = np.min(t_range);  dt  = t-t0
	if az0 is None: az0 = np.mean(utils.minmax(az))
	# Disable az slope if we only have a single degree of freedom, since we need that for the mean level
	if ndof == 1: azslope = False
	daz = az- az0
	# Build our basis
	B      = np.zeros([ndof,nsamp])
	npoint = ndof-azslope
	if npoint == 1:
		B[0]   = t*0+1
		tedges = t[0:1]
	else:
		## equi-spaced by samples
		#tedges = t[np.arange(nsamp-1)//(ndof-1)]
		# equi-spaced in time
		tedges  = np.linspace(t[0], t[-1], npoint)
		# Make sure there's at least one point between each point
		tinds   = np.searchsorted(t, tedges)
		_, good = np.unique(tinds, return_index=True)
		tedges  = tedges[good]
		ndof    = len(tedges)+azslope
		B       = B[:ndof]
		# And fill
		tranges = np.array([tedges[:-1],tedges[1:]])
		durs    = np.concatenate([tranges[1]-tranges[0]])
		# Falling parts
		B[0:ndof-1-azslope]  = (1-(t-tranges[0,:,None])/durs[:,None]) * ((t >= tranges[0,:,None])&(t <= tranges[1,:,None]))
		# Rising parts
		B[1:ndof-0-azslope] += (0+(t-tranges[0,:,None])/durs[:,None]) * ((t >= tranges[0,:,None])&(t <= tranges[1,:,None]))
	if azslope:
		B[-1] = daz
	#print(B)
	# And fit
	rhs = np.einsum("ai,yi->ya", B, pos/dpos**2) # [{y,x},na]
	iA  = np.einsum("ai,yi,bi->yab", B, 1/dpos**2, B) # [{y,x},na,na]
	a   = np.linalg.solve(iA, rhs) # [{y,x},na]
	da  = np.einsum("yaa->ya",np.linalg.inv(iA))**0.5
	# Get the model, residual and chisq
	model = np.einsum("ai,ya->yi", B, a)
	resid = pos - model
	chisq = np.sum((resid/dpos)**2)
	return bunch.Bunch(pos=pos, dpos=dpos, t0=t0, az0=az0, dt=dt, daz=daz, ndof=ndof,
			B=B, a=a, da=da, iA=iA, model=model, resid=resid, chisq=chisq, neff=nt-ndof, ts=tedges,
			dur=t[-1]-t[0], azslope=azslope)

def normalize_piecewise_model(model):
	# Reformat the model so that it always has at least two points and an az slope.
	# This makes it easier to work with later.
	res    = model.copy()
	del res.B, res.iA
	npoint = res.ndof-res.azslope
	if npoint == 1:
		res.a, res.da = [np.concatenate([x[:,:1],x],1) for x in [res.a, res.da]]
		res.ts = np.concatenate([res.ts,[res.t0+res.dur]])
	if not res.azslope:
		res.a, res.da = [np.concatenate([x, np.zeros([2,1])],1) for x in [res.a, res.da]]
	return res

def read_model(fname):
	model = bunch.Bunch(segments=[])
	with open(fname, "r") as ifile:
		for line in ifile:
			# Line has format gind, ctime0, dur, ngood, nbad, baz, waz, bel, params..
			toks = line.split()
			t0, dur, baz, waz, bel = [float(toks[i]) for i in [1,2,5,6,7]]
			data         = np.array([float(w) for w in toks[9:]])
			npoint       = (len(data)-4)//5 if len(data) > 5 else 1
			pdata        = data[0:5*npoint].reshape(npoint,5)
			ts           = pdata[:,0]
			points       = pdata[:,1:].reshape(-1,2,2)
			slope        = data[5*npoint:].reshape(2,2) if npoint > 1 else None
			model.segments.append(bunch.Bunch(t0=t0, dur=dur, baz=baz, waz=waz, bel=bel, npoint=npoint, ts=ts, points=points, slope=slope))
	model.t0s = np.array([segment.t0+ts[0] for segment in model.segments])
	model.durs= np.array([segment.dur      for segment in model.segments])
	return model

def find_segment(model, t):
	i     = np.searchsorted(model.t0s, t)
	# Are we at one of the ends?
	if   i  == 0: return i 
	elif i  == len(model.t0s): return i-1
	# Otherwise we either belong to segment i-1...
	elif t >= model.t0s[i-1] and t <= model.t0s[i-1]+model.segments[i-1].dur:
		return i-1
	# ... or we're between segments, in which case we choose the nearest
	elif abs(model.t0s[i]-t) > abs(model.t0s[i-1]+model.segments[i-1].dur-t):
		return i-1
	else:
		return i

def evaluate_model(segment, t, maxslope=1/3600):
	# Are we earlier than the start?
	ts = segment.ts + segment.t0
	if t <= ts[0]:
		point = np.array(segment.points[0]) # {y,x},{val,dval}
		dur_extrap = ts[0]-t
		slope = None
	# Or after the end?
	elif t >= ts[-1]:
		point = np.array(segment.points[-1])
		dur_extrap = t-ts[-1]
		slope = None
	# Or inside?
	else:
		# Find which points we're between
		i2 = np.searchsorted(ts, t)
		i1 = i2-1
		# Linear interpolation
		x     = (t-ts[i1])/(ts[i2]-ts[i1])
		val   = segment.points[i1,:,0]*(1-x) + segment.points[i2,:,0]*x
		dval  = ((segment.points[i1,:,1]*(1-x))**2 + (segment.points[i2,:,1]*x)**2)**0.5
		point = np.concatenate([val[:,None],dval[:,None]],-1)
		dur_extrap = 0
		slope = segment.slope
	# Fill in slope
	if slope is None:
		slope = np.zeros((2,2))
	# Increase uncertainty based on extrapolation
	point[:,1] = (point[:,1]**2 + (maxslope*dur_extrap)**2)**0.5
	return bunch.Bunch(point=point, slope=slope, baz=segment.baz, extrap=dur_extrap)

class OffsetCache:
	def __init__(self, db):
		self.db    = db
		self.cache = {}
	def get(self, id):
		entry = self.db[id]
		fname = entry.point_offsets
		if fname not in self.cache:
			self.cache[fname] = files.read_point_offsets(fname)
		return self.cache[fname][entry.id]

def get_prediction(model, t):
	i     = find_segment(model, t)
	vals  = evaluate_model(model.segments[i], t)
	return vals

def bin_models(models, bsize=3600):
	# Bin models in time, with a bin size of bsize in seconds
	data = []
	for model in models:
		t      = np.concatenate([seg.ts+seg.t0 for seg in model.segments])
		points = np.concatenate([seg.points    for seg in model.segments],0)
		tmin, tmax = utils.minmax(t)
		data.append(bunch.Bunch(t=t, vals=points[:,:,0], ivars=points[:,:,1]**-2, tmin=tmin, tmax=tmax))
	tmin = np.min([d.tmin for d in data])
	tmax = np.max([d.tmax for d in data])
	npix = int(np.ceil((tmax-tmin)/bsize))
	ncomp= data[0].vals.shape[1]
	vals, ivars = np.zeros([2,len(models),ncomp,npix])
	for di, d in enumerate(data):
		pix = utils.nint((d.t-tmin)/bsize)
		for j in range(ncomp):
			vals [di,j] = np.bincount(pix, d.vals[:,j]*d.ivars[:,j], minlength=npix)[:npix]
			ivars[di,j] = np.bincount(pix, d.ivars[:,j],             minlength=npix)[:npix]
	vals[ivars>0] /= ivars[ivars>0]
	t = tmin+(np.arange(npix)+0.5)*bsize
	# vals: [arr,{y,x},nt]
	return bunch.Bunch(vals=vals, ivars=ivars, t=t, tmin=tmin, bsize=bsize)

def build_typical_array_offsets(models, bsize=3600, bouter=24):
	arrs = sorted(list(models.keys()))
	data = bin_models([models[key] for key in models], bsize=bsize)
	# Split our binned data into bigger bins, with bouter points each
	narr, ncomp, nt = data.vals.shape
	bvals = data.vals [:,:,:nt//bouter*bouter].reshape(narr,ncomp,nt//bouter,bouter)
	bivars= data.ivars[:,:,:nt//bouter*bouter].reshape(narr,ncomp,nt//bouter,bouter)
	# For each group, find the one with the largest total ivar, and make
	# it the reference
	best  = np.argmax(np.sum(bivars,(1,3)),0)
	ref   = np.moveaxis(bvals [(best, slice(None), np.arange(len(best)), slice(None))], 0,1)
	rivar = np.moveaxis(bivars[(best, slice(None), np.arange(len(best)), slice(None))], 0,1)
	mask  = rivar > 0
	# Our model is
	# d = p + a + n => a = np.sum((d-p)*N")/np.sum(N")
	offs  = np.sum(bivars*mask*(bvals-ref), -1)
	oivar = np.sum(bivars*mask,             -1)
	offs[oivar>0] /= oivar[oivar>0]
	# Return the information we need to get the array offsets at some random time
	return bunch.Bunch(arrs=arrs, offs=offs, oivar=oivar, tmin=data.tmin, bsize=bsize*bouter)

def get_array_offsets(array_offsets, t):
	pix = (t-array_offsets.tmin)/array_offsets.bsize
	offs= utils.interpol(array_offsets.offs, [pix])
	return {arr:offs[i] for i, arr in enumerate(array_offsets.arrs)}

def find_segments(t0s, durs, ts):
	"""Given a set of segments defined by starting points t0s and durations durs,
	return the index of the segment each time in t belongs to, or -1 if it's not
	in any"""
	inds = np.searchsorted(t0s, ts)
	good = (inds>0)&(ts-t0s[inds-1]<=durs[inds-1])
	inds[~good] = 0
	return inds-1

def group_cumulative(vals, target=15):
	# I can't see how to do this without looping. np.cumsum//target would lead to
	# the fractional overshoot of one group counting for the next, which doesn't
	# make sense.
	edges = [0]
	s     = 0
	for i,v in enumerate(vals):
		s += v
		if s >= target:
			edges.append(i+1)
			s = 0
	# extend the last group to the end of the array, to avoid too low sum there
	if len(edges) < 2: edges.append(len(vals))
	else: edges[-1] = len(vals)
	return edges

def read_freq_fluxes(desc):
	toks = desc.split(",")
	res  = {}
	for tok in toks:
		ftag, fname = tok.split(":")
		icat = dory.read_catalog(fname)
		res[ftag] = icat.flux[:,0]*1e3 # Jy -> mJy
	return res

def amps_to_fluxes_default(amps, freq=150, fwhm=1.4):
	barea = 2*np.pi*(fwhm*utils.arcmin*utils.fwhm)**2
	fconv = utils.flux_factor(barea, freq*1e9)
	return amps * fconv / 1e3 # uK -> mJy

def get_array(ids):
	return np.char.replace(np.char.replace(np.char.rpartition(db.ids,".")[:,2],"ar","pa"),":","_")

def names2uinds(names, return_unique=False):
	"""Given a set of names, return an array where each input element has been replaced by
	a unique integer for each unique name, counting from zero"""
	order = np.argsort(names)
	uinds = np.zeros(len(names),int)
	uvals, inverse = np.unique(names[order], return_inverse=True)
	uinds[order] = inverse
	return uinds if not return_unique else (uinds, uvals)

def remove_pointmodel_azslope(scans):
	"""Remove the azimuth slope from the pointing model of scans.
	Modifies scans in-place"""
	for scan in scans:
		scan.site.azslope_daz = 0
		scan.site.azslope_del = 0

def build_geometry(scandb, entrydb, res=0.5*utils.arcmin, margin=0.5*utils.degree):
	# all tods in group have same site. We also assume the same detector layout
	entry    = entrydb[scandb.ids[0]]
	site     = files.read_site(entry.site)
	detpos   = actdata.read_point_offsets(entry).point_template
	# Array center and radius in focalplane coordinates
	acenter  = np.mean(detpos,0)
	arad     = np.max(np.sum((detpos-acenter)**2,1)**0.5)
	# Find the center point of this group. We do this by transforming the array center to
	# celestial coordinates at the mid-point time of the group.
	t1 = np.min(scandb.data["t"]-0.5*scandb.data["dur"])
	t2 = np.max(scandb.data["t"]+0.5*scandb.data["dur"])
	t0 = 0.5*(t1+t2)
	baz, bel, waz, wel = [scandb.data[x][0]*utils.degree for x in ["baz", "bel", "waz", "wel"]]
	ra0, dec0 = coordinates.transform("bore", "cel", acenter, time=utils.ctime2mjd(t0), bore=[baz,bel,0,0], site=site)
	# We want to map in coordinates where (ra0,dec0) are mapped to (0,0) and where the up
	# direction is the same as in horizontal coordinates at time t0.
	ra_up, dec_up = coordinates.transform("tele", "cel", [0, np.pi/2], time=utils.ctime2mjd(t0), site=site)
	sys = ["cel", [[ra0, dec0, 0, 0, ra_up, dec_up], 0]]
	sysinfo = np.array([[ra0,dec0],[ra_up,dec_up]]) # more storage-friendly representation of sys
	# We need the bounding box for our map in these coordinates. We will get that by translating
	# (azmin,azmid,azmax)*(elmin,elmax)*(tmin,tmax) for each tod.
	ipoints = np.array(np.meshgrid(
		[t1,t2],
		np.linspace(baz-waz/2, baz+waz/2, 100),
		[bel-wel/2,bel+wel/2]
	)).reshape(3,-1)
	zero    = ipoints[0]*0
	opoints = coordinates.transform("bore", sys, zero+acenter[:,None], time=utils.ctime2mjd(ipoints[0]), bore=[ipoints[1],ipoints[2],zero,zero], site=site)
	box     = utils.bounding_box(opoints.T)
	box     = utils.widen_box(box, (arad+margin)*2, relative=False) # x2 = both sides, not just total width
	box[:,0]= box[::-1,0] # descending ra
	# Use this to build our geometry
	shape, wcs = enmap.geometry(box[:,::-1], res=res, proj="car", ref=(0,0))
	return bunch.Bunch(sys=sys, shape=shape, wcs=wcs, site=site, trange=[t1,t2], t0=t0, acenter=acenter, arad=arad,
			baz=baz, bel=bel, waz=waz, wel=wel, sysinfo=sysinfo)

def find_obs_times(radecs, targ_el, t0, site, step=1, tol=1e-4, maxiter=100):
	"""Find the unix time when each object with celestial coordinates radecs[{ra,dec},...]
	hits the given target elevation. This can fail for sources that never rise high enough
	in the sky to reach targ_el. Those have to be very close to north/south, though.
	They could be handled by looking at the whole array instead of just the array center,
	but for now we will just return NaN for these.
	"""
	# The maximum el for each source is i/2-abs(dec-lat). We could use this to
	# figure out which sources don't properly cross the array, but for now we
	# will just skip them.
	maxel = np.pi/2 - np.abs(radecs[1]-site.lat*utils.degree)
	good  = np.where(maxel > targ_el)[0]
	if len(good) == 0: return np.full(radecs_shape[1], np.nan)
	def getel(dt): return coordinates.transform("cel", "tele", radecs, time=utils.ctime2mjd(t0+dt), site=site)[1]
	def getder(dt): return (getel(dt+step/2)-getel(dt-step/2))/step
	dt    = np.zeros(radecs.shape[1:])
	for it in range(maxiter):
		el    = getel(dt)
		deriv = getder(dt)
		resid = el-targ_el
		dt   -= resid/deriv
		resid = np.max(np.abs(resid)[good])
		if resid < tol: break
	res      = t0+dt
	bad      = np.abs(el-targ_el) > tol
	res[bad] = np.nan
	return res

def recover_boresight(pos_tele, pos_fplane):
	"""Given a position in both telescope coordinates (pos_tele[{az,el},...]) and
	focalplane coordinates (pos_fplane[{x,y},...]), infer the boresight coordinates
	in telescope coordinates: [{baz,bel},...]. Telescope coordinates are very similar to
	horizontal coordinates, and only differ due to the telescope baseline tilt."""
	# p_fplane = [xf,yf,zf]
	# Ry(-bel) p_fplane = [
	#   xf cos(bel) - zf sin(bel),
	#   yf,
	#   zf cos(bel) + xf sin(bel),
	# ]
	# Rz(baz) Ry(-bel) p_fplane = [
	#  [xf cos(bel) - zf sin(bel)] cos(baz) - yf sin(baz),
	#  yf cos(baz) + [xf cos(bel) - zf sin(bel)] sin(baz),
	#  zf cos(bel) + xf sin(bel)
	# ]
	# This should equal [xt,yt,zt]. The z axis lets us solve for bel:
	# zf cos(bel) + xf sin(bel) = zt
	# bel = -2 atan2(±sqrt(xf**2+zf**2-zt**2)-xf, zt+zf)
	# There are two solutions. I think there should only be one with sensible angle
	# bounds, though. So choose the one with abs(bel) <= pi/2
	# Given bel we can compute q = xf cos(bel) - zf sin(bel), and use this to solve for baz:
	# q  cos(baz) - yf sin(baz) = xt
	# yf cos(baz) + q  sin(baz) = yt
	# baz = 2 atan2(±sqrt(yf**2+q**2-xt**2)-yf, xt+q)
	# Wolfram alpha couldn't solve the two equations jointly, for some reason.
	# The solution above is for the first one. Of the sign options there, choos the
	# one that also fulfills the second equation.
	pos_tele, pos_fplane = [a.T for a in np.broadcast_arrays(pos_tele.T, pos_fplane.T)]
	ishape = pos_tele.shape[1:]
	xf, yf, zf = utils.ang2rect(pos_fplane).reshape(3,-1)
	xt, yt, zt = utils.ang2rect(pos_tele).reshape(3,-1)
	pm  = np.array([1,-1])[:,None]
	bel = -2*np.arctan2(pm*(xf**2+zf**2-zt**2)**0.5-xf, zt+zf)
	# Of our two possible angles, select the one within the normal elevation range.
	# The argmax stuff is to ensure that we always slect exactly one angle
	tol = np.finfo(bel.dtype).eps*10
	good= np.abs(bel) <= np.pi/2+tol
	def select(arr, good):
		return np.where(np.argmax(good,0)==0, *arr)
	bel = select(bel, good)
	# Compute the azimuth angle too. Here we have a second equation to help us choose
	# the right angle. There is probably a better way of doing this, without all those
	# conditionals
	q   = xf*np.cos(bel) - zf*np.sin(bel)
	baz = 2*np.arctan2(pm*(yf**2+q**2-xt**2)**0.5-yf, xt+q)
	good= np.abs(yf*np.cos(baz) + q*np.sin(baz) - yt) < tol
	baz = select(baz, good)
	# Go back to original shape
	baz = baz.reshape(ishape)
	bel = bel.reshape(ishape)
	return np.array([baz, bel])

def build_src_jacobi(info, spos_cel, stimes, delta=1*utils.arcmin):
	"""Compute the [{ora,ocde},{x,y},nsrc] jacobian for the sources with celestial coordinates
	spos_cel[{ra,dec},nsrc] that hit the array center at times stimes[nsrc]. ora, odec
	refer to the "ra" and "dec" of the rotated output coordinate system desribed by info.sys."""
	# Why did I get acenter_el separately? As long as our stimes are correct,
	# the transform should give us both. In my tests it [1] of the trans is
	# identical to acenter_el to at least 10 digits. So what I have here isn't
	# wrong, but it's unnecessarily complicated
	acenter_el = coordinates.recenter(info.acenter, [0,0,0,info.bel])[1]
	acenter_az = coordinates.transform("cel","tele", spos_cel, time=utils.ctime2mjd(stimes), site=info.site)[0]
	spos_tele  = np.array([acenter_az, acenter_az*0+acenter_el])
	sbaz, sbel = recover_boresight(spos_tele, info.acenter)
	# With the boresight known, we can now propagate [x,y], [x+dx,y] and [x,y+dy], where [x,y] = info.acenter,
	# to our final output system
	ipoints = info.acenter[:,None]+np.array([[0,1,0],[0,0,1]])*delta
	zero    = stimes*0
	opoints = coordinates.transform("bore", info.sys, ipoints[:,:,None], time=utils.ctime2mjd(stimes), bore=[sbaz, sbel, zero, zero], site=info.site)
	jac_osys= np.moveaxis([opoints[:,1]-opoints[:,0], opoints[:,2]-opoints[:,0]], 0, 1)/delta # [{ora,odec},{x,y},nok]
	# Get it in terms of pixels too
	opix    = enmap.sky2pix(info.shape, info.wcs, opoints[::-1])
	jac_opix= np.moveaxis([opix[:,1]-opix[:,0], opix[:,2]-opix[:,0]], 0, 1)/delta # [{oy,ox},{x,y},nok]
	return jac_osys, jac_opix

def build_map(info, scans, dtype=np.float32, tag=None, comm=None):
	if comm is None: comm = mpi.COMM_WORLD
	pre = "" if tag is None else tag + " "
	L.info(pre + "Initializing equation system")
	signal_cut  = mapmaking.SignalCut(scans, dtype=dtype, comm=comm)
	area        = enmap.zeros((ncomp,)+info.shape, info.wcs, dtype)
	signal_sky  = mapmaking.SignalMap(scans, area, comm=comm, sys=info.sys)
	window      = mapmaking.FilterWindow(config.get("tod_window"))
	eqsys       = mapmaking.Eqsys(scans, [signal_cut, signal_sky], weights=[window], dtype=dtype, comm=comm)
	L.info(pre + "Building RHS")
	eqsys.calc_b()
	L.info(pre + "Building preconditioner")
	signal_cut.precon = mapmaking.PreconCut(signal_cut, scans)
	signal_sky.precon = mapmaking.PreconMapBinned(signal_sky, scans, [window])
	L.info(pre + "Solving")
	solver = cg.CG(eqsys.A, eqsys.b, M=eqsys.M, dot=eqsys.dot)
	while solver.i < args.niter:
		t1 = time.time()
		solver.step()
		t2 = time.time()
		L.info(pre + "CG step %5d %15.7e %6.1f %6.3f" % (solver.i, solver.err, (t2-t1), (t2-t1)/len(scans)))
	# Ok, now that we have our map. Extract it and ivar. That's the only stuff we need from this
	map  = eqsys.dof.unzip(solver.x)[1]
	ivar = signal_sky.precon.div[0,0]
	return map, ivar

def build_exposure_mask(ivar, quant=0.9, tol=0.01, edge=2*utils.arcmin, ignore=1*utils.arcmin, thin=100):
	samps= ivar[ivar>0]
	if len(samps) > 100*thin: samps = samps[::thin]
	if len(samps) == 0: return enmap.zeros(ivar.shape, ivar.wcs, bool)
	ref  = np.percentile(ivar[ivar>0][::thin], 100*quant)
	mask = ivar > ref*tol
	# This shrinks the mask by edge, but ignores holes smaller than ignore
	mask = enmap.shrink_mask(mask, edge+ignore)
	mask = enmap.grow_mask  (mask, ignore)
	return mask

def too_masked(mask, rcore=10, maxcut=0.1, maxcut_core=0.05):
	"""mask: True = usable, False = unusable"""
	h,w = mask.shape
	return np.mean(~mask[h//2-rcore:h//2+rcore,w//2-rcore:w//2+rcore]) > maxcut_core or np.mean(~mask) > maxcut

def extract_srcs(map, ivar, spos, tsize=30):
	# Get the source thumbnails
	spix       = utils.nint(map.sky2pix(spos[::-1]).T)
	pboxes     = np.moveaxis([spix-tsize//2,spix-tsize//2+tsize],0,1) # [srcs,{from,to},{y,x}]
	inds, maps, ivars, centers, pixshapes = [], [], [], [], []
	for si, pbox in enumerate(pboxes):
		# Is this thumbnail acceptable? Reject if source itself is masked,
		# or if too large a fraction is masked
		tivar = ivar.extract_pixbox(pbox)
		if too_masked(tivar>0): continue
		tmap  = map.extract_pixbox(pbox)
		# We could store the geometry of each tile, but that's cumbersome. For such
		# small area like this, all we need is the pixel shape and source position.
		center   = tmap.sky2pix(spos[::-1,si])
		pixshape = tmap.pixshape()
		# And append to output lists
		for alist, a in zip([inds,maps,ivars,centers,pixshapes],[si,tmap,tivar,center,pixshape]):
			alist.append(a)
	return bunch.Bunch(inds=np.array(inds), maps=np.array(maps), ivars=np.array(ivars),
			centers=np.array(centers), pixshapes=np.array(pixshapes))

def get_polrot(isys, osys, spos, site):
	"""Get the angle by which the polarization was rotated when going from isys to osys."""
	return coordinates.transform(isys, osys, spos, site=site, pol=True)[2]

def build_noise_model_simple(map, ivar, tsize=30):
	"""Build a simple noise model consisting of a single 2d power spectrum for the whitened map,
	scaled down to a single tile's size. This ignores sky curvature and any weather etc. changes
	during the scan, but will probably be pretty good. We will also ignore QU correlations, so the
	result will just be TT,QQ,UU."""
	# Start by making the maps whole multiples of the tile size. This will
	# make the final downgrade of the 2d power spectrum clean and robust
	(nby, nbx), (ey, ex) = np.divmod(map.shape[-2:], tsize)
	map, ivar = [a[...,ey//2:ey//2+nby*tsize, ex//2:ex//2+nbx*tsize] for a in [map, ivar]]
	# Whitened and mask-corrected map
	wmap = map * ivar**0.5 / np.mean(ivar > 0)**0.5
	ps2d = np.abs(enmap.fft(wmap))**2
	ps2d = symcap_ps2d_med(ps2d)
	thumb_ps2d = enmap.downgrade(ps2d, [nby,nbx])
	return thumb_ps2d

def symcap_ps2d(ps2d):
	ps1d, l1d = ps2d.lbin()
	l = ps2d.modlmap()
	ps2d = enmap.samewcs(np.maximum(ps2d, utils.interp(l, l1d, ps1d)))
	return ps2d

def symcap_ps2d_med(ps2d):
	ps2d = ps2d.copy()
	l    = ps2d.modlmap()
	dl   = max(abs(l[1,0]),abs(l[0,1]))
	reg  = (l/dl).astype(int)+1
	inds = np.arange(1,np.max(reg)+1)
	for I in utils.nditer(ps2d.shape[:-2]):
		ps1d_med = ndimage.median(ps2d[I], reg, inds)
		ps2d_med = ps1d_med[reg-1]
		ps2d[I]  = np.maximum(ps2d[I], ps2d_med)
	return ps2d

def build_noise_model_dummy(map, ivar, tsize=30):
	"""Build a simple noise model consisting of a single 2d power spectrum for the whitened map,
	scaled down to a single tile's size. This ignores sky curvature and any weather etc. changes
	during the scan, but will probably be pretty good. We will also ignore QU correlations, so the
	result will just be TT,QQ,UU."""
	# Start by making the maps whole multiples of the tile size. This will
	# make the final downgrade of the 2d power spectrum clean and robust
	(nby, nbx), (ey, ex) = np.divmod(map.shape[-2:], tsize)
	map, ivar = [a[...,ey//2:ey//2+nby*tsize, ex//2:ex//2+nbx*tsize] for a in [map, ivar]]
	# Whitened and mask-corrected map
	wmap = map * ivar**0.5 / np.mean(ivar > 0)**0.5
	ps2d = (1 + ((map.modlmap()+10)/3000)**-3.5)
	thumb_ps2d = enmap.downgrade(ps2d, [nby,nbx])
	return np.repeat(thumb_ps2d[None], 3, 0)

def mask_bright_srcs(shape, wcs, spos, samp, amp_lim=10e3, rmask=7*utils.arcmin):
	spos_bright = spos[:,samp>amp_lim]
	r = enmap.distance_from(shape, wcs, spos[::-1,samp>amp_lim], rmax=rmask)
	return r >= rmask

def write_bunch(fname, bunch):
	with h5py.File(fname, "w") as ofile:
		for key in bunch:
			ofile[key] = bunch[key]

def write_empty(fname, desc="empty"):
	with open(fname, "w") as ofile:
		ofile.write("%s\n" % desc)

def maybe_skip(condition, reason, ename):
	if not condition: return False
	L.info("%s. Skipping" % reason)
	write_empty(ename, reason)
	return True


def make_groups(ifiles, grouping="none"):
	"""Given a set of input files with name format */thumbs_timetag_array_ftag.hdf,
	return a list of groups that should be analysed together. Each group consists
	of a list of indices into the list of input files. The valid groupings are
	"none":  Each file in its own group.
	"array": Group the different frequencies of the same array for each oberving time.
	"all":   Group all observations taken at the same time."""
	if grouping == "none":
		return [[i] for i in range(len(ifiles))]
	elif grouping == "array":
		tags = np.char.rpartition(np.array(ifiles), "/")[:,2]
		tags = np.char.rpartition(tags, "_")[:,0]
		return utils.find_equal_groups(tags)
	elif grouping == "all":
		tags = np.char.rpartition(np.array(ifiles), "/")[:,2]
		tags = np.char.rpartition(tags, "_")[:,0]
		tags = np.char.rpartition(tags, "_")[:,0]
		return utils.find_equal_groups(tags)
	else:
		raise ValueError("Unknown grouping '%s'" % str(grouping))

def read_blacklist(bdesc):
	if bdesc is None: return None
	try: return np.loadtxt(bdesc, usecols=(0,), ndmin=2)[:,0].astype(int)
	except OSError: return np.array([int(w) for w in bdesc.split(",")])

def add_geometry(data):
	"""The data structure we read in doesn't contain any wcs information, so
	set that up here, along the r and l in each pixel for convenience"""
	odata = data.copy()
	shape = data.maps.shape[-2:]
	wcss  = []
	rs    = []
	ls    = []
	for i in range(len(data.table)):
		wcs = wcsutils.WCS(naxis=2)
		wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
		wcs.wcs.cdelt = data.table["pixshape"][i][::-1]/utils.degree
		wcs.wcs.crpix = data.table["center"][i][::-1]+1
		wcss.append(wcs)
		rs.append(enmap.modrmap(shape, wcs, [0,0]))
		ls.append(enmap.modlmap(shape, wcs))
	odata.wcss = np.array(wcss)
	odata.rs   = rs
	odata.ls   = ls
	return odata

def expand_table(data):
	# Add the telescope coordinates (basically the same as horizontal coords).
	# Ideally this information would have already been present in table to begin with,
	# such that all act-depenent stuff was there.
	from enlib import coordinates
	import numpy.lib.recfunctions as rfn
	pos_tele   = coordinates.transform("cel", "tele", data.table["pos_cel"].T, time=utils.ctime2mjd(data.table["ctime"]))
	model_flux = data.table["ref_flux"]
	data.table = rfn.append_fields(data.table, ["az","el", "model_flux"], [pos_tele[0], pos_tele[1], model_flux])
	return data

def mask_data(data, tol=0.1):
	odata = data.copy()
	refs  = np.median(data.ivars,(-2,-1))
	mask  = data.ivars > refs[:,None,None]*tol
	odata.ivars *= mask
	odata.maps  *= mask[:,None]
	return odata

def get_lbeam_flat(r, br, shape, wcs):
	"""Given a 1d beam br(r), compute the 2d beam transform bl(ly,lx) for
	the l-space of the map with the given shape, wcs, assuming a flat sky.
	The resulting beam is normalized to have a peak of 1."""
	cpix   = np.array(shape[-2:])//2-1
	cpos   = enmap.pix2sky(shape, wcs, cpix)
	rmap   = enmap.shift(enmap.modrmap(shape, wcs, cpos), -cpix)
	bmap   = enmap.ndmap(np.interp(rmap, r, br, right=0), wcs)
	lbeam  = enmap.fft(bmap, normalize=False).real
	return lbeam

def gapfill_data(data, n=3, tol=15):
	odata = data.copy()
	if n <= 1: return odata
	# First figure out which pixels are bad. We can't trust ivar to tell us this,
	# apparently. So we will compare the map pixel values to median-filtered versions
	# of the same, and see if they differ by more than expected from the noise.
	# It looks like all problematic pixels in T are also bad in Q, and to a lesser
	# extent in U. This lets us use pol as the masking diagnostic, which helps avoiding
	# problems with strong sources being masked.
	fmaps = ndimage.median_filter(data.maps, (1,)*(data.maps.ndim-2)+(n,1))
	diffs = np.max(np.abs(data.maps-fmaps)[:,1:],1)
	ref_err = np.median(data.ivars, (-2,-1))**-0.5
	bad   = diffs > ref_err[:,None,None]*tol
	bad   = np.tile(bad[:,None], (1,fmaps.shape[1],1,1))
	# Replace bad pixels with filtered versions
	odata.maps[bad] = fmaps[bad]
	odata.gapfilled = np.sum(bad,(-2,-1))
	return odata

def calibrate_data(data):
	"""Transform from uK to mJy/sr"""
	# uK to mJy/sr
	odata       = data.copy()
	unit        = utils.dplanck(data.freq*1e9, utils.T_cmb) / 1e3
	odata.maps  = data.maps  * unit
	odata.ivars = data.ivars * unit**-2
	return odata

def nonan(a):
	res = np.asanyarray(a).copy()
	res[~np.isfinite(a)] = 0
	return res

def apod_crossfade(a, n):
	a = np.asanyarray(a).copy()
	def helper(a1, a2, axis):
		if axis < 0: axis += a1.ndim
		x = (np.arange(1, a1.shape[axis]+1)/(2*a1.shape[axis]))[(slice(None),)+(None,)*(a1.ndim-1-axis)]
		reverse = (slice(None),)*axis + (slice(None,None,-1),)
		return a1*(1-x) + a2[reverse]*x, a2*x + a1[reverse]*(1-x)
	a[...,:,-n:], a[...,:,:n] = helper(a[...,:,-n:], a[...,:,:n], -1)
	a[...,-n:,:], a[...,:n,:] = helper(a[...,-n:,:], a[...,:n,:], -2)
	return a

def reproject_data(data, oshape, owcs, apod=10, pointoff=None):
	"""Reproject the maps from having separate wcses per thumb to a common
	geometry given by oshape, owcs. Replaces maps, ivars and ps2d. Does not change table."""
	odata= data.copy()
	opos = enmap.posmap(oshape, owcs)
	nsrc = len(data.table)
	# Estimate how much apod we can afford. We want to leave an unapodized area
	# at least 4 arcmin + 2 bsigma on each edge
	res    = np.mean(np.product(data.table["pixshape"],1))**0.5
	bsigma = (data.barea/(2*np.pi))**0.5
	apod   = min(apod, utils.ceil(min(data.maps.shape[-2:])/2 - (4*utils.arcmin+2*bsigma)/res))
	odata.maps = enmap.zeros(data.maps. shape[:-2]+oshape[-2:], owcs, data.maps. dtype)
	odata.ivars= enmap.zeros(data.ivars.shape[:-2]+oshape[-2:], owcs, data.ivars.dtype)
	# optional pointing offset
	pointoff = np.zeros((nsrc,2)) + (0 if pointoff is None else pointoff)
	for sind in range(nsrc):
		# jacobian {y,x},{focx,focy}. We switch the last axis to make it {y,x},{focy,focx} instead
		jac  = data.table["jacobi"][sind,:,::-1]
		ipix = np.einsum("ab,bij->aij", jac, opos - pointoff[sind,:,None,None]) + data.table["center"][sind,:,None,None]
		map  = enmap.apod(enmap.enmap(data.maps[sind], data.wcss[sind]), apod)
		#map  = apod_crossfade(enmap.enmap(data.maps[sind], data.wcss[sind]), apod)
		# Ideally we would transform QU to tan too, then measure, and finally transform back
		# all the way to cel when interpreting the result. In practice curvature across the
		# thumbnail is tiny, so it's enough to transform QU from the mapping system to cel here.
		map  = enmap.rotate_pol(map, -data.table["polrot"][sind])
		ivar = enmap.enmap(data.ivars[sind], data.wcss[sind])
		odata.maps [sind] = utils.interpol(map,  ipix, mode="cubic", border="wrap")
		with utils.nowarn():
			# Interpolate var instead of ivar. This makes low-hit regions grow instead of shrink
			# when interpolating, making us downweight areas that have contributions from low-hit
			# pixels. This removes a lot of stripiness and bad pixels in the result.
			odata.ivars[sind] = nonan(1/utils.interpol(1/ivar, ipix, mode="lin", border="wrap"))
		# ivars is in units of white noise per original pixel. Transform that to the equivalent
		# white noise per new pixel
		old_pixsize = np.product(data.table["pixshape"][sind])
		new_pixsize = odata.ivars[sind].pixsize()
		odata.ivars[sind] *= new_pixsize/old_pixsize

	# Handle the power spectrum too
	mean_wcs = wcsutils.WCS(naxis=2)
	mean_wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
	mean_wcs.wcs.cdelt = np.mean(data.table["pixshape"][:,::-1],0)/utils.degree
	mean_wcs.wcs.crpix = np.mean(data.table["center"][:,::-1],0)+1
	# This will have inf entries, but will invert before using it
	with utils.nowarn():
		odata.ps2d = 1/enmap.enmap(utils.interpol(1/data.ps2d, enmap.l2pix(data.maps.shape, mean_wcs, enmap.lmap(oshape, owcs))), owcs)
	return odata

def beam2fwhm(r, br):
	return 2*r[br>np.max(br)/2][-1]

def select_srcs(data, sel):
	odata = data.copy()
	odata.table = data.table[sel]
	odata.maps  = data.maps[sel]
	odata.ivars = data.ivars[sel]
	if "wcss" in odata:
		odata.wcss  = data.wcss[sel]
	return odata

class DataError(Exception): pass

def prepare_data(data, oshape, owcs, blacklist=None, pointoff=None):
	if blacklist is not None:
		data = select_srcs(data, ~utils.contains(data.table["sid"], blacklist))
	if data.maps.size == 0: raise DataError("No sources left")
	data.ftag = data.ftag.decode()
	data = add_geometry(data)
	data = gapfill_data(data)
	data = calibrate_data(data)
	data = expand_table(data)
	#data = mask_data(data)

	data = reproject_data(data, oshape, owcs, pointoff=pointoff)
	data.lbeam = get_lbeam_flat(data.beam[0], data.beam[1],    oshape, owcs)
	data.lbeam2= get_lbeam_flat(data.beam[0], data.beam[1]**2, oshape, owcs)
	norm = np.max(data.lbeam)
	data.lbeam  /= norm
	data.lbeam2 /= norm**2
	data.fwhm    = beam2fwhm(data.beam[0], data.beam[1])
	return data

def solve(rho, kappa):
	mask = kappa==0
	with utils.nowarn():
		flux  = rho/kappa
		dflux = kappa**-0.5
		snr   = flux/dflux
		for a in [flux, dflux, snr]:
			a[mask] = 0
	return flux, dflux, snr

def group_srcs_snmin(gdata, inds, snmin=5):
	"""Given a set of inds [(gind,sind),...] into gdata, split these into sub-groups
	that each have a (model) S/N ratio of snmin, but keep individual sources that are
	already that bright separate. So undetectable sources will be merged, but not
	already detectable ones. Multiple versions of the same source will always be merged."""
	table      = np.concatenate([gdata[gind].table[sind:sind+1] for gind, sind in inds]).view(np.recarray)
	ref_dflux  = np.array([gdata[gind].ref_dflux[sind] for gind, sind in inds])
	model_snr  = table["model_flux"]/ref_dflux
	# First merge multiple instances of the same source
	wgroups    = utils.find_equal_groups(table["sid"])
	# Order these by time
	t     = table["ctime"][[g[0] for g in wgroups]]
	order = np.argsort(t)
	# And loop through, building the final groups
	ogroups = []
	weak_g, weak_snr2, weak_prev = [], 0, -1
	for ind in order:
		g   = wgroups[ind]
		snr2= np.sum(model_snr[g]**2)
		if snr2 >= snmin**2:
			# Is the work-group already strong enough? If so, accept it as is
			ogroups.append(g)
		else:
			# Otherwise accumulate into the weak group
			weak_g    += g
			weak_snr2 += snr2
			if weak_snr2 >= snmin**2:
				weak_prev = len(ogroups)
				ogroups.append(weak_g)
				weak_g, weak_snr2 = [], 0
	# If we have some left-over weak groups, merge them with the last one.
	# Otherwise just accept them as a sub-standard group
	if weak_g:
		if weak_prev >= 0: ogroups[weak_prev] += weak_g
		else:              ogroups.append(weak_g)
	# Finally translate the indices from internal to full indices
	ogroups = [[inds[i] for i in g] for g in ogroups]
	return ogroups

def find_peaks(snr, snlim=None, bounds=None, fwhm=None):
	"""Find the peak location in the S/N map snr. Any cropping etc. is assumed to have
	already been done."""
	if bounds is not None: snr   = snr.submap(bounds)
	if snlim  is None:     snlim = 4
	dtype = [("snr", "d"), ("pos", "2d"), ("dpos", "2d")]
	labels, nlabel = ndimage.label(snr >= snlim)
	if nlabel == 0: return np.zeros([0], dtype).view(np.recarray)
	allofthem = np.arange(nlabel)+1
	peak_snr  = ndimage.maximum(snr, labels, allofthem)
	peak_pos  = snr.pix2sky(np.array(ndimage.center_of_mass(snr**2, labels, allofthem)).T).T
	# Estimate the position error from the FWHM and snr. This is safer than peak chisq -1 area,
	# since that area could be very small (e.g. less than one pixel)
	res = np.zeros(nlabel, dtype).view(np.recarray)
	res.snr = peak_snr
	res.pos = peak_pos
	objects   = ndimage.find_objects(labels)
	for i, obj in enumerate(objects):
		sub_snr = snr[obj]
		if fwhm is None:
			mask    = sub_snr >= peak_snr[i]/2
			area    = np.sum(sub_snr.pixsizemap()*mask)
			area   /= 2 # we want area of B, not of N"B²
			fwhm_est= 2*(area/np.pi)**0.5
		else:
			fwhm_est= fwhm
		res.dpos[i] = 0.6*fwhm/peak_snr[i]
	# Sort them by peak snr
	order     = np.argsort(peak_snr)[::-1]
	res = res[order]
	return res

class FitMultiModalError(Exception): pass
class FitNonDetectionError(Exception): pass
class FitUnexpectedFluxError(Exception): pass

def fit_group(gdata, inds, bounds=None, snmin=None, tol=0.2, use_brightest=False):
	"""Fit the source offset (the negative of the pointing offset) and the
	source flux for each source in the time-group given by tgroups. Returns
	a bunch with members pos, dpos, snr, ctime, fluxes[nsrc,TQU], dfluxes[nsrc,TQU]"""
	# First flatten rho, kappa and table for convenience
	rho   = enmap.enmap([gdata[gi].rho  [subi] for gi, subi in inds])
	kappa = enmap.enmap([gdata[gi].kappa[subi] for gi, subi in inds])
	table = np.concatenate([gdata[gi].table[subi:subi+1] for gi, subi in inds]).view(np.recarray)
	fwhms = np.array([gdata[gi].fwhm for gi, subi in inds])
	model_flux = table["model_flux"]
	model_ivar = kappa[:,0].at([0,0],order=1)
	model_snr  = model_flux * model_ivar**0.5
	# Build the optimally weighted combined map
	rel_flux  = model_flux / np.max(model_flux)
	rho_tot   = np.sum(rho[:,0]  *rel_flux[:,None,None],0)
	kappa_tot = np.sum(kappa[:,0]*rel_flux[:,None,None]**2,0)
	snr_tot   = solve(rho_tot, kappa_tot)[2]
	# Find the (hopefully) single significant peak in the result
	fwhm_eff  = np.sum(fwhms * model_snr**2)/np.sum(model_snr**2)
	peaks     = find_peaks(snr_tot, bounds=bounds, snlim=snmin, fwhm=fwhm_eff)
	# Disqualify any unreasonably weak peaks
	tot_snr = np.sum(model_snr**2)**0.5
	if len(peaks) == 0: raise FitNonDetectionError()
	if use_brightest:
		peaks = peaks[[np.argmax(peaks.snr)]]
	else:
		peaks = peaks[peaks.snr > tot_snr * tol]
		if len(peaks) == 0: raise FitNonDetectionError()
		if len(peaks)  > 1: raise FitMultiModalError()
	# Ok, we have a good fit
	fit = bunch.Bunch(pos=peaks[0].pos, dpos=peaks[0].dpos, snr=peaks[0].snr, model_snr=tot_snr, snr_map=snr_tot)
	# Measure the properties of individual srcs
	fit.table   = table
	fit.flux    = rho.at(fit.pos)/kappa.at(fit.pos)
	fit.dflux   = kappa.at(fit.pos)**-0.5
	fit.ftags   = [gdata[gi].ftag for gi,subi in inds]
	fit.inds    = inds
	fit.ref_dflux = np.mean(kappa_tot)**-0.5
	# Measure the weighted mean ctime, az and el for the group, as well as a weighted stddev
	W = model_snr**2
	def meandev(a, W):
		mean = np.sum(a*W)/np.sum(W)
		dev  = (np.sum((a-mean)**2*W)/np.sum(W))**0.5
		return mean, dev
	for name in ["ctime", "az", "el"]:
		fit[name], fit["d"+name] = meandev(table[name], W)
	for i, name in enumerate(["ra","dec"]):
		fit[name], fit["d"+name] = meandev(utils.unwind(table["pos_cel"][:,i]), W)
	# The boresight info is assumed to be constant inside a group
	for name in ["baz", "waz", "bel"]:
		fit[name] = gdata[0][name]
	return fit

def merge_files(ifiles, ofile):
	lines = []
	for ifile in ifiles:
		with open(ifile, "r") as f:
			lines += f.readlines()
	lines = sorted(lines)
	with open(ofile, "w") as f:
		f.writelines(lines)

def debug_outlier(gdata, inds, bounds=None):
	"""Output thumbnails and catalog information for the point sources that went
	into the given inds, and exit"""
	with open("debug_info.txt", "w") as ofile:
		for i, (gi, subi) in enumerate(inds):
			data = gdata[gi]
			enmap.write_map("debug_map_%02d.fits" % i, data.maps[subi])
			enmap.write_map("debug_rho_%02d.fits" % i, data.rho[subi])
			enmap.write_map("debug_kappa_%02d.fits" % i, data.kappa[subi])
			enmap.write_map("debug_ivar_%02d.fits" % i, data.ivars[subi])
			with utils.nowarn():
				enmap.write_map("debug_snr_%02d.fits" % i, data.rho[subi]/gdata[gi].kappa[subi]**0.5)
			tab  = data.table[subi].view(np.recarray)
			desc = "%2d %5d %8.3f %8.3f %8.2f" % (i, tab.sid, tab.pos_cel[0]/utils.degree, tab.pos_cel[1]/utils.degree, tab.ref_flux)
			ofile.write(desc + "\n")
			print(desc)
	tot_rho   = np.sum(data.rho,0)
	tot_kappa = np.sum(data.kappa,0)
	tot_flux, tot_dflux, tot_snr = solve(tot_rho, tot_kappa)
	enmap.write_map("debug_snr_tot.fits", tot_snr)
	fit = fit_group(gdata, inds, bounds=bounds)
	enmap.write_map("debug_snr_tot2.fits", fit.snr_map)
	sys.exit(0)


if mode == "build":
	import numpy as np, warnings, time, h5py, os, sys
	from enlib  import config, coordinates, mapmaking, bench, scanutils, log, cg, dory
	from pixell import utils, enmap, pointsrcs, bunch, mpi
	from enact  import filedb, actdata, actscan, files
	from scipy  import ndimage, optimize, spatial

	config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
	config.default("downsample", 1, "Factor with which to downsample the TOD")
	config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
	config.default("tod_window", 5.0, "Number of samples to window the tod by on each end")
	config.default("eig_limit", 0.1, "Pixel condition number below which polarization is dropped to make total intensity more stable. Should be a high value for single-tod maps to avoid thin stripes with really high noise")
	config.set("pmat_map_order", 0)

	# Not sure why linear gapfilling performs so much better here. With the other
	# one I got prominent blotches in cut regions.
	# Looks like linear overall has much more of those cases, even if it can be better in a few
	# instances.
	#config.set("gapfill", "linear")

	parser = config.ArgumentParser()
	parser.add_argument("mode", help="build")
	parser.add_argument("srcs", help="Source database file")
	parser.add_argument("sel",  help="TOD selction query")
	parser.add_argument("odir", help="Output directory")
	parser.add_argument("prefix", nargs="?", help="Output file name prefix")
	parser.add_argument("-r", "--res",    type=float, default=0.5, help="Resolution in arcmins")
	parser.add_argument("-R", "--rad",    type=float, default=15,  help="Thumbnail radius in arcmins")
	parser.add_argument("-n", "--niter",  type=int,   default=10)
	parser.add_argument("-c", "--cont",   action="store_true")
	parser.add_argument("-M", "--maps",   type=int,   default=0, help="0: no debug map output. N: Output debug maps downgraded by this factor")
	parser.add_argument("-F", "--fix",       type=str, default=None)
	parser.add_argument(      "--freq-cats", type=str, default=None)
	parser.add_argument(      "--isys",      type=str, default="cel")
	args = parser.parse_args()

	utils.mkdir(args.odir)
	comm_world  = mpi.COMM_WORLD
	comm_self   = mpi.COMM_SELF
	dtype = np.float32 if config.get("map_bits") == 32 else np.float64
	ncomp = 3
	tsize = int(np.ceil(2*args.rad/args.res))
	root  = args.odir + "/" + (args.prefix + "_" if args.prefix else "")
	down  = config.get("downsample")
	tag2freq = {"f030": 27, "f040": 39, "f090": 98, "f150": 150, "f220": 225}
	# Set up logging
	utils.mkdir(root + ".log")
	logfile   = root + ".log/log%03d.txt" % comm_world.rank
	log_level = log.verbosity2level(config.get("verbosity"))
	L = log.init(level=log_level, file=logfile, rank=comm_world.rank, shared=False)

	# Read our point source database: .ra, .dec and .I (amp uK). Should probably change to flux later
	# These sources will be used for all arrays and frequencies, with the assumption that all have the
	# same amplitude. May improve this later, but that's what it is for now.
	srcs     = pointsrcs.read(args.srcs)
	spos_raw = np.array([srcs.ra, srcs.dec]) # *utils.degree

	# Set up our fluxes. This is a bit inelegant. But it works.
	if args.freq_cats:
		freq_fluxes = read_freq_fluxes(args.freq_cats)
	else:
		fluxes = amps_to_fluxes_default(srcs.I)
		freq_fluxes = {key: fluxes for key in tag2freq}

	# First identify the scanning periods. We do this using all our data, not just the parts we
	# want to use later. A scanning period is a period where baz and bel are constant and the
	# end of one tod (for one array and freq) is followed by the beginning of the next.
	filedb.init()
	periods = scanutils.find_scan_periods(filedb.scans)

	# Then group the tods we actually want into these periods
	db   = filedb.scans.select(args.sel)
	pids = np.searchsorted(periods[:,0], db.data["t"])-1
	# No tods should fall outside these periods, but a few (16 in s17,cmb) do so anyway due to
	# start/end time inconsistencies. These are few enough that we will just skip them
	good  = np.where((db.data["t"] >= periods[pids,0]) & (db.data["t"] < periods[pids,1]))[0]
	db   = db.select(good)
	pids = pids[good]
	# Group by pid and array
	atag        = np.char.replace(np.char.replace(np.char.rpartition(db.ids,".")[:,2],"ar","pa"),":","_")
	aid, arrays = names2uinds(atag, return_unique=True)
	narr        = len(arrays)
	flat_ids    = pids*narr + aid
	gvals, order, edges = utils.find_equal_groups_fast(flat_ids)
	# Loop over each such group. We will map each group, extract thumbnails and output a file for each
	for gi in range(comm_world.rank, len(gvals), comm_world.size):
		pid, aid = np.divmod(gvals[gi], narr)
		t5       = ("%.0f" % periods[pid,0])[:5]
		wdir     = root + t5 + "/"
		utils.mkdir(wdir)
		oname    = wdir + "thumbs_%.0f_%s.hdf"   % (periods[pid,0], arrays[aid])
		ename    = wdir + "thumbs_%.0f_%s.empty" % (periods[pid,0], arrays[aid])
		if args.cont and (os.path.isfile(oname) or os.path.isfile(ename)): continue
		inds     = order[edges[gi]:edges[gi+1]]
		baz, waz, bel = [db.data[name][inds[0]]*utils.degree for name in ["baz", "waz", "bel"]]
		tag      = "%4d/%d" % (gi+1, len(gvals))
		L.info("Processing %4d/%d period %4d arr %s @%.0f dur %3.0f with %2d tods" % (gi+1, len(gvals), pid, arrays[aid], periods[pid,0], periods[pid,1]-periods[pid,0], len(inds)))
		# Build our output geometry
		info = build_geometry(db.select(inds), filedb.data, res=args.res*utils.arcmin)
		if maybe_skip(info.shape[-2]*info.shape[-1] > 50e6, tag + " Unreasonably large area covered", ename): continue
		# Get the output coordinates for our sources, and find those that hit our rough area
		tmid      = np.mean(info.trange)
		spos_cel  = coordinates.transform(args.isys, "cel", spos_raw, time=[utils.ctime2mjd(tmid)], site=info.site)
		spos_flat = coordinates.transform("cel", info.sys, spos_cel, site=info.site)
		polrot    = get_polrot("cel", info.sys, spos_cel, site=info.site)
		spix      = enmap.sky2pix(info.shape, info.wcs, spos_flat[::-1])
		sinds= np.where(np.all((spix >= 0) & (spix < np.array(info.shape)[:,None]),0))[0]
		if maybe_skip(len(sinds)==0, tag + " No sources hit geometry", ename): continue
		# Find the time when each source was hit, and get rid of those were we can't recover
		# a proper effective time
		acenter_el = coordinates.recenter(info.acenter, [0,0,0,info.bel])[1]
		times      = find_obs_times(spos_cel[:,sinds], acenter_el, info.t0, info.site)
		sinds, times = sinds[np.isfinite(times)], times[np.isfinite(times)]
		if maybe_skip(len(sinds)==0, tag + " No sources after finding times", ename): continue
		# Get the pointing offset jacobian for each source. It's in terms of ora,odec for now.
		# Will be converted to pixel units later.
		jac_flat, jac_pix = build_src_jacobi(info, spos_cel[:,sinds], times)
		# ok, we have all the source geometry stuff we need. Build our map
		inds, scans = scanutils.read_scans(db.ids, inds, actscan.ACTScan, db=filedb.data, downsample=down)
		# Remove any azimuth slope from the input model, as we don't handle it here.
		# It is safe to do this because we'll measure it ourself anyway. Using a less accurate
		# input model just affects how far away we need to search, and the az slope is just a
		# 1-arcmin effect, so it won't bring them outside our search window. The standard baseline
		# pointing model doesn't have azimuth slopes anyway.
		remove_pointmodel_azslope(scans)
		if maybe_skip(len(scans)==0, tag + " Could not read any scans", ename): continue
		if args.fix:
			# Instead of actually building the map, we read in an existing thumb file, and replace
			# all its contents except the map and noise model with what we have here.
			ifname = os.path.join(args.fix, *utils.pathsplit(oname)[-2:])
			if maybe_skip(not os.path.isfile(ifname), tag + " No input data for fix %s" % ifname, ename): continue
			idata  = bunch.read(ifname)
			ps2d   = idata.ps2d
			# Build a tdata struct. Would ideally not reuse centers and pixshapes, but
			# the single precision issue I'm trying to fix here did not affect them noticably
			tinds  = utils.find(sinds, idata.table["sid"])
			tdata  = bunch.Bunch(inds=tinds, maps=idata.maps, ivars=idata.ivars,
					centers=idata.table["center"], pixshapes=idata.table["pixshape"])
		else:
			map, ivar   = build_map(info, scans, dtype=dtype, tag=tag, comm=comm_self)
			if args.maps > 0:
				if args.maps > 1:
					# Downgrade this way to avoid unhit pixel problems
					small_ivar = enmap.downgrade(ivar,     args.maps)
					small_map  = enmap.downgrade(map*ivar, args.maps)
					small_map[:,small_ivar>0] /= small_ivar[small_ivar>0]
				else: small_map, small_ivar = map, ivar
				enmap.write_map(wdir + "map_%04d_%s.fits"  % (periods[pid,0], arrays[aid]), small_map)
				enmap.write_map(wdir + "ivar_%04d_%s.fits" % (periods[pid,0], arrays[aid]), small_ivar)
			# Mask too unexposed areas. Why do I do this? Isn't it better to leave those values in
			# until after filtering in the analyse part?
			ivar      *= build_exposure_mask(ivar)
			# And extract our thumbnails
			tdata      = extract_srcs(map, ivar, spos_flat[:,sinds], tsize=tsize)
			if maybe_skip(len(tdata.inds)==0, tag + " No sources properly hit", ename): continue
			# Finally we need a noise model. There is no crosslinking and the weather and telescope state
			# probably doesn't change much over an hour. So aside from curvature, a single 2d power spectrum
			# for the whitened map is probably enough. The curvature is moderately big, but I don't think it
			# will matter much. We will mask out the brightest sources first
			bright_mask = mask_bright_srcs(info.shape, info.wcs, spos_flat[:,sinds[tdata.inds]], srcs.I[sinds[tdata.inds]])
			ps2d       = build_noise_model_simple(map, ivar*bright_mask, tsize=tsize)
			del map, ivar
		# Get the beam and frequency too
		ftag       = scans[0].entry.tag
		freq       = tag2freq[ftag]
		barea      = utils.calc_beam_area(scans[0].beam)
		fluxes     = freq_fluxes[ftag][sinds[tdata.inds]]
		# And the input pointing offsets that were used for building the maps, typically from Matthew
		input_pointoff = np.mean([scan.d.point_correction for scan in scans],0)
		# We have everything we need. Build our source information table
		table  = np.zeros(len(tdata.inds), [
			("sid", "i"), ("ctime", "d"), ("pos_cel", "2d"), ("center", "2d"), ("pixshape", "2d"),
			("polrot", "d"), ("jacobi", "2,2d"), ("ref_flux", "d")
		]).view(np.recarray)
		table.sid     = sinds[tdata.inds]               # index of srcs in input list
		table.ctime   = times[tdata.inds]               # time array center hits each
		table.pos_cel = spos_cel[:,sinds[tdata.inds]].T # celestial coords {ra,dec} of srcs
		table.center  = tdata.centers                   # tile pixel coords of src {y,x}
		table.pixshape= tdata.pixshapes                 # physical {height,width} of each pixel
		table.polrot  = polrot[sinds[tdata.inds]]       # pol angle change from cel to osys
		table.jacobi  = np.moveaxis(jac_pix,2,0)[tdata.inds] # jacobian {y,x},{focx,focy} (note the ordering!)
		table.ref_flux= fluxes
		# And put it in our result data structure
		data = bunch.Bunch(
			table = table,
			maps  = tdata.maps,
			ivars = tdata.ivars,
			ps2d  = ps2d,
			ftag  = ftag,
			freq  = freq,
			beam  = scans[0].beam,
			barea = barea,
			sysinfo = info.sysinfo,
			input_pointoff = input_pointoff,
			ids   = np.char.encode(db.ids[inds]),
			baz = baz, waz = waz, bel = bel,
		)
		# Finally write it out
		write_bunch(oname, data)
		del scans, info, tdata, table, data

elif mode == "analyse":
	# Fit pointing and flux to the thumbnail data files produced by "build".
	# The current approach is fast, but assumes that only one source is present
	# within the subset of the thumbnail selected by args.bounds. This means that
	# several sources that could otherwise have been useful had to be cut to avoid
	# outliers from matching the wrong source.
	import numpy as np, argparse, os, sys, glob, warnings
	from scipy  import ndimage
	from pixell import utils, bunch, enmap, mpi, wcsutils, uharm, analysis
	parser = argparse.ArgumentParser()
	parser.add_argument("ifiles", nargs="+")
	parser.add_argument("odir")
	parser.add_argument("-g", "--group",            type=str,   default="array")
	parser.add_argument(      "--blacklist",        type=str,   default=None)
	parser.add_argument("-s", "--snmin",            type=float, default=4)
	parser.add_argument(      "--snmin-individual", type=float, default=6)
	parser.add_argument("-S", "--sngroup",          type=float, default=8)
	parser.add_argument("-F", "--sngroup-flux",     type=float, default=15)
	parser.add_argument(      "--multi-tol",        type=float, default=0.2)
	parser.add_argument("-B", "--bounds",           type=str,   default="-6:6,-6:6",
			help="Search area bounds in arcmins. Must be at leasta beam bigger than the exlusion area used for the blacklist")
	parser.add_argument("-d", "--debug-outlier", type=str, default=None)
	parser.add_argument(      "--use-brightest", action="store_true")
	parser.add_argument(      "--flux-tol",      type=float, default=10)
	parser.add_argument(      "--dflux-tol",     type=float, default=3)
	#parser.add_argument(      "--median-filter", type=str, default=None)
	args = parser.parse_args()

	comm = mpi.COMM_WORLD
	utils.mkdir(args.odir)

	# 1. Get and group our input files
	ifiles = sorted(sum([glob.glob(ifile) for ifile in args.ifiles],[]))
	if len(ifiles) == 0:
		print("No input files given!")
		sys.exit(1)
	groups = make_groups(ifiles, grouping=args.group)
	# Get our source blacklist too. These are the sids of sources that should always be ignored.
	blacklist = read_blacklist(args.blacklist)
	#print(blacklist)
	# And the bounding box for our fit
	bounds = np.array([[float(w) for w in word.split(":")] for word in args.bounds.split(",")]).T*utils.arcmin
	flux_tol  = args.flux_tol
	dflux_tol = args.dflux_tol

	# Read in the first file to set up the output geometry
	data = bunch.read(ifiles[0])
	rmax = np.max(np.abs(np.array(data.maps.shape[-2:])*data.table["pixshape"]/2))
	res  = np.mean(np.product(data.table["pixshape"],1))**0.5 / 4
	oshape, owcs = enmap.thumbnail_geometry(r=rmax, res=res)
	del data
	uht  = uharm.UHT(oshape, owcs, mode="flat")

	# Are we debugging an outlier?
	debug = utils.parse_ints(args.debug_outlier) if args.debug_outlier else None

	indfile   = open(args.odir + "/ifits_%03d.txt" % (comm.rank), "w")
	gposfile  = open(args.odir + "/gfits_pos_%03d.txt" % (comm.rank), "w")
	gfluxfile = open(args.odir + "/gfits_flux_%03d.txt" % (comm.rank), "w")
	for gind in range(comm.rank, len(groups), comm.size):
		if debug is not None and gind != debug[0]: continue
		group = groups[gind]
		# 2. Read in the data for the files in this group. Just a few MB
		# This also calibrates and standardized the data
		gdata = []
		for ind in group:
			try:
				data = bunch.read(ifiles[ind])
				data = prepare_data(data, oshape, owcs, blacklist=blacklist)
				data.fname = ifiles[ind]
				gdata.append(data)
			except IOError as e:
				print("Error reading data for %s: %s" % (ifiles[ind], str(e)))
			except DataError as e:
				print("Error preparing data for %s: %s" % (ifiles[ind], str(e)))
		if len(gdata) == 0:
			print("No data found for group %d. Skipping" % gind)
			continue
		# 3. Compute the matched filter for each
		for i, data in enumerate(gdata):
			data.rho, data.kappa = analysis.matched_filter_constcorr_lowcorr(data.maps, data.lbeam, data.ivars[:,None], 1/data.ps2d, uht, B2=data.lbeam2, high_acc=True)
			#ivars3 = np.repeat(data.ivars[:,None],3,1)
			#ips2d  = 1/np.maximum(data.ps2d, np.max(data.ps2d)/1e20)
			#data.rho, data.kappa = analysis.matched_filter_constcorr_dual(data.maps, data.lbeam, ivars3, ips2d, uht)
			data.flux_map, data.dflux_map, data.snr_map = solve(data.rho, data.kappa)
			data.ref_dflux = data.dflux_map[:,0].at([0,0], order=1)
			#enmap.write_map("flux_map.fits", data.flux_map)
			#enmap.write_map("snr_map.fits", data.snr_map)
			#1/0

		# 4. Fit each individual source
		good_inds = []
		for i, data in enumerate(gdata):
			for sind in range(len(data.table)):
				try:
					fit = fit_group(gdata, [(i,sind)], bounds=bounds, snmin=args.snmin_individual, tol=args.multi_tol,
							use_brightest=args.use_brightest)
					# Try to prevent some outliers. I used to have no outliers, but some came back
					# after I relaxed the masking.
					weird_snr  = fit.snr > fit.model_snr * flux_tol
					weird_flux = fit.flux[0,0] > fit.table["ref_flux"][0] * flux_tol
					weird_dflux= fit.dflux[0,0] > fit.ref_dflux * dflux_tol
					#print("weird_snr   %d fit.snr   %10.4f model_snr %10.4f tol %8.5f" % (weird_snr, fit.snr, fit.model_snr, flux_tol))
					#print("weird_flux  %d fit.flux  %10.4f ref_flux  %10.4f tol %8.4f" % (weird_flux, fit.flux[0,0], fit.table["ref_flux"][0], flux_tol))
					#print("weird_dflux %d fit.dflux %10.4f ref_dflux %10.4f tol %8.4f" % (weird_dflux, fit.dflux[0,0], fit.ref_dflux, dflux_tol))

					if (weird_snr or weird_flux) and weird_dflux:
						raise FitUnexpectedFluxError()
					# Write individual fit to file
					flux, dflux = fit.flux[0], fit.dflux[0]
					msg  = "%.0f %5d %4s" % (fit.ctime, fit.table["sid"][0], fit.ftags[0])
					msg += "  %7.2f  %7.4f %7.4f  %6.4f %6.4f" % (fit.snr, *-fit.pos/utils.arcmin, *fit.dpos/utils.arcmin)
					msg += "  %8.2f  %8.2f %7.2f %8.2f %7.2f %8.2f %7.2f" % (fit.table["model_flux"][0],
							flux[0], dflux[0], flux[1], dflux[1], flux[2], dflux[2])
					msg += "  %7.2f %6.2f %7.2f  %7.2f %7.2f %7.2f" % (fit.baz/utils.degree, fit.waz/utils.degree, fit.bel/utils.degree,
							fit.az/utils.degree, fit.ra/utils.degree, fit.dec/utils.degree)
					msg += "  %5.2f %4d %d %4d" % (fit.ctime/3600%24, gind, i, sind)
					indfile.write(msg + "\n")
					# Update our model flux
					data.table["model_flux"][sind] = flux[0]
					data.ref_dflux[sind] = dflux[0]
				except FitMultiModalError:
					sys.stderr.write("Multimodal distribution for group %d file %d %s srcind %d id %d. Skipping\n" % (
						gind, i, data.fname, sind, data.table["sid"][sind]))
					continue
				except FitUnexpectedFluxError:
					sys.stderr.write("Unexpected flux for group %d file %d %s srcind %d id %d. Skipping\n" % (
						gind, i, data.fname, sind, data.table["sid"][sind]))
					continue
				except FitNonDetectionError: pass
				good_inds.append((i,sind))
		indfile.flush()
		if len(good_inds) == 0:
			sys.stderr.write("No usable fits in group %d file %d %s. Skipping\n" % (
				gind, i, data.fname))
			continue
		# 5. Group sources into groups with decent S/N while keeping strong sources separate
		src_groups = group_srcs_snmin(gdata, good_inds, snmin=args.sngroup)
		if args.debug_outlier: debug_outlier(gdata, src_groups[debug[1]], bounds=bounds)
		# 6. Fit each of these groups
		for sgi, src_group in enumerate(src_groups):
			try:
				fit = fit_group(gdata, src_group, bounds=bounds, snmin=args.snmin, use_brightest=args.use_brightest)
				# Write group fits to file. Two different outputs: Positions, which are per-group,
				# and fluxes, which are per-source. First the per-group stuff. This is similar to the
				# individual fit format, but without the fluxes.
				src_tags = ",".join([str(sid)+":"+ftag for sid, ftag in zip(fit.table["sid"], fit.ftags)])
				msg  = "%.0f %3.0f" % (fit.ctime, fit.dctime)
				msg += "  %7.2f  %6.3f %6.3f  %6.3f %6.3f" % (fit.snr, *-fit.pos/utils.arcmin, *fit.dpos/utils.arcmin)
				msg += "  %7.2f %6.2f %7.2f  %7.2f %5.2f %7.2f %4.2f %7.2f %5.2f" % (
						fit.baz/utils.degree, fit.waz/utils.degree, fit.bel/utils.degree,
						fit.az/utils.degree, fit.daz/utils.degree,
						fit.ra/utils.degree, fit.dra/utils.degree,
						fit.dec/utils.degree, fit.ddec/utils.degree)
				msg += "  %5.2f %4d %2d %s" % (fit.ctime/3600%24, gind, sgi, src_tags)
				print("%3d %s" % (comm.rank, msg))
				gposfile.write(msg + "\n")
			except FitMultiModalError:
				sys.stderr.write("Multimodal distribution for group %d file %d %s pos-src-group %d. Skipping\n" % (
					gind, i, data.fname, sgi))
				continue
			except FitNonDetectionError: pass
		gposfile.flush()
		# 7. Redo the fit with higher S/N thresholds for the flux measurements. We do this to reduce noise bias
		src_groups = group_srcs_snmin(gdata, good_inds, snmin=args.sngroup_flux)
		for sgi, src_group in enumerate(src_groups):
			try:
				fit = fit_group(gdata, src_group, bounds=bounds, snmin=args.snmin, use_brightest=args.use_brightest)
				for i in range(len(fit.table)):
					flux, dflux = fit.flux[i], fit.dflux[i]
					msg  = "%.0f %5d %4s" % (fit.table["ctime"][i], fit.table["sid"][i], fit.ftags[i])
					msg += "  %7.2f %7.2f  %6.3f %6.3f" % (flux[0]/dflux[0], fit.snr, *-fit.pos/utils.arcmin)
					msg += "  %8.2f  %8.2f %7.2f %8.2f %7.2f %8.2f %7.2f" % (fit.table["ref_flux"][i],
							flux[0], dflux[0], flux[1], dflux[1], flux[2], dflux[2])
					msg += "  %7.2f %6.2f %7.2f  %7.2f %7.2f %7.2f" % (fit.baz/utils.degree, fit.waz/utils.degree, fit.bel/utils.degree,
						fit.table["az"][i]/utils.degree, *fit.table["pos_cel"][i]/utils.degree)
					msg += "  %5.2f %4d %d %4d" % (fit.table["ctime"][i]/3600%24, gind, fit.inds[i][0], fit.inds[i][1])
					gfluxfile.write(msg + "\n")
			except FitMultiModalError:
				sys.stderr.write("Multimodal distribution for group %d file %d %s flux-src-group %d. Skipping\n" % (
					gind, i, data.fname, sgi))
				continue
			except FitNonDetectionError: pass
		gfluxfile.flush()

	indfile.close()
	gposfile.close()
	gfluxfile.close()

	comm.Barrier()

	# Ok, we're fully done. Merge the individual files and sort them by time
	if comm.rank == 0:
		merge_files([args.odir + "/ifits_%03d.txt"      % i for i in range(comm.size)], args.odir + "/ifits.txt")
		merge_files([args.odir + "/gfits_pos_%03d.txt"  % i for i in range(comm.size)], args.odir + "/gfits_pos.txt")
		merge_files([args.odir + "/gfits_flux_%03d.txt" % i for i in range(comm.size)], args.odir + "/gfits_flux.txt")
		sys.stderr.write("Done\n")

elif mode == "model":
	# Split the data points output from "analyse" into segments with the same scanning patterns and
	# not too large gaps in time. Remove any outliers, and model the points as a slowly changing
	# function in time and a gradient in azimuth.
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("mode", help="model")
	parser.add_argument("gfit_pos")
	parser.add_argument("odir")
	parser.add_argument("-e", "--minerr",    type=float, default=0.01)
	parser.add_argument(      "--maxdur",    type=float, default=10)
	args = parser.parse_args()
	import numpy as np
	from pixell import enmap, utils, mpi, bunch
	from scipy import ndimage

	comm = mpi.COMM_WORLD
	utils.mkdir(args.odir)
	maxgap = 360**2
	maxdur = 3600*args.maxdur
	outlier_sigma = 8

	# Read in the analysis result.
	# ctime, dt, snr, y, x, dy, dx, baz, waz, bel, az, daz, ra, dra, dec, ddec
	data = np.loadtxt(args.gfit_pos, ndmin=2, usecols=list(range(16))).T

	# It should be sorted, but make sure anyway
	order = np.argsort(data[0])
	data  = data[:,order]
	# Avoid az jumps
	data[7] = utils.rewind(data[7], ref=0, period=360)

	# Split into groups with edges where ctime, baz, waz or bel change too much
	groups = build_groups(data[[0,7,8,9]], [maxgap,1,1,1])
	groups = split_long_groups(groups, data[0], maxdur=maxdur)

	# Add a minimum error value to each point to avoid having individual points
	# completely dominating. This represents the sort of model error we expect.
	data[5:7] = (data[5:7]**2 + args.minerr**2)**0.5

	modpointfile = open(args.odir + "/modpoints.txt", "w")
	modelfile    = open(args.odir + "/models.txt", "w")
	# Process each group
	for gind in range(comm.rank, len(groups), comm.size):
		i1, i2 = groups[gind]
		gdata  = data[:,i1:i2]
		# Remove outliers
		bad    = classify_outliers(gdata[3:5], gdata[5:7], nsigma=outlier_sigma)
		gdata  = gdata[:,~bad]
		if gdata.size == 0:
			continue
		#if len(gdata[0]) < 6 or gdata[0][5] < 1600896096.0-2000: continue
		model  = fit_model_piecewise(pos=gdata[3:5], dpos=gdata[5:7], t=gdata[0], az=gdata[10], az0=gdata[7,0])
		# Output the model parameters themselves.
		# group index, ctime0, dur, ngood, nbad, baz, waz, bel, nparam
		msg1 = "%4d %.0f %5.0f %3d %3d %8.3f %8.3f %8.3f %2d" % (gind, gdata[0,0], gdata[0,-1]-gdata[0,0],
				np.sum(~bad), np.sum(bad), gdata[7,0], gdata[8,0], gdata[9,0], model.a.shape[1])
		# Then the actual parameters.
		msg2 = ""
		omodel = normalize_piecewise_model(model)
		nparam = omodel.a.shape[-1]
		for i in range(nparam):
			# Normal parameters are in arcmin, which is what we want them as. But if nparam > 1, then the
			# last parameter is the az gradient, which is in arcmin per degree. This is a small number, so it
			# needs more precision.
			if i < nparam-1:
				# t y dy x dx
				msg2 += (" %5.0f" + " %8.3f"*4) % (omodel.ts[i]-omodel.t0, omodel.a[0,i], omodel.da[0,i], omodel.a[1,i], omodel.da[1,i])
			else:
				# y dy/daz err dz/daz err
				msg2 += (" %8.5f"*4) % (omodel.a[0,i], omodel.da[0,i], omodel.a[1,i], omodel.da[1,i])
		# Some summary stuff for the screen print
		tot_snr   = np.sum(gdata[2]**2)**0.5
		agrad_snr = np.sum(np.maximum(0,(omodel.a[:,-1]/np.maximum(omodel.da[:,-1],1e-10))**2-1))**0.5
		msg3 = " %2d %6.1f %6.2f" % (nparam, tot_snr, agrad_snr)
		print(msg1 + msg3)
		modelfile.write(msg1 + msg2 + "\n")

		# Output the fit model evaluated at each data point
		a0   = model.a.copy()
		if model.azslope:
			a0[:,-1] = 0 # No az dependence
		mod0 = np.einsum("ai,ya->yi",model.B,a0)
		for i, d in enumerate(gdata.T):
			msg = "%4d %3.0f %.0f  %8.3f %8.3f  %8.3f %8.3f" % (
					gind, 100*i/gdata.shape[1], d[0],
					model.model[0,i], model.model[1,i],
					mod0[0,i], mod0[1,i])
			modpointfile.write(msg + "\n")

		modelfile.flush()
		modpointfile.flush()

	modpointfile.close()
	modelfile.close()

elif mode == "remodel":
	# Read in a model and some new data points, and spit out a new model that's the
	# old model, but modified by those data points. The use case for this is when we have
	# an array with only low-S/N measurements and so can't build a decent model on our own,
	# and instead base it on the model from a better array.
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("mode", help="remodel")
	parser.add_argument("imodel")
	parser.add_argument("gfit_pos")
	parser.add_argument("odir")
	args = parser.parse_args()
	import numpy as np
	from pixell import enmap, utils, mpi, bunch
	from scipy import ndimage

	comm = mpi.COMM_WORLD
	utils.mkdir(args.odir)
	snr_cap       =  8
	snr_target    = 50
	outlier_sigma =  8
	min_dpos      = 0.05

	# Read in the input model. has t0s[:] and segments[{t0,dur,baz,npoint[n,2],ts[n],points[n,2,2],slope[2,2]}]
	model = read_model(args.imodel)

	# Read in the analysis result.
	# ctime, dt, snr, y, x, dy, dx, baz, waz, bel, az, daz, ra, dra, dec, ddec
	data = np.loadtxt(args.gfit_pos, ndmin=2, usecols=list(range(16))).T

	# It should be sorted, but make sure anyway
	order = np.argsort(data[0])
	data  = data[:,order]
	# Avoid az jumps
	data[7] = utils.rewind(data[7], ref=0, period=360)

	inds = find_segments(model.t0s, model.durs, data[0])
	vals, order, edges = utils.find_equal_groups_fast(inds)
	for i, ind in enumerate(vals):
		if ind < 0: continue
		model.segments[ind].point_inds = order[edges[i]:edges[i+1]]
	del inds, vals, order, edges

	# The model consists of segments, but a single segment may be too short
	# for a meaningful fit, so we will fit multiple segments jointly. Split
	# the segments into sufficently large groups. Start by counting the capped
	# total snr of the points in each segment
	for seg in model.segments:
		if "point_inds" not in seg: seg.point_inds = []
		seg.point_snr_capped = np.sum(np.minimum(data[2,seg.point_inds], snr_cap)**2)**0.5

	snrs  = np.array([seg.point_snr_capped for seg in model.segments])
	edges = group_cumulative(snrs**2, snr_target**2)

	# For each group we will fit an offset between the arrays. The az slope could
	# also be different, but we'll keep it the same for now.
	ogind     = 0
	modelfile = open(args.odir + "/models.txt", "w")
	for gind in range(len(edges)-1):
		i1, i2 = edges[gind:gind+2]
		# Get all the matching data points
		dinds = np.concatenate([seg.point_inds for seg in model.segments[i1:i2]]).astype(int)
		gdata = data[:,dinds]
		# Get our model parameters. Embarassingly unvectorized
		pred  = bunch.concatenate([evaluate_model(seg, data[0,i]) for seg in model.segments[i1:i2] for i in seg.point_inds])
		# To just fit for a pointing offset, we subtract of the az part and then get
		# the weighted mean of the results.
		daz   = gdata[10]-pred.baz
		azcorr= pred.slope[:,:,:] * daz[:,None,None]
		dposs = gdata[[3,4]] - azcorr[:,:,0].T - pred.point[:,:,0].T
		ddposs= (gdata[[5,6]]**2 + azcorr[:,:,1].T**2 + pred.point[:,:,1].T**2)**0.5
		bad   = classify_outliers(dposs, ddposs, nsigma=outlier_sigma)
		dposs, ddposs = dposs[:,~bad], ddposs[:,~bad]
		# Skip if we're empty
		if dposs.size == 0: continue
		# Enforce minium position uncertainty
		ddposs = np.maximum(ddposs, min_dpos)
		# Get the mean offset
		dpos  = np.sum(dposs*ddposs**-2,1)
		ddpos = np.sum(ddposs**-2,1)**-0.5
		dpos /= ddpos**-2
		# Update the model with this offset
		for seg in model.segments[i1:i2]:
			if len(seg.point_inds) == 0: continue
			seg.points[:,:,0] += dpos
			seg.points[:,:,1]  = (seg.points[:,:,1]**2 + ddpos**2)**0.5
			tot_snr = np.sum(data[2,seg.point_inds]**2)**0.5
			# Output the model
			j = seg.point_inds[0]
			msg1 = "%4d %.0f %5.0f %3d %3d %8.3f %8.3f %8.3f %2d" % (ogind, seg.t0, seg.dur,
					len(seg.point_inds), 0, seg.baz, seg.waz, seg.bel, len(seg.ts))
			# Then the actual parameters.
			msg2 = ""
			for i in range(len(seg.ts)):
				# t y dy x dx
				msg2 += (" %5.0f" + " %8.3f"*4) % (seg.ts[i], *seg.points[i].reshape(-1))
			# y dy/daz err dz/daz err
			msg2 += (" %8.5f"*4) % tuple(seg.slope.reshape(-1))
			# Some summary stuff for the screen print
			agrad_snr = np.sum((seg.slope[:,0]/np.maximum(seg.slope[:,1],1e-10))**2)**0.5
			modelfile.write(msg1 + msg2 + "\n")
			modelfile.flush()
			ogind += 1
	modelfile.close()

elif mode == "calfile":
	# Given the model files and a tod selector, output the new pointing model
	import numpy as np, warnings, time, h5py, os, sys
	from enlib  import config, bench, log
	from pixell import utils, bunch
	from enact  import filedb, actdata, files
	parser = config.ArgumentParser()
	parser.add_argument("mode", help="calfiles")
	parser.add_argument("sel")
	parser.add_argument("modelfiles", nargs="+", help="array:filename")
	parser.add_argument("odir")
	parser.add_argument("-e", "--maxerr-other",  type=float, default=0.2)
	parser.add_argument("-E", "--maxerr-giveup", type=float, default=2)
	args = parser.parse_args()

	# Parse our model files
	models = {}
	for fi, modelfile in enumerate(args.modelfiles):
		arr, fname = modelfile.split(":")
		model      = read_model(fname)
		models[arr]= model

	# Get all our scans
	filedb.init()
	db  = filedb.scans.select(args.sel)
	ids = db.ids
	uids, uinds = np.unique(np.char.partition(ids, ":")[:,0], return_index=True)

	array_offsets = build_typical_array_offsets(models)

	# Our baseline offsets
	baseline_reader = OffsetCache(filedb.data)

	utils.mkdir(args.odir)
	opointfile = open(args.odir + "/pointing_offsets.txt", "w")
	oslopefile = open(args.odir + "/pointing_slopes.txt",  "w")
	ofullfile  = open(args.odir + "/pointing_full.txt",    "w")

	for ind, uid in zip(uinds, uids):
		# Find which model we belong to
		arr   = uid.split(".")[-1].replace("ar","pa")
		t     = db.data["t"][ind]
		bel   = db.data["bel"][ind]
		# Handle cases where the obs time is missing by inferring it
		# from the timestamp
		if not np.isfinite(t):
			t = float(uid.split(".")[0])+300
		vals   = get_prediction(models[arr], t)
		status = arr
		# Are we good enough?
		err   = np.max(vals.point[:,1])
		if err > args.maxerr_other:
			# Too big error. Try the values from another array. They may not match perfectly, but
			# better than just defaulting to zero
			arr_vals = [[arr, vals, err]]
			for arr2, model2 in models.items():
				if arr2 != arr:
					vals2 = get_prediction(model2, t)
					arr_vals.append([arr2, vals2, np.max(vals2.point[:,1])])
			best = np.argmin([v[2] for v in arr_vals])
			arr2, vals2, err2 = arr_vals[best]
			if err2 <= args.maxerr_giveup:
				# Another array was good enough. Use it, but take into accoun the
				# typical offset between the detectors
				offs = get_array_offsets(array_offsets, t)
				vals2.point[:,0] += offs[arr]-offs[arr2]
				status, vals, err = arr2, vals2, err2
			else:
				# No array good enough. Use zero
				vals.point[:,0] = 0
				vals.slope[:,0] = 0
				status = "bad"

		# And prepare to output. For this we need the baseline pointing model too
		baseline = baseline_reader.get(uid)[::-1] / utils.arcmin
		vals.point[:,0] += baseline
		# Output standard pointing file supports the format
		# id baz bel x_arcmin y_arcmin x_rad y_rad x_err y_err desc
		# Will use that for compatibility
		msg = "%s %8.3f %8.3f %8.3f %8.3f %12.8f %12.8f %8.3f %8.3f %s" % (
			uid, vals.baz, bel, vals.point[1,0], vals.point[0,0],
			vals.point[1,0]*utils.arcmin, vals.point[0,0]*utils.arcmin,
			vals.point[1,1], vals.point[0,1],
			status)
		opointfile.write(msg + "\n")

		# Output paired azimuth slope file. This will have format
		# id az0 yslope xslope dyslope dxslope extrap status 
		msg = "%s %8.3f %8.5f %8.5f %8.5f %8.5f %6.2f %s" % (
				uid, vals.baz,
				vals.slope[1,0], vals.slope[0,0],
				vals.slope[1,1], vals.slope[0,1],
				vals.extrap/3600,
				status)
		oslopefile.write(msg + "\n")

		# Also output a file with everything, and some extra stuff:
		# id az0 x_arcmin y_arcmin xslope_arcmin_deg yslope_arcmin_deg [+uncertainties]
		msg = "%s %8.3f  %8.3f %8.3f   %8.5f %8.5f   %8.3f %8.3f   %8.5f %8.5f  %8.3f %8.3f    %6.2f %s" % (
			uid, vals.baz,
			vals.point[1,0], vals.point[0,0], # x,y order
			vals.slope[1,0], vals.slope[0,0],
			vals.point[1,1], vals.point[0,1], # uncertainty
			vals.slope[1,1], vals.slope[0,1],
			vals.point[1,0]-baseline[1], # offsets without baseline added in
			vals.point[0,0]-baseline[0],
			vals.extrap/3600, status,
			)
		ofullfile.write(msg + "\n")

	opointfile.close()
	oslopefile.close()
	ofullfile.close()

elif mode == "stack":
	# Stack thumbnails in time-bins per array using the position and flux fits
	# from analyse mode
	import numpy as np, argparse, os, sys, glob, warnings
	from scipy  import ndimage
	from pixell import utils, bunch, enmap, mpi, wcsutils, uharm, analysis
	import numpy.lib.recfunctions as rfn
	parser = argparse.ArgumentParser()
	parser.add_argument("ifiles", nargs="+", help="arr:gfits_pos1 arr:gfits_pos2 ... thumb1 thumb2 thumb3 ...")
	parser.add_argument("odir")
	parser.add_argument("-T", "--tstep",     type=float, default=3600)
	parser.add_argument(      "--blacklist", type=str,   default=None)
	parser.add_argument("-N", "--max-rms",   type=float, default=0.20)
	parser.add_argument(      "--time-tol",  type=float, default=10)
	parser.add_argument("-l", "--lknee",     type=float, default=1500)
	parser.add_argument("-a", "--alpha",     type=float, default=-3)
	parser.add_argument("-r", "--res",       type=float, default=0.5)
	parser.add_argument("-R", "--rmax",      type=float, default=8)
	parser.add_argument(      "--nline",     type=int,   default=None)
	args = parser.parse_args()

	def organize_ifiles_thumbs(ifiles):
		"""Given a list of thumbnail files, returns a dictionary
		with a key for each array-freq (atag) combination. The value
		for each of these will be a bunch with members .ctimes and .fnames.
		.ctimes is an array of the starting ctime for each, and .fnames
		has the corresponding file file names copied from ifiles.
		The order is sorted by ctime."""
		res = {}
		for ifile in ifiles:
			toks = os.path.basename(ifile).split(".")[0].split("_")
			ctime = float(toks[1])
			atag  = "_".join(toks[2:4])
			if atag not in res:
				res[atag] = bunch.Bunch(ctimes=[], fnames=[])
			res[atag].ctimes.append(ctime)
			res[atag].fnames.append(ifile)
		# Convert from lists to arrays, and sort
		for atag in res:
			for key in res[atag]:
				res[atag][key] = np.array(res[atag][key])
			order = np.argsort(res[atag].ctimes)
			for key in res[atag]:
				res[atag][key] = res[atag][key][order]
		return res

	def apply_filter(map, ivar, filter, tol=1e-4, ref=0.9):
		rhs = enmap.ifft(enmap.fft(map*ivar)*(1-filter)).real
		div = enmap.ifft(enmap.fft(    ivar)*(1-filter)).real
		ref = np.percentile(div, ref*100,(-2,-1))[...,None,None]*tol
		ref[ref<=0] = 1
		div = np.maximum(div, ref)
		bad = rhs/div
		omap = (map-bad)*(ivar>0)
		return omap

	def find_close(a, b, tol=1):
		"""Find matches between the (1d) lists a and b that are closer than tol.
		Returns [{ind_a, ind_b}]"""
		a = np.asarray(a)
		b = np.asarray(b)
		pairs = utils.crossmatch(a[:,None],b[:,None],rmax=tol,mode="closest")
		return [(ia,ib) for ia,ib in pairs if abs(a[ia]-b[ib])<=tol]

	def sort_atags(atags):
		"""Sort atags by freq first, then by array"""
		atags = np.array(atags)
		parts = np.char.partition(atags,"_")
		arrs  = parts[:,0]
		ftags = parts[:,2]
		order = np.lexsort([arrs, ftags])
		return atags[order]

	gfits_flux_dtype = [
			("ctime","d"),("sid","i"),("ftag","U4"),("snr","d"),("tot_snr","d"),
			("y","d"),("x","d"),("ref_flux","d"),("Tflux","d"),("dTflux","d"),
			("Qflux","d"),("dQflux","d"),("Uflux","d"),("dUflux","d"),
			("baz","d"),("waz","d"),("bel","d"),("az","d"),("ra","d"),("dec","d"),
			("hour","d"),("gind","i"),("ind1","i"),("ind2","i")]

	dtype= np.float32
	comm = mpi.COMM_WORLD
	utils.mkdir(args.odir)

	# Get our source blacklist too. These are the sids of sources that should always be ignored.
	blacklist = read_blacklist(args.blacklist)

	ifiles_fits   = []
	ifiles_thumbs = []
	for ifile in args.ifiles:
		if ":" in ifile:
			ifiles_fits.append(ifile)
		else:
			ifiles_thumbs.append(sorted(glob.glob(ifile)))
	ifiles_thumbs = sum(ifiles_thumbs,[])

	# Read in the first file to set up the output geometry
	oshape, owcs = enmap.thumbnail_geometry(r=args.rmax*utils.arcmin, res=args.res*utils.arcmin)

	# Set up our simple filter
	l = enmap.modlmap(oshape, owcs).astype(dtype)
	filter = (1+(np.maximum(l,0.5)/args.lknee)**args.alpha)**-1

	# Organize the thumbnails by array and time, so that we can quickly
	# look up files by time.
	ifiles_thumbs = organize_ifiles_thumbs(ifiles_thumbs)

	# Read in all our fits and merge them into a single array after
	# annotating them with the array
	fits  = []
	atags = []
	for ifile in ifiles_fits:
		aname, fname = ifile.split(":")
		data = np.loadtxt(fname, dtype=gfits_flux_dtype, max_rows=args.nline)
		data = rfn.append_fields(data, ["arr"], [np.full(len(data), aname, dtype="U3")])
		ftags= np.unique(data["ftag"])
		fits.append(data)
		for ftag in ftags:
			atags.append("%s_%s" % (aname, ftag))
	fits = np.concatenate(fits)
	fits = fits[np.argsort(fits["ctime"])]
	fits["baz"] = utils.rewind(fits["baz"], ref=0, period=360)
	atags = np.unique(atags)
	atags = sort_atags(atags)

	# Remove the blacklisted sources
	if blacklist is not None:
		fits = fits[~utils.contains(fits["sid"], blacklist)]

	# Split into groups with edges where ctime, baz, waz or bel change too much
	groups = build_groups([fits["ctime"],fits["baz"],fits["waz"],fits["bel"]],[args.tstep,1,1,1])
	groups = split_long_groups(groups, fits["ctime"], maxdur=args.tstep)
	if comm.rank == 0: print("Processing %d groups" % len(groups))

	# Process each group
	for gi in range(comm.rank, len(groups), comm.size):
		group = groups[gi]
		gfits = fits[group[0]:group[1]]
		# Get all thumbnail files that cover this time period
		t1    = gfits["ctime"][0]
		t2    = gfits["ctime"][-1]
		opre  = args.odir + "/stack_%.0f" % t1
		omaps, oivars = enmap.zeros((2,len(atags),)+oshape, owcs, dtype)
		used_fits = []
		for ai, atag in enumerate(atags):
			print("Processing group %4d array %s" % (gi, atag))
			arr, ftag = atag.split("_")
			afits = gfits[(gfits["arr"] == arr)&(gfits["ftag"]==ftag)]
			# For each array-freq (atag) we will build the S/N-weighted
			# stack of all the sources
			rhs = enmap.zeros(oshape, owcs, dtype)
			div = enmap.zeros(oshape, owcs, dtype)
			# Go through all the thumbnails for this array
			finfo = ifiles_thumbs[atag]
			i1 = np.searchsorted(finfo.ctimes, t1-args.time_tol)
			i2 = np.searchsorted(finfo.ctimes, t2+args.time_tol)+1
			for fname in finfo.fnames[i1:i2]:
				data    = bunch.read(fname)
				# Get the thumbnails that correspond to entries in gfits
				matches = find_close(afits["ctime"], data.table["ctime"], tol=1)
				mfits   = afits[[i1 for i1,i2 in matches]]
				data    = select_srcs(data, [i2 for i2,i2 in matches])
				# mfits and data are now in the same order. Next reproject while
				# applying the pointing offsets from mfits
				pointoff= np.array([mfits["y"],mfits["x"]]).T*utils.arcmin
				#pointoff = None
				try:
					data  = prepare_data(data, oshape, owcs, pointoff=pointoff)
				except DataError:
					continue
				# Ok, we can now do our filtering and stacking. Only keep T. For the stacking
				# we need the estimated signal strength. This comes as flux but maps are in flux
				# per steradian. Make the units compatible by converting the flux to that too
				T       = mfits["Tflux"] / data.barea
				maps    = apply_filter(data.maps[:,0,:,:], data.ivars, filter) / T[:,None,None]
				ivars   = data.ivars * T[:,None,None]**2
				rhs    += np.sum(maps*ivars,0)
				div    += np.sum(ivars,0)
				# Collect mfits so that we can output the mean properties later
				used_fits.append(mfits)
			# Phew! We've processed all the data for this atag in this group. Solve
			with utils.nowarn():
				map = rhs/div
				map[~np.isfinite(map)] = 0
			# And copy into our output file
			omaps[ai]  = map
			oivars[ai] = div
		if len(used_fits) == 0: continue
		# Count how many atags we have with sufficient snr. We will put this in the
		# output file name so that it's easier to avoid plotting empty or overly noisy
		# images
		mean_ivar = np.mean(oivars,(-2,-1))
		ngood     = np.sum(mean_ivar > args.max_rms**-2)
		# Group is done. Write
		enmap.write_map(opre + "_map_%d.fits" % ngood,  omaps)
		# Output mean scan properties in group
		used_fits = np.concatenate(used_fits)
		with open(opre + "_info.txt", "w") as ofile:
			# ctime snr y x baz waz bel hour n
			weight = (used_fits["Tflux"]/used_fits["dTflux"])**2
			def mean(a): return np.sum(a*weight)/np.sum(weight)
			def slope(t,a):
				dt = (t - mean(t))/3600
				B  = np.array([dt*0+1,dt])
				W  = 1/np.maximum(1/weight,np.max(1/weight)*0.1)
				rhs=np.sum(B*a*W,-1)
				div=np.sum(B[:,None]*B[None,:]*W,-1)
				try: amps = np.linalg.solve(div,rhs)
				except np.LinAlgError: amps = np.zeros(2)
				return amps[1]
			ofile.write("%.0f %.0f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %5.2f %2d %4d\n" % (
				t1, mean(used_fits["ctime"]),
				max(0,np.sum(weight)-len(weight))**0.5,
				mean (used_fits["y"]), mean (used_fits["x"]),
				slope(used_fits["ctime"], used_fits["y"]),
				slope(used_fits["ctime"], used_fits["x"]),
				mean(used_fits["baz"]), mean(used_fits["waz"]), mean(used_fits["bel"]),
				(mean(used_fits["ctime"])/3600)%24,
				ngood, len(weight)))

elif mode == "debug":
	import numpy as np, warnings, time, h5py, os, sys
	from enlib  import config, coordinates, mapmaking, bench, scanutils, log, cg, dory, pmat, array_ops, sampcut
	from pixell import utils, enmap, pointsrcs, bunch, mpi, uharm, analysis, wcsutils, fft
	from enact  import filedb, actdata, actscan, files
	from scipy  import ndimage, optimize, spatial

	config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
	config.default("downsample", 1, "Factor with which to downsample the TOD")
	config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
	config.default("tod_window", 5.0, "Number of samples to window the tod by on each end")
	config.default("eig_limit", 0.1, "Pixel condition number below which polarization is dropped to make total intensity more stable. Should be a high value for single-tod maps to avoid thin stripes with really high noise")
	config.set("cut_obj", "Saturn:-0.5")

	parser = config.ArgumentParser()
	parser.add_argument("mode", help="debug")
	parser.add_argument("odir", help="Output directory")
	parser.add_argument("-r", "--res",    type=float, default=0.5, help="Resolution in arcmins")
	parser.add_argument("-R", "--rad",    type=float, default=30,  help="Thumbnail radius in arcmins")
	parser.add_argument("-B", "--bounds", type=str,   default="-6:6,-6:6")
	args = parser.parse_args()

	utils.mkdir(args.odir)
	dtype = np.float32 if config.get("map_bits") == 32 else np.float64
	ncomp = 3
	tsize = int(np.ceil(2*args.rad/args.res))
	root  = args.odir + "/"
	down  = config.get("downsample")
	comm  = mpi.COMM_WORLD
	ftag  = "f090"
	freq  = 98

	# Set up our simulated src
	#spos    = np.array([[-73.90014225],[-22.34030722]])*utils.degree
	#spos    = np.array([[-93.84146271], [-22.01905534]])*utils.degree
	spos     = np.array([[ -93.66632134],[ -22.02386303]])*utils.degree
	fluxes = np.array([10e3])

	# We will debug this tod
	#id = "1565329097.1565340166.ar5:f090"
	#id = "1494733123.1494748796.ar5:f090"
	id = "1494470754.1494482280.ar6:f090"
	filedb.init()
	db = filedb.scans.select(id)

	simulate  = False
	sim_error = np.array([2,1])*utils.arcmin

	# Build our output geometry
	info = build_geometry(db.select(id), filedb.data, res=args.res*utils.arcmin)
	# Get the output coordinates for our sources, and find those that hit our rough area
	spos_flat = coordinates.transform("cel", info.sys, spos, site=info.site)
	polrot    = get_polrot("cel", info.sys, spos, site=info.site)
	# Find the time when each source was hit
	acenter_el = coordinates.recenter(info.acenter, [0,0,0,info.bel])[1]
	times      = find_obs_times(spos, acenter_el, info.t0, info.site)
	# Get the pointing offset jacobian for each source. It's in terms of ora,odec for now.
	# Will be converted to pixel units later.
	jac_flat, jac_pix = build_src_jacobi(info, spos, times)
	
	# Set up our true tod, in the case that we're simulating
	if simulate:
		print("Sim signal")
		data_true   = actdata.read(filedb.data[id], exclude=["tod"])
		data_true.point_correction += sim_error
		data_true.point_offset[:] = data_true.point_template+data_true.point_correction
		data_true   = actdata.calibrate(data_true)
		scan_true   = actscan.ACTScan(filedb.data[id], d=data_true)
		zero        = amps*0
		srcs        = np.array([spos[1],spos[0],amps,zero,zero,zero+1,zero+1,zero]).T
		pmat_srcs   = pmat.PmatPtsrc(scan_true, srcs)
		tod_sim     = np.zeros((scan_true.ndet,scan_true.nsamp),dtype)
		pmat_srcs.forward(tod_sim, srcs)
		del data_true, scan_true, pmat_srcs, srcs

	# The input az slope is not taken into account when making the
	# new model. This is the source of all the issues. Solutions:
	# 1. Force the slope to be zero for the input pointing
	# 2. Find a way to propagate the input slope into the output.
	# The slope is just O(1 arcmin), so forcing it to zero should not
	# make things so bad that the sources can't be found. So let's try
	# #1.

	# We will loop over the analysis twice. First we stard from our
	# fiducial pointing and measure the pointing error. Then we
	# use that as the new pointing correction and check what the new
	# pointing measurement is after applying that

	point_slope = np.zeros(3)
	point_corr  = np.zeros(2)
	#point_corr += np.array([-0.00012, -0.01958])*utils.arcmin
	#point_corr  += np.array([ 0.00039, -0.02376])*utils.arcmin
	#point_corr += np.array([-0.46876, -0.09063])*utils.arcmin

	for iround in range(2):
		idata       = actdata.read(filedb.data[id], exclude=["tod"])
		idata.point_correction += point_corr
		print("Using full point_corr %8.5f %8.5f" % (idata.point_correction[0]/utils.arcmin, idata.point_correction[1]/utils.arcmin))
		idata.point_offset[:] = idata.point_template+idata.point_correction
		# Override point slope
		if "point_slope" in idata: idata.point_slope[:] = point_slope
		idata       = actdata.calibrate(idata)
		scan_fid    = actscan.ACTScan(filedb.data[id], d=idata)
		barea       = utils.calc_beam_area(scan_fid.beam)
		flux_factor = utils.flux_factor(barea, freq*1e9)
		amps        = fluxes/flux_factor
		if simulate:
			tod = tod_sim.copy()
		else:
			pmat_cut  = pmat.PmatCut(scan_fid)
			tod       = scan_fid.get_samples()
			utils.deslope(tod, inplace=True)
			tod       = tod.astype(dtype)
			f = (np.arange(tod.shape[1]//2+1)*scan_fid.srate/tod.shape[1]).astype(dtype)
			print(f)
			filter = (1+((f+f[1])/1.0)**-3.5)**-1
			print(filter)
			ftod = fft.rfft(tod)*filter
			fft.irfft(ftod, tod)
			del ftod, filter
		# Build a noiseless binned map
		print("Build rhs")
		area        = enmap.zeros((ncomp,)+info.shape, info.wcs, dtype)
		print("post_calib %8.5f %8.5f" % (scan_fid.offsets[0,1]/utils.arcmin, scan_fid.offsets[0,2]/utils.arcmin))
		print("info.sys", info.sys)
		print("area", area.shape, area.wcs)
		pmat_map    = pmat.PmatMap(scan_fid, area, sys=info.sys)
		rhs         = area*0
		pmat_map.backward(tod, rhs)
		print("Build div")
		div         = enmap.zeros((ncomp,ncomp)+info.shape, info.wcs, dtype)
		for i in range(3):
			area[:] = 0
			area[i] = 1
			tod[:]  = 0
			pmat_map.forward(tod, area)
			pmat_map.backward(tod, div[i])
		print("Build map")
		idiv = array_ops.eigpow(div, -1, axes=[-4,-3], lim=1e-3, fallback="scalar")
		map  = enmap.samewcs(array_ops.matmul(idiv, rhs, axes=[-4,-3]),area)
		ivar = idiv[0,0]
		# output debug maps
		enmap.write_map(root + "/debug_map_%d.fits" % iround,  map)
		enmap.write_map(root + "/debug_ivar_%d.fits" % iround, ivar)
		print("build table")
		# And extract our thumbnails
		tdata = extract_srcs(map, ivar, spos_flat, tsize=tsize)
		# And the input pointing offsets that were used for building the maps, typically from Matthew
		input_pointoff = scan_fid.d.point_correction
		# We have everything we need. Build our source information table
		table  = np.zeros(len(tdata.inds), [
			("sid", "i"), ("ctime", "d"), ("pos_cel", "2d"), ("center", "2d"), ("pixshape", "2d"),
			("polrot", "d"), ("jacobi", "2,2d"), ("ref_flux", "d")
		]).view(np.recarray)
		table.sid     = [0]                             # index of srcs in input list
		print("tdata.inds", tdata.inds)
		table.ctime   = times[tdata.inds]               # time array center hits each
		table.pos_cel = spos[:,tdata.inds].T # celestial coords {ra,dec} of srcs
		table.center  = tdata.centers                   # tile pixel coords of src {y,x}
		table.pixshape= tdata.pixshapes                 # physical {height,width} of each pixel
		table.polrot  = polrot[tdata.inds]       # pol angle change from cel to osys
		print("jac_pix")
		print(jac_pix*utils.arcmin)
		print(jac_pix.shape)
		table.jacobi  = np.moveaxis(jac_pix,2,0)[tdata.inds] # jacobian {y,x},{focx,focy} (note the ordering!)
		table.ref_flux= fluxes
		baz, waz, bel = [db.data[name]*utils.degree for name in ["baz", "waz", "bel"]]

		# Build an artificial ps2d
		ps2d = build_noise_model_dummy(map, ivar, tsize=tsize)

		# And put it in our result data structure
		data = bunch.Bunch(
			table = table,
			maps  = tdata.maps,
			ps2d  = ps2d,
			ivars = tdata.ivars,
			ftag  = ftag.encode(),
			freq  = freq,
			beam  = scan_fid.beam,
			barea = barea,
			sysinfo = info.sysinfo,
			input_pointoff = input_pointoff,
			ids   = np.char.encode(db.ids),
			baz = baz, waz = waz, bel = bel,
		)
		# Debug write
		write_bunch(root + "/debug_data_%d.hdf" % iround, data)

		# Analyse our simulated data
		print("prepare data")

		# Set the bounding box for our split
		bounds = np.array([[float(w) for w in word.split(":")] for word in args.bounds.split(",")]).T*utils.arcmin
		rmax = np.max(np.abs(np.array(data.maps.shape[-2:])*data.table["pixshape"]/2))
		res  = np.mean(np.product(data.table["pixshape"],1))**0.5 / 4
		oshape, owcs = enmap.thumbnail_geometry(r=rmax, res=res)
		uht  = uharm.UHT(oshape, owcs, mode="flat")

		data = prepare_data(data, oshape, owcs)
		data.fname = "debug"

		print("matched filter")
		data.rho, data.kappa = analysis.matched_filter_constcorr_lowcorr(data.maps, data.lbeam, np.repeat(data.ivars[:,None],3,1), 1/data.ps2d, uht, B2=data.lbeam2)
		data.flux_map, data.dflux_map, data.snr_map = solve(data.rho, data.kappa)
		data.ref_dflux = data.dflux_map[:,0].at([0,0], order=1)
		enmap.write_map(root + "/debug_omaps_%d.fits" % iround, data.maps)
		enmap.write_map(root + "/debug_osnr_%d.fits" % iround, data.snr_map)

		# 4. Fit each individual source
		print("fit")
		fit = fit_group([data], [(0,0)], bounds=bounds, snmin=0, use_brightest=True)
		# Write individual fit to file
		flux, dflux = fit.flux[0], fit.dflux[0]
		msg  = "%.0f %5d %4s" % (fit.ctime, fit.table["sid"][0], fit.ftags[0])
		msg += "  %7.2f  %6.3f %6.3f  %6.3f %6.3f" % (fit.snr, *-fit.pos/utils.arcmin, *fit.dpos/utils.arcmin)
		msg += "  %8.2f  %8.2f %7.2f %8.2f %7.2f %8.2f %7.2f" % (fit.table["model_flux"][0],
				flux[0], dflux[0], flux[1], dflux[1], flux[2], dflux[2])
		msg += "  %7.2f %6.2f %7.2f  %7.2f %7.2f %7.2f" % (fit.baz/utils.degree, fit.waz/utils.degree, fit.bel/utils.degree,
				fit.az/utils.degree, fit.ra/utils.degree, fit.dec/utils.degree)
		msg += "  %5.2f %4d %d %4d" % (fit.ctime/3600%24, 0, i, 0)
		print(msg)
		point_corr += -fit.pos[::-1]
		#point_corr += -np.array([1,0])*utils.arcmin
		print(point_corr/utils.arcmin)
