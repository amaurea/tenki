# -*- coding: utf-8 -*-
import numpy as np, sys, os, h5py, copy, time
from scipy import optimize
from enlib import fft
from enlib import config, mpi, errors, log, utils, coordinates, pmat, zipper
from enlib import wcs as enwcs, enmap, array_ops
from enlib.cg import CG
from enact import filedb, actdata, actscan
import astropy.io.fits
config.default("verbosity", 1, "Verbosity of output")
config.default("work_az_step", 0.1, "Az resolution for workspace tagging in degrees")
config.default("work_el_step", 0.1, "El resolution for workspace tagging in degrees")
config.default("work_ra_step", 10,  "RA resolution for workspace tagging in degrees")
config.default("work_tag_fmt", "%04d_%04d_%03d_%02d", "Format to use for workspace tags")
config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("downsample", 1, "Factor with which to downsample the TOD")
fft.engine = "fftw"

# Fast and incremental mapping program.
# Idea:
#  1. Map in 2 steps: tod -> work and work -> tod
#  2. Work coordinates are pixel-shifted versions of sky coordinates,
#     with a different shift for each scanning pattern. Defined in
#     3 steps:
#      1. Shift in RA based on example sweep (fwd and back separately)
#      2. Subdivide pixels in dec to make them approximately equi-spaced
#         in azimuth.
#      3. Transpose for memory access efficiency.
#  3. Ignore detector correlations. That way the work-space noise matrix
#     is purely horizontal.
#  4. Each tod-detector makes an almost horizontal line in horizontal
#     coordinates. If it were perfectly horizontal, then we could apply
#     individual noise spectra to each detector, and just build an average
#     inv spec per line. Due to deviations from being horizontal, single
#     detector scans will gradually move from one line to another. We can
#     approximately model this by making a nit-weighted average per line.
#  5. For each new tod, determine which work-space it belongs to, and
#     apply bw += Pt'Wt Ft Wt d and and Ww**2 += Pt'Wt**2 1, where Wt is the sqrt of the
#     sample weighting and Ft is the frequency filter, which is constant per
#     line of this workspace, and Pt is the tod-to-work pointing matrix.
#     (Wt is just a number per detector, so it commutes with Ft)
#  6. We model the work maps as mw = Pw m + n, cov(n)" = Ww Fw Ww
#     (Pw' Ww Fw Ww Pw)m = Pw' Ww Fw Ww mw approx Pw' bw
#     So we can solve the full map via CG on the work maps, all of which
#     are pixel-space operations, with no interpolation.
#  7. This map will be approximately unbiased if Pw' Ww Fw Ww mw is approximately Pw' bw.
#     Deviations come from:
#      1. Scans not mapping completely horizontally to work spaces
#      2. Detector are offset in az
#      3. Inexact mapping from t to x in work coordinates
#
# In theory we could have one work-space per scanning pattern, but
# that would make them very big. Instead, it makes sense to divide
# them into blocks by starting RA (RA of bottom-left corner of scan).

# How would this work in practice? Split operation into 3 steps:
# STEP 1: Classify
#  1. For each new tod-file, read in its pointing and determine its scanning
#     pattern.
#  2. Determine a workspace-id, which is given by el-az1-az2-rablock-array-noise.
#     Rablock is an integer like int(sRA/15). Noise is an integer like int(log(spec(0.1Hz)/spec(10Hz)))
#  3. Output a file with lines of [tod] [workspace-id]
# STEP 2: Build
#  1. Read in file from previous step, and group tods by wid.
#  2. For each wid, create its metadata. This should be fully specified by the
#     workspace-id + settings, so multiple runs create compatible workspaces. So
#     reading in a TOD should not be necessary for this.
#     Metadata is: workspace area, pixel shifts, frequency filter and t-x scaling.
#  3. For each tod in group, read and calibrate it. Then measure the noise spec
#     per det. Get white noise level for det-weighting, and check how well noise
#     profile matches freq filter. Cut outliers (tradeoff).
#  4. Project into our workspace rhs and div, adding it to the existing ones.
# STEP 3: Solve
#  1. Loop through workspaces and build up bm = Pw' bw and diag(Wm) = Pw' diag(Ww)
#  2. Solve system through CG-iteration.
#
# Should we maintain a single set of workspaces that we keep updating with more data?
# Or should we create a new set of workspaces per week, say? The latter will take more
# disk space, but makes it possible to produce week-wise maps and remove any glitch
# weeks. As long as the workspace-ids are fully deterministic, weeks can still be easily
# coadded later.
#
# I prefer the latter. It makes each run independent, and you don't risk losing data
# by making an error while updating the existing files.

def read_todtags(fname):
	todtags = {}
	with open(fname, "r") as f:
		for line in f:
			if len(line) == 0 or line[0] == "#":
				continue
			id, tag = line.split()
			if tag not in todtags:
				todtags[tag] = []
			todtags[tag].append(id)
	return todtags

class WorkspaceTagger:
	def __init__(self, az_step=None, el_step=None, ra_step=None, fmt=None):
		self.az_step = config.get("work_az_step", az_step)*utils.degree
		self.el_step = config.get("work_el_step", el_step)*utils.degree
		self.ra_step = config.get("work_ra_step", ra_step)*utils.degree
		self.fmt     = config.get("work_tag_fmt", fmt)
	def build(self, az1, az2, el, ra1):
		iaz1 = int(np.round(az1/self.az_step))
		iaz2 = int(np.round(az2/self.az_step))
		iel  = int(np.round(el/self.el_step))
		ira  = int(np.floor(ra1/self.ra_step))
		return self.fmt % (iaz1,iaz2,iel,ira)
	def analyze(self, tag):
		iaz1, iaz2, iel, ira = [int(w) for w in tag.split("_")]
		az1, az2, el, ra1 = iaz1*self.az_step, iaz2*self.az_step, iel*self.el_step, ira*self.ra_step
		return az1, az2, el, ra1

def build_fullsky_geometry(res):
	nx = int(np.round(360/res))
	ny = int(np.round(180/res))
	wcs = enwcs.WCS(naxis=2)
	wcs.wcs.cdelt[:] = [-res,res]
	wcs.wcs.crpix = [1+nx/2,1+ny/2]
	wcs.wcs.crval = [0,0]
	wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
	return (ny+1,nx), wcs

def build_workspace_wcs(res):
	wcs = enwcs.WCS(naxis=2)
	wcs.wcs.cdelt[:] = res
	wcs.wcs.crpix[:] = 1
	wcs.wcs.crval[:] = 0
	# Should really be plain, but let's pretend it's
	# CAR for ease of plotting.
	wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
	return wcs

def find_t_giving_ra(az, el, ra, site=None, nt=50, t0=55500):
	"""Find a t (mjd) such that ra(az,el,t) = ra. Uses a simple 10-sec-res
	grid search for now."""
	def err(t):
		ora, odec = coordinates.transform("hor","cel",[az,el],time=t,site=site)
		return utils.rewind(ra-ora, 0)
	ts = np.linspace(t0-0.5,t0+0.5,nt)
	errs = err(ts)
	i = np.where((errs[2:]*errs[:-2] < 0)&(abs(errs[2:]-errs[:-2])<np.pi))[0][0]
	return optimize.brentq(err, ts[i], ts[i+2])

def get_srate(ctime):
	step = ctime.size/10
	ctime = ctime[::step]
	return float(step)/utils.medmean(ctime[1:]-ctime[:-1])

def valid_az_range(az1, az2):
	az1 %= 2*np.pi
	az2 %= 2*np.pi
	return (az1-np.pi)*(az2-np.pi) > 0

class WorkspaceError(Exception): pass

def build_workspace_geometry(wid, bore, point_offset, global_wcs, site=None, tagger=None,
		padding=100, max_ra_width=2.5*utils.degree, ncomp=3, dtype=np.float64):
	if tagger is None: tagger = WorkspaceTagger()
	if isinstance(wid, basestring): wid = tagger.analyze(wid)
	if not valid_az_range(wid[0], wid[1]): raise WorkspaceError("Azimuth crosses north/south")

	trans = TransformPos2Pospix(global_wcs, site=site)
	az1, az2, el, ra1 = wid
	# Extract the workspace definition of the tag name
	ra_ref = ra1 + tagger.ra_step/2
	# We want ra(dec) for up- and down-sweeps for the middle of
	# the workspace. First find a t that will result in a sweep
	# that crosses through the middle of the workspace.
	foc_offset = np.mean(point_offset,0)
	t0   = utils.ctime2mjd(bore[0,0])
	t_ref = find_t_giving_ra(az1+foc_offset[0], el+foc_offset[1], ra_ref, site=site, t0=t0)
	# We also need the corners of the full workspace area.
	t1   = find_t_giving_ra(az1+foc_offset[0], el+foc_offset[1], ra1, site=site, t0=t0)
	t2   = find_t_giving_ra(az1+foc_offset[0], el+foc_offset[1], ra1+tagger.ra_step+max_ra_width, site=site, t0=t0)
	#print "t1", t1, "t2", t2
	#print "az1", az1/utils.degree, "az2", az2/utils.degree
	#print "ra", ra1/utils.degree, (ra1+tagger.ra_step+max_ra_width)/utils.degree
	bore_box_hor = np.array([[t1,az1,el],[t2,az2,el]])
	bore_corners_hor = utils.box2corners(bore_box_hor)
	work_corners_hor = bore_corners_hor[None,:,:] + (point_offset[:,[0,0,1]] * [0,1,1])[:,None,:]
	work_corners_hor = work_corners_hor.T.reshape(3,-1)
	work_corners     = trans(work_corners_hor[1:], time=work_corners_hor[0])
	ixcorn, iycorn   = np.round(work_corners[2:]).astype(int)
	iybox = np.array([np.min(iycorn)-padding,np.max(iycorn)+1+padding])
	# Generate an up and down sweep
	srate  = get_srate(bore[0])
	period = pmat.get_scan_period(bore[1], srate)
	dmjd   = period/2./24/3600
	xshifts = []
	yshifts = []
	work_dazs = []
	nwxs, nwys = [], []
	for si, (afrom,ato) in enumerate([[az1,az2],[az2,az1]]):
		sweep = generate_sweep_by_dec_pix(
				[[ t_ref,     afrom+foc_offset[0],el+foc_offset[1]],
					[t_ref+dmjd,ato  +foc_offset[0],el+foc_offset[1]]
				],iybox,trans)
		# Get the shift in ra pix per dec pix. At this point,
		# the shifts are just relative to the lowest-dec pixel
		xshift = np.round(sweep[5]-sweep[5,0,None]).astype(int)
		# Get the shift in dec pix per dec pix. These tell us where
		# each working pixel starts as a function of normal dec pixel.
		# For example [0,1,3,6] would mean that the work to normal pixel
		# mapping is [0,1,1,2,2,2]. This is done to make dwdec/daz approximately
		# constant
		daz = np.abs(sweep[1,1:]-sweep[1,:-1])
		daz_ratio = np.maximum(1,daz/np.min(daz[1:-1]))
		yshift  = np.round(utils.cumsum(daz_ratio, endpoint=True)).astype(int)
		yshift -= yshift[0]
		# Now that we have the x and y mapping, we can determine the
		# bounds of our workspace by transforming the corners of our
		# input coordinates.
		#print "iyc", iycorn-iybox[0]
		#print "ixc", ixcorn
		#for i in np.argsort(iycorn):
		#	print "A %6d %6d" % (iycorn[i], ixcorn[i])
		#print "min(ixc)", np.min(ixcorn)
		#print "max(ixc)", np.max(ixcorn)
		#print "xshift", xshift[iycorn-iybox[0]]
		wycorn = ixcorn - xshift[iycorn-iybox[0]]
		#print "wycorn", wycorn
		#print "min(wyc)", np.min(wycorn)
		#print "max(wyc)", np.max(wycorn)
		# Modify the shifts so that any scan in this workspace is always transformed
		# to valid positions. wx and wy are transposed relative to x and y.
		# Padding is needed because of the rounding involved in recovering the
		# az and el from the wid.
		xshift += np.min(wycorn)
		xshift -= padding
		wycorn2= ixcorn - xshift[iycorn-iybox[0]]
		#print "wycorn2", wycorn2
		#print "min(wyc2)", np.min(wycorn2)
		#print "max(wyc2)", np.max(wycorn2)
		#sys.stdout.flush()
		nwy = np.max(wycorn)-np.min(wycorn)+1 + 2*padding
		nwx = yshift[-1]+1
		# Get the average azimuth spacing in wx
		work_daz = (sweep[1,-1]-sweep[1,0])/(yshift[-1]-yshift[0])
		print work_daz/utils.degree
		# And collect so we can pass them to the Workspace construtor later
		xshifts.append(xshift)
		yshifts.append(yshift)
		nwxs.append(nwx)
		nwys.append(nwy)
		work_dazs.append(work_daz)
	# The shifts from each sweep are guaranteed to have the same length,
	# since they are based on the same iybox.
	nwx = np.max(nwxs)
	# To translate the noise properties, we need a mapping from the x and t
	# fourier spaces. For this we need the azimuth scanning speed.
	scan_speed = 2*(az2-az1)/period
	work_daz  = np.mean(work_dazs)
	wgeo = WorkspaceGeometry(nwys, nwx, xshifts, yshifts, iybox[0], scan_speed, work_daz, global_wcs, ncomp=ncomp, dtype=dtype)
	return wgeo

class WorkspaceGeometry:
	#:, pre=(), dtype=np.float64):
	def __init__(self, nwys, nwx, xshifts, yshifts, y0, scan_speed, daz, wcs, ncomp=3, dtype=np.float64):
		"""Construct a workspace geometry in shifted coordinates
		wy = x - xshifts[y-y0], wx = yshifts[y-y0], where x and y
		are pixels belonging to the world coordinate system wcs.
		wy and wx are transposed relative to x and y to improve the
		memory access pattern. This means that sweeps are horizontal
		in these coordinates."""
		# Define our coordinate transformation
		self.nwx  = nwx
		self.nwys = np.array(nwys)
		self.y0   = y0
		self.gwcs = wcs
		self.daz  = daz
		self.scan_speed = scan_speed
		self.xshifts = np.array(xshifts)
		self.yshifts = np.array(yshifts)
		# Define our internal geometry. The
		# lwcs part is only used for plotting the workspace.
		self.shape = (ncomp,np.sum(nwys),nwx)
		self.lwcs  = build_workspace_wcs(wcs.wcs.cdelt)
		self.num_az_freq = nwx/2+1
		self.ncomp = ncomp
		self.dtype = dtype
	def copy(self): return copy.deepcopy(self)
	def to_hfile(self, hfile):
		hfile["nwys"]    = self.nwys
		hfile["nwx"]     = self.nwx
		hfile["xshifts"] = self.xshifts
		hfile["yshifts"] = self.yshifts
		hfile["y0"]      = self.y0
		hfile["daz"]     = self.daz
		hfile["scan_speed"] = self.scan_speed
		hfile["ncomp"]   = ncomp
		hfile["dtype"]   = np.dtype(self.dtype).char
		header = self.gwcs.to_header()
		for key in header:
			hfile["gwcs/"+key] = header[key]
	@classmethod
	def from_hfile(cls, hfile):
		nwys    = hfile["nwys"].value
		nwx     = hfile["nwx"].value
		xshifts = hfile["xshifts"].value
		yshifts = hfile["yshifts"].value
		y0      = hfile["y0"].value
		daz     = hfile["daz"].value
		scan_speed = hfile["scan_speed"].value
		ncomp   = hfile["ncomp"].value
		dtype   = np.dtype(hfile["dtype"].value)
		header = astropy.io.fits.Header()
		hwcs   = hfile["gwcs"]
		for key in hwcs:
			header[key] = hwcs[key].value
		gwcs = enwcs.WCS(header).sub(2)
		return cls(nwys, nwx, xshifts, yshifts, y0, daz, scan_speed, gwcs, ncomp=ncomp, dtype=dtype)

class Workspace:
	"""A Workspace consists of:
	1. Variables specifying a shifted, per-scan-pattern coordinate system and how to
	   transform between it a pixelization of celestial coordinates.
	2. rhs: Filtered data that has been projected onto the workspace coordinates
	3. hdiv: [TQU,TQU] hitcounts for the projection
	4. wfilter: A workspace-representation of the filter that has been applied to the data."""
	def __init__(self, geometry, rhs=None, hdiv=None, wfilter=None, ids=[]):
		if rhs  is None: dtype = geometry.dtype
		if rhs  is None:
			rhs  = enmap.zeros(geometry.shape, geometry.lwcs, dtype)
		if hdiv is None:
			hdiv = enmap.zeros((geometry.ncomp,) + geometry.shape, geometry.lwcs, dtype)
		if wfilter is None:
			wfilter = np.zeros([geometry.shape[-2],geometry.num_az_freq],dtype=dtype)
		self.geometry = geometry
		self.rhs     = rhs
		self.hdiv    = hdiv
		self.wfilter = wfilter
		self.ids     = list(ids)
	def copy(self): return copy.deepcopy(self)
	def reduce(self, comm):
		res = self.copy()
		res.rhs[:]     = utils.allreduce(self.rhs, comm)
		res.hdiv[:]    = utils.allreduce(self.hdiv, comm)
		res.wfilter[:] = utils.allreduce(self.wfilter, comm)
		res.ids        = comm.allreduce(list(res.ids))
		return res
	def __add__(self, other):
		res = self.copy()
		res.rhs     += other.rhs
		res.hdiv    += other.hdiv
		res.wfilter += other.wfilter
		res.ids      = list(res.ids) + list(other.ids)
		return res
	def to_hfile(self, hfile):
		hfile["rhs"]  = self.rhs
		hfile["hdiv"] = self.hdiv
		hfile["wfilter"] = self.wfilter
		hfile["ids"] = np.array(ids)
		self.geometry.to_hfile(hfile.create_group("geometry"))
	@classmethod
	def from_hfile(cls, hfile):
		rhs  = hfile["rhs"].value
		hdiv = hfile["hdiv"].value
		wfilter = hfile["wfilter"].value
		ids = list(hfile["ids"].value)
		geometry = WorkspaceGeometry.from_hfile(hfile["geometry"])
		return cls(geometry, rhs, hdiv, wfilter, ids)

def write_workspace(fname, workspace):
	with h5py.File(fname, "w") as hfile:
		workspace.to_hfile(hfile)

def read_workspace(fname):
	with h5py.File(fname, "r") as hfile:
		return Workspace.from_hfile(hfile)

def unify_sweep_ypix(sweeps):
	y1 = max(*tuple([int(np.round(s[-1,6])) for s in sweeps]))
	y2 = min(*tuple([int(np.round(s[-1,6])) for s in sweeps]))+1
	for i in range(len(sweeps)):
		iy = np.round(sweeps[i][6]).astype(int)
		sweeps[i] = sweeps[i][:,(iy>=y1)&(iy<y2)]
	return np.array(sweeps), [y1,y2]

class TransformPos2Pospix:
	def __init__(self, wcs, site=None, isys="hor", osys="cel"):
		self.wcs  = wcs
		self.isys = isys
		self.osys = osys
		self.site = site
	def __call__(self, ipos, time):
		opos = coordinates.transform(self.isys, self.osys, ipos, time=time, site=self.site)
		x, y = self.wcs.wcs_world2pix(opos[0]/utils.degree,opos[1]/utils.degree,0)
		nx = int(np.abs(360.0/self.wcs.wcs.cdelt[0]))
		x = utils.unwind(x, nx, ref=nx/2)
		opos[0] = utils.unwind(opos[0])
		return np.array([opos[0],opos[1],x,y])

class TransformPos2Pix:
	"""Transforms from scan coordinates to pixel-center coordinates.
	This becomes discontinuous for scans that wrap from one side of the
	sky to another for full-sky pixelizations."""
	def __init__(self, scan, wcs):
		self.scan = scan
		self.wcs  = wcs
	def __call__(self, ipos):
		"""Transform ipos[{t,az,el},nsamp] into opix[{y,x,c,s},nsamp]."""
		shape = ipos.shape[1:]
		ipos  = ipos.reshape(ipos.shape[0],-1)
		time  = self.scan.mjd0 + ipos[0]/utils.day2sec
		opos  = coordinates.transform("hor", "cel", ipos[1:], time=time, site=self.scan.site, pol=True)
		opix  = np.zeros((4,)+ipos.shape[1:])
		opix[:2] = self.wcs.wcs_world2pix(*tuple(opos[:2]/utils.degree)+(0,))[::-1]
		nx    = int(np.abs(360/self.wcs.wcs.cdelt[0]))
		opix[1]  = utils.unwind(opix[1], period=nx, ref=nx/2)
		opix[2]  = np.cos(2*opos[2])
		opix[3]  = np.sin(2*opos[2])
		return opix.reshape((opix.shape[0],)+shape)

def generate_sweep_by_dec_pix(hor_box, iy_box, trans, padstep=None,nsamp=None,ntry=None):
	"""Given hor_box[{from,to},{t,az,el}] and a hor2{ra,dec,y,x} transformer trans,
	and a integer y-pixel range iy_box[{from,to}]. Compute an azimuth sweep that samples every y pixel once and
	covers the whole dec_range."""
	if nsamp   is None: nsamp   = 100000
	if padstep is None: padstep = 4*utils.degree
	if ntry    is None: ntry    = 10
	pad = padstep
	for i in range(ntry):
		t_range, az_range, el_range = np.array(hor_box).T
		az_range = utils.widen_box(az_range, pad, relative=False)
		# Generate a test sweep, which hopefully is wide enough and dense enough
		time = np.linspace(t_range[0],t_range[1], nsamp)
		ipos = [
				np.linspace(az_range[0],az_range[1], nsamp),
				np.linspace(el_range[0],el_range[1], nsamp)]
		opos = trans(ipos, time)
		opos = np.concatenate([[time],ipos,opos],0)
		# Make sure we cover the whole dec range we should.
		# We all our samples are in the range we want to use,
		# then we probably didn't cover the whole range.
		# ....|..++++....|... vs. ...--|+++++++|--...
		iy = np.round(opos[6]).astype(int)
		if not (np.any(iy < iy_box[0]) and np.any(iy >= iy_box[1])):
			pad += padstep
			continue
		good = (iy >= iy_box[0]) & (iy < iy_box[1])
		opos = opos[:,good]
		# Sort by output y pixel (not rounded)
		order = np.argsort(opos[6])
		opos = opos[:,order]
		# See if we hit every y pixel
		iy = np.round(opos[6]).astype(int)
		uy = np.arange(iy_box[0],iy_box[1])
		ui = np.searchsorted(iy, uy)
		if len(np.unique(ui)) < len(uy):
			nsamp *= 2
			continue
		opos = opos[:,ui]
		return opos

class PmatWorkspaceTOD(pmat.PointingMatrix):
	"""Fortran-accelerated scan <-> enmap pointing matrix implementation
	for workspaces."""
	def __init__(self, scan, wgeo):
		# Build the pointing interpolator
		self.trans = TransformPos2Pix(scan, wgeo.gwcs)
		self.poly  = pmat.PolyInterpol(self.trans, scan.boresight, scan.offsets)
		# Build the pixel shift information. This assumes ces-like scans in equ-like systems
		self.sdir    = pmat.get_scan_dir(scan.boresight[:,1])
		self.period  = pmat.get_scan_period(scan.boresight[:,1], scan.srate)
		self.wgeo    = wgeo
		self.nphi    = int(np.abs(360./wgeo.gwcs.wcs.cdelt[0]))
		self.core    = pmat.get_core(wgeo.dtype)
		self.scan    = scan
	def get_pix_phase(self):
		ndet, nsamp = self.scan.ndet, self.scan.nsamp
		wgeo   = self.wgeo
		pix    = np.zeros([ndet,nsamp],np.int32)
		phase  = np.zeros([ndet,nsamp,2],self.wgeo.dtype)
		self.core.pmat_map_get_pix_poly_shift_xy(pix.T, phase.T, self.scan.boresight.T,
				self.scan.hwp_phase.T, self.scan.comps.T, self.poly.coeffs.T, self.sdir,
				wgeo.y0, wgeo.nwx, wgeo.nwys, wgeo.xshifts.T, wgeo.yshifts.T, self.nphi)
		return pix, phase
	def forward(self, tod, map, pix, phase, tmul=1, mmul=1, times=None):
		"""m -> tod"""
		if times is None: times = np.zeros(5)
		self.core.pmat_map_use_pix_direct(1, tod.T, tmul, map.T, mmul, pix.T, phase.T, times)
	def backward(self, tod, map, pix, phase, tmul=1, mmul=1, times=None):
		"""tod -> m"""
		if times is None: times = np.zeros(5)
		self.core.pmat_map_use_pix_direct(-1, tod.T, tmul, map.T, mmul, pix.T, phase.T, times)

class PmatWorkspaceMap(pmat.PointingMatrix):
	def __init__(self, wgeo):
		self.wgeo = wgeo
		self.nphi = int(np.abs(360./wgeo.gwcs.wcs.cdelt[0]))
		self.core = pmat.get_core(wgeo.dtype)
	def forward(self, work, map):
		wgeo = self.wgeo
		self.core.pmat_workspace(1,  work.T, map.T, wgeo.y0, wgeo.nwx, wgeo.nwys, wgeo.xshifts.T, wgeo.yshifts.T, self.nphi)
	def backward(self, work, map):
		wgeo = self.wgeo
		self.core.pmat_workspace(-1, work.T, map.T, wgeo.y0, wgeo.nwx, wgeo.nwys, wgeo.xshifts.T, wgeo.yshifts.T, self.nphi)

def measure_inv_noise_spectrum(ft, nbin):
	ndet, nfreq = ft.shape
	ps    = np.abs(ft)**2
	binds = np.arange(nfreq)*nbin/nfreq
	Nmat  = np.zeros([ndet,nbin])
	hits  = np.bincount(binds)
	for di in range(ndet):
		Nmat[di] = np.bincount(binds, ps[di])
	Nmat /= hits
	iNmat = 1/Nmat
	return iNmat, binds

def project_tod_on_workspace(scan, tod, wgeo):
	"""Compute the tod onto a map using the pixelization defined
	in the workspace, and return it along with a [TQU,TQU] hitmap
	and a hits-by-detector-by-y array."""
	rhs  = enmap.zeros(wgeo.shape, wgeo.lwcs, wgeo.dtype)
	hdiv = enmap.zeros((rhs.shape[:1]+rhs.shape),rhs.wcs, rhs.dtype)
	# Project it onto the workspace
	pcut = pmat.PmatCut(scan)
	pmap = PmatWorkspaceTOD(scan, wgeo)
	pix, phase = pmap.get_pix_phase()
	# Build rhs
	junk = np.zeros(pcut.njunk,dtype=rhs.dtype)
	pcut.backward(tod, junk)
	pmap.backward(tod, rhs, pix, phase)
	# Build div
	tmp = hdiv[0].copy()
	for i in range(ncomp):
		tmp[:] = np.eye(ncomp)[i,:,None,None]
		pmap.forward(tod, tmp, pix, phase)
		pcut.backward(tod, junk)
		pmap.backward(tod, hdiv[i], pix, phase)
	# Find each detector's hits by wy. Some detectors have
	# sufficient residual curvature that they hit every wy.
	yhits = np.zeros([scan.ndet, rhs.shape[-2]],dtype=np.int32)
	core  = pmat.get_core(dtype)
	core.bincount_flat(yhits.T, pix.T, rhs.shape[-2:], 0)
	return rhs, hdiv, yhits

def project_binned_spec_on_workspace(ispec, srate, yhits, wgeo):
	#  wrhs[c,y,x] = hdiv[c,b,y,x] (F" Fw F Psm sky)[b,y,x]
	#  Fw[y,k] = (tdsum yhits[y])" (tdsum yhits[y]*Ft[y,k*dfaz*vaz/dft])
	#  hdiv = tdsum diag(Pwt Pwt')
	ndet, nbin = ispec.shape
	nafreq = wgeo.nwx/2+1
	afreq  = np.arange(nafreq)/(wgeo.daz*wgeo.nwx)
	tfreq  = afreq * wgeo.scan_speed
	bind   = np.minimum((2*tfreq/srate*nbin).astype(int),nbin)
	ospec  = np.zeros([wgeo.shape[-2],nafreq])
	# Build the weighted average
	for di in range(d.ndet):
		ospec += yhits[di,:,None] * ispec[di,bind][None,:]
	return ospec

def offset_wcs(wcs, pos):
	"""Find the integer pixel offset required to give the pos=[dec,ra]
	(radians), the pixel position [0,0]. Return owcs, offset[{y,x}], where owcs
	is a copy) of wcs where this offset has been applied."""
	pos    = np.array(pos)/utils.degree
	offset = wcs.wcs_world2pix(pos[1],pos[0],0)
	owcs   = wcs.deepcopy()
	owcs.wcs.crpix -= offset
	return owcs, offset[::-1]

class FastmapSolver:
	def __init__(self, workspaces, template, comm=None):
		"""Initialize a FastmapSolver for the equation system given by the workspace list
		workspaces. The template argument specifies the output coordinate system. This
		enmap have a wcs which is pixel-compatible with that used to build the workspaces."""
		if comm is None: comm = mpi.COMM_WORLD
		# Find the global coordinate offset needed to match our
		# global wcs with the template wcs
		corner = template.pix2sky([0,0])
		gwcs, offset = offset_wcs(workspaces[0].geometry.gwcs, corner)
		# Prepare workspaces for solving in these coordinates
		self.workspaces = []
		for work in workspaces:
			work = work.copy()
			# Normalize hdiv
			with utils.nowarn():
				work.hdiv_norm = work.hdiv / np.sum(work.hdiv[0,0],-1)[None,None,:,None]
			work.hdiv_norm[~np.isfinite(work.hdiv_norm)] = 0
			# Update the global wcs and pixel coordinates
			work.geometry.gwcs = gwcs
			work.geometry.y0  -= offset[0]
			work.geometry.xshifts -= offset[1]
			# Set up our ponting matrix
			work.pmat = PmatWorkspaceMap(work.geometry)
			self.workspaces.append(work)
		# Update our template to match the geometry we're actually using.
		# If the original template was compatible, this will be a NOP geometry-wise
		template = enmap.zeros((work.geometry.ncomp,)+template.shape[-2:], work.geometry.gwcs, work.geometry.dtype)
		# Build a simple binned preconditioner
		# FIXME: This just makes things worse
		#idiv = enmap.zeros((work.geometry.ncomp,work.geometry.ncomp)+template.shape[-2:], work.geometry.gwcs, work.geometry.dtype)
		#for work in self.workspaces:
		#	wmap = enmap.zeros(work.geometry.shape, work.geometry.lwcs, work.geometry.dtype)
		#	for i in range(work.geometry.ncomp):
		#		tmp = idiv[0]*0
		#		tmp[i] = 1
		#		work.pmat.forward(wmap, tmp)
		#		wmap[:] = array_ops.matmul(work.hdiv, wmap, [0,1])
		#		wmap[:] = array_ops.matmul(np.rollaxis(work.hdiv,1), wmap, [0,1])
		#		work.pmat.backward(wmap, idiv[i])
		#self.prec = array_ops.eigpow(idiv, -1, axes=[0,1])
		self.dof  = zipper.ArrayZipper(template)
		self.comm = comm
	def A(self, x):
		map = self.dof.unzip(x)
		res = map*0
		for work in self.workspaces:
			# This is normall P'N"P. In our case 
			wmap = enmap.zeros(work.geometry.shape, work.geometry.lwcs, work.geometry.dtype)
			work.pmat.forward(wmap, map)
			ft  = fft.rfft(wmap)
			ft *= work.wfilter
			fft.ifft(ft, wmap, normalize=True)
			wmap[:] = array_ops.matmul(work.hdiv_norm, wmap, [0,1])
			# Noise weighting would go here. No weighting for now
			wmap[:] = array_ops.matmul(np.rollaxis(work.hdiv_norm,1), wmap, [0,1])
			fft.rfft(wmap, ft)
			ft *= work.wfilter
			fft.ifft(ft, wmap, normalize=True)
			work.pmat.backward(wmap, res)
		res = utils.allreduce(res, self.comm)
		return self.dof.zip(res)
	def M(self, x):
		map = self.dof.unzip(x)
		map[:] = array_ops.matmul(self.prec, map, [0,1])
		return self.dof.zip(map)
	def calc_b(self):
		res = self.dof.unzip(np.zeros(self.dof.n))
		for work in self.workspaces:
			# Noise weighitng would go here. No weighting for now
			wmap = np.ascontiguousarray(work.rhs.copy())
			# This matmul is inefficient due to the memory layout.
			# Several transposing copies are involved.
			wmap[:] = array_ops.matmul(np.rollaxis(work.hdiv_norm,1), work.rhs, [0,1])
			ft  = fft.rfft(wmap)
			ft *= work.wfilter
			fft.ifft(ft, wmap, normalize=True)
			work.pmat.backward(wmap, res)
		res = utils.allreduce(res, self.comm)
		return res



if len(sys.argv) < 2:
	sys.stderr.write("Usage python fastmap.py [command], where command is classify, build or solve\n")
	sys.exit(1)

command = sys.argv[1]
comm    = mpi.COMM_WORLD
dec_pad   = 0.5*utils.degree
ra_pad    = 0.5*utils.degree
ra_max_width = 2*utils.degree

if command == "classify":
	# For each selected tod, output its id and a wokspace id (wid) defining the workspace
	# it belongs in.
	parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
	parser.add_argument("command")
	parser.add_argument("sel")
	args = parser.parse_args()
	filedb.init()

	min_samps = 20e3
	log_level = log.verbosity2level(config.get("verbosity"))
	L = log.init(level=log_level, rank=comm.rank)
	tagger = WorkspaceTagger()

	ids = filedb.scans[args.sel]
	for ind in range(comm.rank, len(ids), comm.size):
		id    = ids[ind]
		entry = filedb.data[id]
		try:
			# We need the tod and all its dependences to estimate which noise
			# category the tod falls into. But we don't need all the dets.
			# Speed things up by only reading 25% of them.
			d = actdata.read(entry, ["boresight","point_offsets","site"])
			d = actdata.calibrate(d, exclude=["autocut"])
			if d.ndet == 0 or d.nsamp == 0:
				raise errors.DataMissing("Tod contains no valid data")
			if d.nsamp < min_samps:
				raise errors.DataMissing("Tod is too short")
		except errors.DataMissing as e:
			L.debug("Skipped %s (%s)" % (id, e.message))
			continue
		L.debug(id)

		# Get the scan el and az bounds
		az1 = np.min(d.boresight[1])
		az2 = np.max(d.boresight[1])
		el  = np.mean(d.boresight[2])

		if not valid_az_range(az1, az2):
			L.debug("Skipped %s (%s)" % (id, "Azimuth crosses poles"))
			continue

		# Then get the ra block we live in. This is set by the lowest RA-
		# detector at the lowest az of the scan at the earliest time in
		# the scan. So transform all the detectors.
		ipoint = np.zeros([2, d.ndet])
		ipoint[0] = az1 + d.point_offset[:,0]
		ipoint[1] = el  + d.point_offset[:,1]
		mjd    = utils.ctime2mjd(d.boresight[0,0])
		opoint = coordinates.transform("hor","cel",ipoint,time=mjd,site=d.site)
		ra1    = np.min(opoint[0])
		wid    = tagger.build(az1,az2,el,ra1)
		print "%s %s" % (id, wid)
		sys.stdout.flush()

elif command == "build":
	# Given a list of id tag, loop over tags, and project tods on
	# a work space per tag.
	parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
	parser.add_argument("command")
	parser.add_argument("todtags")
	parser.add_argument("odir")
	args = parser.parse_args()
	filedb.init()

	log_level = log.verbosity2level(config.get("verbosity"))
	L = log.init(level=log_level, rank=comm.rank)
	dtype   = np.float32 if config.get("map_bits") == 32 else np.float64
	nbin    = 10000
	ncomp   = 3
	tagger  = WorkspaceTagger()
	downsample = config.get("downsample")
	gshape, gwcs = build_fullsky_geometry(0.5/60)
	utils.mkdir(args.odir)

	todtags = read_todtags(args.todtags)
	print "Found %d tags" % len(todtags)
	wids = sorted(todtags.keys())
	for wid in wids:
		ids = todtags[wid]
		# We need the focalplane, which will be contant for all
		# tods in a wid, to get accurate bounds of the workspace
		# we will create.
		d = actdata.read(filedb.data[ids[0]], ["boresight","point_offsets","site"])
		d = actdata.calibrate(d, exclude=["autocut"])
		# Prepare the workspace for this wid
		try:
			wgeo = build_workspace_geometry(wid, d.boresight, d.point_offset, global_wcs=gwcs, site=d.site, ncomp=ncomp, dtype=dtype)
		except WorkspaceError as e:
			L.debug("Skipped pattern %s (%s)" % (wid, e.message))
			continue
		print "%-18s %5d %5d" % ((wid,) + tuple(wgeo.shape[-2:]))
		tot_work = Workspace(wgeo)
		oname = "%s/%s.hdf" % (args.odir, wid)

		# And process the tods that fall within this workspace
		for ind in range(comm.rank, len(ids), comm.size):
			id    = ids[ind]
			entry = filedb.data[id]
			try:
				d = actscan.ACTScan(entry)
				if d.ndet == 0 or d.nsamp == 0:
					raise errors.DataMissing("Tod contains no valid data")
				d = d[:,::downsample]
				d = d[:,:]
			except errors.DataMissing as e:
				L.debug("Skipped %s (%s)" % (id, e.message))
				continue
			L.debug("Processing %s" % id)
			# Get the actual tod
			tod = d.get_samples()
			tod -= np.mean(tod,1)[:,None]
			tod = tod.astype(dtype)
			# Compute the per-detector spectrum
			ft  = fft.rfft(tod) * d.nsamp ** -0.5
			tfilter, binds = measure_inv_noise_spectrum(ft, nbin)
			# Apply inverse noise weighting to the tod
			ft *= tfilter[:,binds]
			ft *= d.nsamp ** -0.5
			fft.ifft(ft, tod)
			del ft
			my_rhs, my_hdiv, my_yhits = project_tod_on_workspace(d, tod, wgeo)
			my_wfilter = project_binned_spec_on_workspace(tfilter, d.srate, my_yhits, wgeo)
			# Add to the totals
			tot_work.rhs  += my_rhs
			tot_work.hdiv += my_hdiv
			tot_work.wfilter += my_wfilter
			tot_work.ids.append(id)
			del my_rhs, my_hdiv, my_yhits, my_wfilter
		# Reduce
		tot_work = tot_work.reduce(comm)
		if comm.rank == 0:
			write_workspace(oname, tot_work)

elif command == "solve":
	parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
	parser.add_argument("command")
	parser.add_argument("template")
	parser.add_argument("ifiles", nargs="+")
	parser.add_argument("odir")
	args = parser.parse_args()

	log_level = log.verbosity2level(config.get("verbosity"))
	L = log.init(level=log_level, rank=comm.rank)
	utils.mkdir(args.odir)

	L.info("Reading template")
	template = enmap.read_map(args.template)
	
	L.info("Reading workspaces")
	ifiles = args.ifiles
	nwork  = len(ifiles)
	inds   = range(comm.rank, nwork, comm.size)
	mywork = []
	for ind in inds:
		L.debug("Read %s" % ifiles[ind])
		mywork.append(read_workspace(ifiles[ind]))

	# We will solve this system by conjugate gradients:
	#  Pw' Fw' hdiv_norm' N" hdiv_norm Fw Pw map = Pw' Fw' hdiv_norm' N" rhs
	# What should N" be? rhs = Pt' Nt" (Pt map + n)
	# var(rhs) = var(Pt' N" Pt) = ?

	# By the time we have made workspaces, the pixelization is fixed.
	# Our only freedom is to select which subregion of that pixel space
	# to actually include in our maps. We will take in a template which
	# must be in compatible pixelization to indicate this region.
	L.info("Initializing solver")
	solver = FastmapSolver(mywork, template, comm)
	L.info("Computing right-hand side")
	b = solver.calc_b()
	L.info("Solving")
	cg = CG(solver.A, solver.dof.zip(b))
	for i in range(1000):
		t1 = time.time()
		cg.step()
		t2 = time.time()
		if cg.i % 10 == 0 and comm.rank == 0:
			m = solver.dof.unzip(cg.x)
			enmap.write_map(args.odir + "/step%04d.fits" % cg.i, m)
		if comm.rank == 0:
			print "%5d %15.7e %7.2f" % (cg.i, cg.err, t2-t1)
