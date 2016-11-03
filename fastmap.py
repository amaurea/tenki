import numpy as np, sys, os
from enlib import config, mpi, errors, log, utils, coordinates, pmat, wcs as enwcs, enmap
from enact import filedb, actdata
config.default("verbosity", 1, "Verbosity of output")
config.default("work_az_step", 0.1, "Az resolution for workspace tagging in degrees")
config.default("work_el_step", 0.1, "El resolution for workspace tagging in degrees")
config.default("work_ra_step", 10,  "RA resolution for workspace tagging in degrees")
config.default("work_tag_fmt", "%04d_%04d_%03d_%02d", "Format to use for workspace tags")

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
		ira  = int(np.round(ra1/self.ra_step))
		return self.fmt % (iaz1,iaz2,iel,ira)
	def analyze(self, tag):
		iaz1, iaz2, iel, ira = [int(w) for w in tag.split("_")]
		az1, az2, el, ra1 = iaz1*self.az_step, iaz2*self.az_step, iel*self.el_step, ira*self.ra_step
		return az1, az2, el, ra1

def build_fullsky_geometry(res):
	nx = int(np.round(360/res))
	ny = int(np.round(180/res))
	wcs = enwcs.WCS(naxis=2)
	wcs.wcs.cdelt[:] = res
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

def find_t_giving_ra(az, el, ra, site=None, nt=24*60*6, t0=55500):
	"""Find a t (mjd) such that ra(az,el,t) = ra. Uses a simple 10-sec-res
	grid search for now."""
	ts = np.linspace(t0, t0+1, nt, endpoint=False)
	ras, decs = coordinates.transform("hor","cel",[az,el],time=ts,site=site)
	# Handle angle wrapping
	ras = utils.rewind(ras, ra)
	i_ref = np.argmin(np.abs(ras-ra))
	t_ref = ts[i_ref]
	return t_ref

def get_srate(ctime):
	step = ctime.size/10
	ctime = ctime[::step]
	return float(step)/utils.medmean(ctime[1:]-ctime[:-1])

def build_workspace(wid, bore, point_offset, global_wcs, site=None, tagger=None,
		padding=100, max_ra_width=2.5*utils.degree, pre=()):
	if tagger is None: tagger = WorkspaceTagger()
	if isinstance(wid, basestring): wid = tagger.analyze(wid)
	trans = PospixTransform(global_wcs, site=site)
	az1, az2, el, ra1 = wid
	# Extract the workspace definition of the tag name
	ra_ref = ra1 + tagger.ra_step/2
	# We want ra(dec) for up- and down-sweeps for the middle of
	# the workspace. First find a t that will result in a sweep
	# that crosses through the middle of the workspace.
	foc_offset = np.mean(point_offset,0)
	t0   = utils.ctime2mjd(bore[0,0])
	t_ref = find_t_giving_ra(az1+foc_offset[0], el+foc_offset[1], ra_ref, site=site, t0=t0)
	## Get the dec bounds by evaluating all detectors at az1 and az2 at this time
	#decs = [trans([az+point_offset[0],el+point_offset[1]], time=t_ref)[1] for az in [az1,az2]]
	#decs = np.concatenate(decs,1)
	#dec1 = np.min(decs) - padding
	#dec2 = np.max(decs) + padding
	# We also need the corners of the full workspace area.
	t1   = find_t_giving_ra(az1+foc_offset[0], el+foc_offset[1], ra1, site=site, t0=t0)
	t2   = find_t_giving_ra(az1+foc_offset[0], el+foc_offset[1], ra1+tagger.ra_step+max_ra_width, site=site, t0=t0)
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
	nwxs, nwys = [], []
	for si, (afrom,ato) in enumerate([[az1,az2],[az2,az1]]):
		sweep = generate_sweep_by_dec_pix(
				[[ t_ref,     afrom+foc_offset[0],el+foc_offset[1]],
					[t_ref+dmjd,ato  +foc_offset[0],el+foc_offset[1]]
				],iybox,trans)
		# Get the shift in ra pix per dec pix
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
		wycorn = ixcorn - xshift[iycorn-iybox[0]]
		# Modify the shifts so that any scan in this workspace is always transformed
		# to valid positions. wx and wy are transposed relative to x and y
		xshift -= np.min(wycorn)
		nwy = np.max(wycorn)-np.min(wycorn)+1
		nwx = yshift[-1]+1
		# Make yshift a shift like xshift is, so that wy = y + yshift[y-y0].
		yshift -= np.round(sweep[6]).astype(int)
		# And collect so we can pass them to the Workspace construtor later
		xshifts.append(xshift)
		yshifts.append(yshift)
		nwxs.append(nwx)
		nwys.append(nwy)
	# The shifts from each sweep are guaranteed to have the same length,
	# since they are based on the same iybox.
	nwx = np.max(nwxs)
	workspace = Workspace(nwys, nwx, xshifts, yshifts, iybox[0], global_wcs, pre=pre)
	return workspace

class Workspace:
	def __init__(self, nwys, nwx, xshifts, yshifts, y0, wcs, pre=()):
		"""Construct a workspace in shifted coordinates
		wy = x + xshifts[y-y0], wx = yshifts[y-y0], where x and y
		are pixels belonging to the world coordinate system wcs.
		wy and wx are transposed relative to x and y to improve the
		memory access pattern. This means that sweeps are horizontal
		in these coordinates."""
		self.nwx  = nwx
		self.nwys = np.array(nwys)
		self.y0   = y0
		self.wcs  = wcs
		self.xshifts = np.array(xshifts)
		self.yshifts = np.array(yshifts)
		# Build our internal geometry. The wcs part is only
		# used when plotting the workspace.
		shape = tuple(pre) + (np.sum(nwys),nwx)
		local_wcs = build_workspace_wcs(wcs.wcs.cdelt)
		self.map = enmap.zeros(shape, local_wcs)

def unify_sweep_ypix(sweeps):
	y1 = max(*tuple([int(np.round(s[-1,6])) for s in sweeps]))
	y2 = min(*tuple([int(np.round(s[-1,6])) for s in sweeps]))+1
	for i in range(len(sweeps)):
		iy = np.round(sweeps[i][6]).astype(int)
		sweeps[i] = sweeps[i][:,(iy>=y1)&(iy<y2)]
	return np.array(sweeps), [y1,y2]

class PospixTransform:
	def __init__(self, wcs, site=None, isys="hor", osys="cel"):
		self.wcs  = wcs
		self.isys = isys
		self.osys = osys
		self.site = site
	def __call__(self, ipos, time):
		opos = coordinates.transform(self.isys, self.osys, ipos, time=time, site=self.site)
		x, y = self.wcs.wcs_world2pix(opos[0]/utils.degree,opos[1]/utils.degree,0)
		x = utils.unwind(x, 360.0/self.wcs.wcs.cdelt[0])
		opos[0] = utils.unwind(opos[0])
		return np.array([opos[0],opos[1],x,y])

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

elif command == "build":
	# Given a list of id tag, loop over tags, and project tods on
	# a work space per tag.
	parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
	parser.add_argument("command")
	parser.add_argument("todtags")
	args = parser.parse_args()
	filedb.init()

	log_level = log.verbosity2level(config.get("verbosity"))
	L = log.init(level=log_level, rank=comm.rank)
	tagger  = WorkspaceTagger()
	gshape, gwcs = build_fullsky_geometry(0.5/60)

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
		workspace = build_workspace(wid, d.boresight, d.point_offset, global_wcs=gwcs, site=d.site)
		print "%-18s %5d %5d" % ((wid,) + tuple(workspace.map.shape[-2:]))

		#ids = todtags[tag]
		#for ind in range(comm.rank, len(ids), comm.size):
		#	id = ids[ind]

