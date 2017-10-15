# Use joneig filtering to clean the area around each point source position.
# Treat the remainder as white noise. Make thumbnail maps in horizontal coordinates
# centered on the fiducial source position. Output as fully self-contained hdf files
# that can be analyzed by the fitter program.
import numpy as np, os, time, h5py, warnings
from astropy.io import fits
from enlib import utils, config, mpi, errors, sampcut, gapfill, cg
from enlib import pmat, coordinates, enmap, bench, bunch
from enact import filedb, actdata, actscan, nmat_measure
config.default("pmat_accuracy", 10.0, "Factor by which to lower accuracy requirement in pointing interpolation. 1.0 corresponds to 1e-3 pixels and 0.1 arc minute in polangle")
config.default("pmat_interpol_max_size", 1000000, "Maximum mesh size in pointing interpolation. Worst-case time and memory scale at most proportionally with this.")
parser = config.ArgumentParser(os.environ["HOME"] + "./enkirc")
parser.add_argument("srclist")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("-A", "--minamp",    type=float, default=0)
parser.add_argument("-b", "--box",       type=str,   default="-10:10,-10:10")
parser.add_argument("-p", "--pad",       type=float, default=2)
parser.add_argument("-r", "--res",       type=float, default=0.1)
parser.add_argument("-s", "--restrict",  type=str,   default=None)
parser.add_argument("-m", "--minimaps",  action="store_true")
parser.add_argument("-c", "--cont",      action="store_true")
parser.add_argument("-C", "--cols",      type=str, default="0:1:2")
args = parser.parse_args()

# The joneig approach requires the mask to be as small as possible, especially in the scan
# direction, but the source must be entirely contained inside it. Previous tests have shown
# that x position can vary between -2' to 2'. If we want 3 sigma margin, then we need to add
# about 2' to that, for a total of 4' radius and 8' = 0.13 degree diameter. We scan at
# 1 deg/s or so on the sky, so this corresponds to 7.5 Hz. That's a bit lower than I would prefer,
# considering that our fknee is usually 3 Hz. On the other hand, for planet mapping a
# radius of 0.2 deg, corresponding to a diameter of 0.4 deg (24') has worked fine.

# We will make thumbnail maps in source-centered horizontal coordinates, and measure the
# observed source position there. This offset can be related to the pointing offset, though
# the analytical form is ugly on the curved sky.

filedb.init()
ids  = filedb.scans[args.sel]
comm = mpi.COMM_WORLD
dtype= np.float32
nref = 2
min_accuracy = 1.0 # 1 pixel

utils.mkdir(args.odir)
src_cols = [int(w) for w in args.cols.split(":")]

# Set up thumbnail geometry
res  = args.res*utils.arcmin
pad  = args.pad*utils.arcmin
box  = np.array([[float(w) for w in dim.split(":")] for dim in args.box.split(",")]).T*utils.arcmin
box[0] -= pad
box[1] += pad
shape, wcs = enmap.geometry(pos=box, res=res, proj="car")
area = enmap.zeros(shape, wcs, dtype)

def read_srcs(fname, cols=(0,1,2)):
	if fname.endswith(".fits"):
		data = fits.open(fname)[1].data
		return np.array([data.ra*utils.degree,data.dec*utils.degree,data.sn])
	else:
		data = np.loadtxt(fname, usecols=cols).T
		data[:2] *= utils.degree
		return data

def find_ref_pixs(divs, rcost=1.0, dcost=1.0):
	"""rcost is cost per pixel away from center
	dcost is cost per dB change in div value avay from median"""
	# Find median nonzero div per map
	ref_val = np.asarray(np.median(np.ma.array(divs, mask=divs==0),(-2,-1)))
	with utils.nowarn():
		val_off = 10*(np.log10(divs)-np.log10(ref_val[:,None,None]))
	pix_map = divs.pixmap()
	center  = np.array(divs.shape[-2:])/2
	dist_map= np.sum((pix_map-center[:,None,None])**2,-3)**0.5
	cost    = dist_map * rcost + np.abs(val_off)*dcost
	# Sort each by cost. Will be [{y,x},map,order]
	inds    = np.array(np.unravel_index(np.argsort(cost.reshape(cost.shape[0],-1),1), cost.shape[-2:]))
	inds    = inds.T
	# Want to return [order,map,{y,x}]
	return inds

def measure_corr(pmaps, nmat, divs, tod_work, ref_pixs):
	ref_pixs = np.array(ref_pixs)
	tod_work[:] = 0
	map_work    = enmap.zeros((3,)+divs.shape[-2:], divs.wcs, divs.dtype)
	corrs       = enmap.zeros(divs.shape, divs.wcs, divs.dtype)
	for i, pmap in enumerate(pmaps):
		map_work[:] = 0
		map_work[0,ref_pixs[i,0],ref_pixs[i,1]] = 1
		pmap.forward(tod_work, map_work, tmul=1)
	nmat.apply(tod_work)
	for i, pmap in enumerate(pmaps):
		pmap.backward(tod_work, map_work, mmul=0)
		corrs[i] = map_work[0]
		norm     = (divs[i,ref_pixs[i,0],ref_pixs[i,1]] * divs[i])**0.5
		corrs[i,norm>0] /= norm[norm>0]
		corrs[i] = np.roll(np.roll(corrs[i], -ref_pixs[i,0], -2), -ref_pixs[i,1], -1)
	# Shift center to corner, so the result is the correlation relative to
	# corner pixel
	return corrs

class PmatTot:
	def __init__(self, scan, area, sys=None):
		self.pmap  = pmat.PmatMap(scan, area, sys=sys)
		self.pcut  = pmat.PmatCut(scan)
		self.err   = self.pmap.err
	def forward(self, tod, map, tmul=0, mmul=1):
		junk = np.zeros(self.pcut.njunk, tod.dtype)
		self.pmap.forward(tod, map, tmul=tmul, mmul=mmul)
		self.pcut.forward(tod, junk)
	def backward(self, tod, map, tmul=1, mmul=1):
		junk = np.zeros(self.pcut.njunk, tod.dtype)
		self.pcut.backward(tod, junk)
		self.pmap.backward(tod, map, tmul=tmul, mmul=mmul)

# Get source bounds
bounds = filedb.scans.select(ids).data["bounds"]

# Load source database
srcdata  = read_srcs(args.srclist, cols=src_cols)
# srcpos is [{ra,dec},nsrc] in radians, amps is [nsrc]
srcpos, amps = srcdata[:2], srcdata[2]
allowed  = set(range(amps.size))
allowed &= set(np.where(amps > args.minamp)[0])
if args.restrict is not None:
	selected = [int(w) for w in args.restrict.split(",")]
	allowed &= set(selected)

# And loop through our tods
for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	oid   = id.replace(":","_")
	oname = "%s/%s.fits" % (args.odir, oid)
	ename = oname + ".empty"
	if args.cont:
		if os.path.exists(oname):
			print "%s is done: skipping" % id
			continue
		if os.path.exists(ename):
			print "%s already failed: skipping" % id
			continue

	# Check if we hit any of the sources. We first make sure
	# there's no angle wraps in the bounds, and then move the sources
	# to the same side of the sky.
	poly      = bounds[:,:,ind]*utils.degree
	poly[0]   = utils.rewind(poly[0],poly[0,0])
	srcpos    = srcpos.copy()
	srcpos[0] = utils.rewind(srcpos[0], poly[0,0])
	sids      = np.where(utils.point_in_polygon(srcpos.T, poly.T))[0]
	sids      = np.array(sorted(list(set(sids)&allowed)))
	if len(sids) == 0:
		print "%s has 0 srcs: skipping" % id
		continue
	nsrc = len(sids)
	print "%s has %d srcs: %s" % (id,nsrc,", ".join(["%d (%.1f)" % (i,a) for i,a in zip(sids,amps[sids])]))

	def skip(msg):
		print "%s skipped: %s" % (id, msg)
		with open(ename, "w")  as f:
			f.write(msg + "\n")

	entry = filedb.data[id]
	try:
		with bench.mark("read"):
			d = actdata.read(entry)
		with bench.mark("calib"):
			d = actdata.calibrate(d, exclude=["autocut"])
		if d.ndet < 2 or d.nsamp < 1: raise errors.DataMissing("no data in tod")
	except errors.DataMissing as e:
		skip(e.message)
		continue
	tod = d.tod.astype(dtype)
	del d.tod

	# We need a white noise estimate per detector. How should we get this?
	# 1. Use atmosphere-removed samples. But these have the source in them.
	#    But an individual detector is probably pretty noise dominated, so it might work.
	# 2. Use standard noise model to estimate the noise level. A bit slow, but not that bad.
	# 3. Use the running difference to estimate it. This is simple, but may place too
	#    much emphasis on the high-frequency part.
	# I'll go for #2 now, since it's tried and true.
	scan = actscan.ACTScan(entry, d=d)
	nmat = scan.noise.update(tod, d.srate)

	# Set up the local source-centered coordinate system for each source.
	# This coordinate system specification is pretty clunky and string-based, sadly :/
	syss = ["hor:%.6f_%.6f:cel/0_0:hor" % tuple(srcpos[:,sid]/utils.degree) for sid in sids]

	# We now need to subtract the atmosphere behind every source. For this we need to
	# set up cuts corresponding to those samples. The easiest way to do this is via
	# PmatMat.
	pmaps = []
	failed = False
	for sys in syss:
		pmap = PmatTot(scan, area, sys=sys)
		err  = np.max(pmap.err[:2])
		if err > min_accuracy:
			failed = True
			break
		pmaps.append(pmap)
	if failed:
		skip("Pointing model failed: %s" % err)
		continue

	## Use joneig gapfilling to remove the atmosphere. This will zero out samples outside our
	## regions of interest. This does not appear to work as well as I had hoped. The noise is
	## quite strongly non-white even after this step. This probably makes this approach unworkable.
	#tod -= gapfill.gapfill_joneig(tod, tot_cut, inplace=False)

	# Ok, we can now build the thumb maps. Will use a constant correlation noise model:
	# M = hD C hD, where hD = sqrt(div) and C is fourier-diagonal. Compute C via:
	# set center pixel to 1, P'N"P. This gives us a row of M, the inverse pixel-pixel covmat.
	# Divide out the uncorrelated part: M_ij/d_i/d_j, where i is the pixel we set to 1, and
	# j is each pixel in the map. This should make the central pixel approximately 1.
	# Using this approximate model, we can solve for the map as:
	# m = hD" C" hD" rhs
	# and the likelihood is then
	# lik = (m-Pa)hD C hD(m-Pa)
	# hD m = C" hD" rhs
	# lik = (w - hC hD Pa)**2, w = hC hD hD" C" hD" rhs = hC" hD" rhs

	rhss  = enmap.zeros((nsrc,)+shape, wcs, dtype)
	divs  = enmap.zeros((nsrc,)+shape, wcs, dtype)
	work  = enmap.zeros((3,)   +shape, wcs, dtype)
	nmat.apply(tod)
	for i, pmap in enumerate(pmaps):
		pmaps[i].backward(tod, work, mmul=0)
		rhss[i] += work[0]
	tod[:] = 1.0
	nmat.white(tod)
	for i, pmap in enumerate(pmaps):
		pmaps[i].backward(tod, work, mmul=0)
		divs[i] += work[0]
	# Mask unexposed sources
	hit = np.sum(divs>0,(1,2)) > divs.shape[1]*divs.shape[2]*0.25
	if np.sum(hit) == 0:
		skip("No sources actually hit")
		continue
	sids  = sids[hit]
	nsrc  = len(sids)
	rhss  = rhss[hit]
	divs  = divs[hit]
	pmaps = [p for i,p in enumerate(pmaps) if hit[i]]

	corrs  = enmap.zeros((nsrc,)+shape, wcs, dtype)
	chits  = np.zeros(nsrc)
	ref_pixs = find_ref_pixs(divs)
	for i in range(nref):
		corr = measure_corr(pmaps, nmat, divs, tod, ref_pixs[i])
		for i in range(nsrc):
			if corr[i, 0,0] < 0.1: continue
			corrs[i] += corr[i]
			chits[i] += 1
	if np.any(chits==0):
		skip("Failed to measure correlations")
		continue
	corrs /= chits[:,None,None]
	del tod

	# Write as enmap + fits table
	omap = enmap.samewcs([rhss, divs, corrs], rhss)
	header = omap.wcs.to_header(relax=True)
	header['NAXIS'] = omap.ndim
	for i,n in enumerate(omap.shape[::-1]):
		header['NAXIS%d'%(i+1)] = n
	header['id'] = id
	header['off_x'] = d.point_correction[0]/utils.arcmin
	header['off_y'] = d.point_correction[1]/utils.arcmin
	header['bore_el']  = np.mean(d.boresight[2])/utils.degree
	header['bore_az1'] = np.min(d.boresight[1])/utils.degree
	header['bore_az2'] = np.max(d.boresight[1])/utils.degree
	header['ctime']    = np.mean(d.boresight[0])
	map_hdu = fits.PrimaryHDU(omap, header)

	srcinfo = np.zeros(nsrc, [('sid','i'),('ra','f'),('dec','f'),('amp','f')])
	srcinfo["sid"] = sids
	srcinfo["ra"]  = srcpos[0,sids]/utils.degree
	srcinfo["dec"] = srcpos[1,sids]/utils.degree
	srcinfo["amp"] = amps[sids]
	src_hdu = fits.BinTableHDU(srcinfo, fits.Header({"EXTNAME","srcinfo"}))

	rms  = (nmat.ivar*d.srate)**-0.5
	detinfo = np.zeros(d.ndet, [('uid','i'),('rms','f')])
	detinfo["uid"] = d.dets
	detinfo["rms"] = rms
	det_hdu = fits.BinTableHDU(detinfo, fits.Header({"EXTNAME","detinfo"}))

	hdus   = fits.HDUList([map_hdu, src_hdu, det_hdu])
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore')
		hdus.writeto(oname, clobber=True)
