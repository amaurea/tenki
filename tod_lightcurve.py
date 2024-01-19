# Time-domain forced photometry. Does not fit pointin offses, so should be much
# faster than my other time-domain point source stuff. Measures a per-detector
# amplitude for each point source for each TOD. The point of this is to be able
# to calibrate the point sources against each other even in the presence of sub-array
# gain errors. For example, one could have an interesting transient that might be changing
# on minute-to-minute timescales, which most other sources do not.

# During one TOD different subsets of the array hit each source differently,
# which can make detector gain inconsistenties masquerade as rapid changes in
# flux. We can calibrate this away by using the same subset of detectors for
# the measurement of both the reference sources and the transient.

# The output will be a per-source, per-detector amp, damp, t, where t is the
# timestamp for when the detector hit the source. These will individually be
# very noisy, so they will in practice need to be averaged together, but that
# can be done in postprocessing.
#
# To keep things fast we will use an uncorrelated noise model, which will let us
# solve for each detector separately. This will be slightly suboptimal, but should
# be good enough.

from __future__ import division, print_function
import numpy as np, time, os, sys
from scipy import integrate
from enlib import utils
from enlib import mpi, errors, fft, mapmaking, config, pointsrcs
from enlib import pmat, coordinates, enmap, bench, bunch, nmat, sampcut
from enact import filedb, actdata, actscan

config.set("downsample", 1, "Amount to downsample tod by")
config.set("gapfill", "linear", "Gapfiller to use. Can be 'linear' or 'joneig'")
config.default("pmat_interpol_pad", 10.0, "Number of arcminutes to pad the interpolation coordinate system by")
config.default("pmat_ptsrc_rsigma", 3, "Max number of standard deviations away from a point source to compute the beam profile. Larger values are slower but more accurate, but may lead to misleading effective times in cases where a large region around a source is cut.")

parser = config.ArgumentParser()
parser.add_argument("catalog")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("-s", "--srcs",      type=str,   default=None)
parser.add_argument("-v", "--verbose",   action="count", default=0)
parser.add_argument("-q", "--quiet",     action="count", default=0)
parser.add_argument(      "--minamp",    type=float, default=None)
parser.add_argument(      "--minsn",     type=float, default=1)
parser.add_argument(      "--sys",       type=str,   default="cel")
parser.add_argument(      "--sub",       type=str,   default=None)
args = parser.parse_args()

def read_srcs(fname):
	data = pointsrcs.read(fname)
	return np.array([data.ra, data.dec,data.I])

filedb.init()
db      = filedb.scans.select(args.sel)
ids     = db.ids
sys     = args.sys
comm    = mpi.COMM_WORLD
dtype   = np.float32
verbose = args.verbose - args.quiet
down    = config.get("downsample")
poly_pad= 3*utils.degree
bounds  = db.data["bounds"]
utils.mkdir(args.odir)

srcdata  = read_srcs(args.catalog)

def edges2ranges(edges):
	return np.array([edges[:-1],edges[1:]]).T

def find_ranges(mask):
	"""Given a mask, return [nrange,{from,to}] descripring
	the index ranges over which the mask is true"""
	mask    = np.concatenate([[False],mask,[False]])
	starts  = np.where(~mask[:-1]& mask[1:])[0]
	ends    = np.where( mask[:-1]&~mask[1:])[0]
	return np.array([starts, ends]).T

def find_swipes(az, down=10):
	"""Return the start and end sample for each left/right scan.
	Assumes a regular scanning pattern"""
	amin, amax = utils.minmax(az)
	amid, aamp = 0.5*(amax+amin), 0.5*(amax-amin)
	# Find the max and min of each above-mean and below-mean area.
	# These should be the high and low turnaround points respectively
	rhigh = find_ranges(az>amid+aamp*0.1)
	rlow  = find_ranges(az<amid-aamp*0.1)
	ihigh = np.array([r1+np.argmax(az[r1:r2]) for r1,r2 in rhigh])
	ilow  = np.array([r1+np.argmin(az[r1:r2]) for r1,r2 in rlow ])
	# This only works for a regular scanning pattern
	edges = np.concatenate([[0],np.sort(np.concatenate([ihigh,ilow])),[len(az)]])
	return edges2ranges(edges)

class PmatPtsrcPerpass:
	def __init__(self, scan, srcpos, sys="cel"):
		# First find the sample ranges for each swipe
		self.swipes = find_swipes(scan.boresight[:,1]) # [nswipe,{from,to}]
		nswipe = len(self.swipes)
		# Build source parameter struct for PmatPtsrc
		self.params = np.zeros([nswipe,srcpos.shape[-1],8],np.float64)
		self.params[:,:,:2] = srcpos[::-1,:].T
		self.params[:,:,5:7] = 1
		self.psrc = pmat.PmatPtsrc(scan, self.params[0], sys=sys)
		self.pcut = pmat.PmatCut(scan)
		self.cut = scan.cut
	def forward(self, tod, amps, pmul=1):
		# Amps should be [nswipe,nsrc,nstokes]
		params = self.params.copy()
		params[:,:,2:2+amps.shape[-1]] = amps
		bore = self.psrc.scan.boresight
		for si, (r1,r2) in enumerate(self.swipes):
			# This is not efficient, but lets us avoid modifying the fortran core.
			# It's also super-hacky
			rbore= np.ascontiguousarray(bore[r1:r2])
			rtod = np.ascontiguousarray(tod [:,r1:r2])
			self.psrc.scan.boresight = rbore
			self.psrc.forward(rtod, params[si], pmul=pmul)
			tod[:,r1:r2] = rtod
		self.psrc.scan.boresight = bore
		sampcut.gapfill_linear(self.cut, tod, inplace=True)
	def backward(self, tod, amps=None, pmul=1, ncomp=3):
		params = self.params.copy()
		tod  = sampcut.gapfill_linear(self.cut, tod, inplace=False, transpose=True)
		bore = self.psrc.scan.boresight
		for si, (r1,r2) in enumerate(self.swipes):
			# This is not efficient, but lets us avoid modifying the fortran core
			rbore= np.ascontiguousarray(bore[r1:r2])
			rtod = np.ascontiguousarray(tod [:,r1:r2])
			self.psrc.scan.boresight = rbore
			self.psrc.backward(rtod, params[si], pmul=pmul)
		self.psrc.scan.boresight = bore
		if amps is None: amps = params[...,2:2+ncomp]
		else: amps[:] = params[...,2:2+amps.shape[-1]]
		return amps

def rhand_polygon(poly):
	"""Returns True if the polygon is ordered in the right-handed convention,
	where the sum of the turn angles is positive"""
	poly = np.concatenate([poly,poly[:1]],0)
	vecs = poly[1:]-poly[:-1]
	vecs /= np.sum(vecs**2,1)[:,None]**0.5
	vecs = np.concatenate([vecs,vecs[:1]],0)
	cosa, sina = vecs[:-1].T
	cosb, sinb = vecs[1:].T
	sins = sinb*cosa - cosb*sina
	coss = sinb*sina + cosb*cosa
	angs = np.arctan2(sins,coss)
	tot_ang = np.sum(angs)
	return tot_ang > 0

def pad_polygon(poly, pad):
	"""Given poly[nvertex,2], return a new polygon where each vertex has been moved
	pad outwards."""
	sign  = -1 if rhand_polygon(poly) else 1
	pwrap = np.concatenate([poly[-1:],poly,poly[:1]],0)
	vecs  = pwrap[2:]-pwrap[:-2]
	vecs /= np.sum(vecs**2,1)[:,None]**0.5
	vort  = np.array([-vecs[:,1],vecs[:,0]]).T
	return poly + vort * sign * pad

def get_sids_in_tod(id, src_pos, bounds, ind, isids=None, src_sys="cel"):
	if isids is None: isids = list(range(src_pos.shape[-1]))
	if bounds is not None:
		poly      = bounds[:,:,ind]*utils.degree
		poly[0]   = utils.rewind(poly[0],poly[0,0])
		# bounds are defined in celestial coordinates. Must convert srcpos for comparison
		mjd       = utils.ctime2mjd(float(id.split(".")[0]))
		srccel    = coordinates.transform(src_sys, "cel", src_pos, time=mjd)
		srccel[0] = utils.rewind(srccel[0], poly[0,0])
		poly      = pad_polygon(poly.T, poly_pad).T
		accepted  = np.where(utils.point_in_polygon(srccel.T, poly.T))[0]
		sids      = [isids[i] for i in accepted]
	else:
		sids = isids
	return sids

def get_beam_area(beam):
	r, b = beam
	return integrate.simps(2*np.pi*r*b,r)

# If the point sources are far enough away from each other, then they will
# be indepdendent from each other, and all their amplitudes can be fit in
# a single evaluation. Normally you would do:
#  amps = (P'N"P)" P'N"d
# where you need to evaluate P'N"P via unit vector bashing. But if you know
# that it's diagonal, then you can use a single non-unit vector instead:
# diag(P'N"P ones(nsrc)). But should check how good an approximation this is.
# Intuitively, it's a good approximation if the shadow from one souce doesn't
# touch that from another.
#
# P(pos) = int_amp P(pos,amp|d) damp
#        = K int_amp exp(-0.5*(d-Pa)'N"(d-Pa))
#        = K int_amp exp(-0.5*[
#             a'P'N"P(a-(P'N"P)"P'N"d

# (d-Pa)'N"(d-Pa) = d'N"d - 2d'N"Pa + (Pa)'N"Pa

# Load source database
srcpos, amps = srcdata[:2], srcdata[2]
# Which sources pass our requirements?
base_sids  = set(range(amps.size))
if args.minamp is not None:
	base_sids &= set(np.where(amps > args.minamp)[0])
if args.srcs is not None:
	selected = [int(w) for w in args.srcs.split(",")]
	base_sids &= set(selected)
base_sids = list(base_sids)

if args.sub:
	background = enmap.read_map(args.sub).astype(dtype)

# We will collect the mpi output into these, which we will reduce in the end
olines = []
otimes = []

# Iterate over tods in parallel
for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	ofile = args.odir + "/flux_%s.hdf" % id.replace(":","_")
	# Figure out which point sources are relevant for this tod
	sids = get_sids_in_tod(id, srcpos[:,base_sids], bounds, ind, base_sids, src_sys=sys)
	if len(sids) == 0:
		print("%s has 0 srcs: skipping" % id)
		continue
	try:
		nsrc = len(sids)
		print("%s has %d srcs: %s" % (id,nsrc,", ".join(["%d (%.1f)" % (i,a) for i,a in zip(sids,amps[sids])])))
	except TypeError as e:
		print("Weird: %s" % e)
		print(sids)
		print(amps)
		continue

	# Read the data
	entry = filedb.data[id]
	try:
		scan = actscan.ACTScan(entry, verbose=verbose>=2)
		if scan.ndet < 2 or scan.nsamp < 1: raise errors.DataMissing("no data in tod")
	except errors.DataMissing as e:
		print("%s skipped: %s" % (id, e))
		continue
	# Apply downsampling
	scan = scan[:,::down]
	# Prepeare our samples
	scan.tod = scan.get_samples()
	utils.deslope(scan.tod, w=5, inplace=True)
	scan.tod = scan.tod.astype(dtype)

	# Background subtraction
	if args.sub:
		Pmap = pmat.PmatMap(scan, background, sys=sys)
		Pmap.forward(scan.tod, background, tmul=-1)

	# Build the noise model
	N = scan.noise.update(scan.tod, scan.srate)
	# Build a pointsource pointing matrix that measures each swipe of
	# the telescope over the sky separately
	P = PmatPtsrcPerpass(scan, srcpos[:,sids], sys=sys)

	# rhs
	N.apply(scan.tod)
	rhs = P.backward(scan.tod, ncomp=3)
	# div
	scan.tod[:] = 0
	P.forward(scan.tod, rhs*0+1)
	N.apply(scan.tod)
	div = P.backward(scan.tod, ncomp=3)

	# Use beam to turn amp into flux. We want the flux in mJy, so divide by 1e3
	beam_area = get_beam_area(scan.beam)
	_, uids   = actdata.split_detname(scan.dets) # Argh, stupid detnames
	freq      = scan.array_info.info.nom_freq[uids[0]]
	fluxconv  = utils.flux_factor(beam_area, freq*1e9)/1e3

	div      /= fluxconv**2
	rhs      /= fluxconv

	# Solve. Unhit sources will be nan with errors inf
	with utils.nowarn():
		flux  = rhs/div
		dflux = div**-0.5
	del rhs, div

	# Get the mean time for swipe. This will be nan for unhit sources
	scan.tod[:] = scan.boresight[None,:,0]
	N.white(scan.tod)
	trhs = P.backward(scan.tod, ncomp=1)
	# We want the standard deviation too
	scan.tod[:] = scan.boresight[None,:,0]**2
	N.white(scan.tod)
	t2rhs = P.backward(scan.tod, ncomp=1)
	# Get the div and hits
	scan.tod[:] = 1
	N.white(scan.tod)
	tdiv = P.backward(scan.tod, ncomp=1)
	scan.tod[:] = 1
	hits = P.backward(scan.tod, ncomp=1)
	with utils.nowarn():
		t  = trhs/tdiv
		t2 = t2rhs/tdiv
		trms = (t2-t**2)**0.5
		t0 = utils.mjd2ctime(scan.mjd0)
	t += t0

	# At this point we have flux, dflux [nswipe,nsrc,nstokes] and t[nswipe,nsrc]
	# Output to terminal and work file, skipping unhit sources
	nswipe = len(flux)
	for swi in range(nswipe):
		for si in range(nsrc):
			# Skip entries where we don't have a well-defined time or flux
			good = True
			for a in [t, trms, flux, dflux]:
				good = good and np.isfinite(a[swi,si,0])
			if not good: continue
			msg = "%13.2f %4.2f %4d %8.2f %7.2f %8.2f %7.2f %8.2f %7.2f %s" % (
					t[swi,si,0], trms[swi,si,0],
					sids[si],
					flux[swi,si,0], dflux[swi,si,0],
					flux[swi,si,1], dflux[swi,si,1],
					flux[swi,si,2], dflux[swi,si,2],
					id,
				)
			print(msg)
			olines.append(msg)
			otimes.append(t[swi,si,0])

# Reduce
otimes = utils.allgatherv(otimes, comm)
olines = utils.allgatherv(olines, comm)
order  = np.argsort(otimes)
olines = olines[order]
if comm.rank == 0:
	with open(args.odir + "/lightcurves.txt", "w") as ofile:
		for oline in olines:
			ofile.write(oline + "\n")
