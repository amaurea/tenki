"""Create maps in azimuth phase, in order to investigate magnetic pickup. We assume
that relative azimuth is all that matters, so time, elevation and azimuth center
can be ignored."""
import numpy as np, sys, os, time, h5py
from enact import filedb, data
from enlib import enmap, config, utils, array_ops, fft, errors, gapfill, rangelist, mpi
from scipy.interpolate import UnivariateSpline

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("selector")
parser.add_argument("odir")
parser.add_argument("-f", "--filter", type=float, default=0)
parser.add_argument("-c", "--cols", default="[0:32]")
args = parser.parse_args()

utils.mkdir(args.odir)

comm = mpi.COMM_WORLD
ids  = filedb.scans[args.selector]
nbin = 1000
ncol, nrow = 32,33
ndet = ncol*nrow

cols = []
for centry in args.cols.split(","):
	toks = centry.split(":")
	if len(toks) == 1:
		cols.append(int(centry))
	else:
		cols += range(int(toks[0]),int(toks[1]))
colmask = np.full(ncol, False, dtype=np.bool)
colmask[cols] = True
absdets = np.where(colmask[np.arange(ndet)%ncol])[0]

def white_est(tod): return np.std(tod[:,1:]-tod[:,:-1],1)
def highpass(tod, f, srate=400):
	tod = tod.copy()
	ft = fft.rfft(tod)
	ft[:,:int(f/float(srate)*tod.shape[1])] = 0
	fft.ifft(ft, tod, normalize=True)
	return tod

class Eq:
	def __init__(self, nvar, ncomp, nbin, name="", diag=False):
		self.rhs = np.zeros([nvar, ncomp,nbin])
		if diag:
			self.div = np.zeros([nvar,ncomp,nbin])
		else:
			self.div = np.zeros([nvar, ncomp,ncomp,nbin])
		self.name = name
	@property
	def nvar(self): return self.rhs.shape[0]
	@property
	def ncomp(self): return self.rhs.shape[1]
	@property
	def nbin(self): return self.rhs.shape[2]
	@property
	def diag(self): return self.div.ndim == 3
	def reduce(self):
		res = Eq(self.nvar, self.ncomp, self.nbin, name=self.name, diag=self.diag)
		comm.Allreduce(self.rhs, res.rhs)
		comm.Allreduce(self.div, res.div)
		return res
	def solve(self):
		if self.diag: return self.rhs/self.div
		else: return array_ops.solve_multi(self.div, self.rhs, axes=[1,2])

tod_eq = Eq(2,3,nbin,"tod")
acc_eq = Eq(2,1,nbin,"acc")
det_eq = Eq(2,ndet,nbin,"det",diag=True)

def output_cum(si):
	if comm.rank == 0:
		hfile = h5py.File(args.odir + "/cum%04d.hdf" % si, "w")
	for eq in [tod_eq,acc_eq,det_eq]:
		tot = eq.reduce()
		if comm.rank == 0:
			hfile[eq.name + "/rhs"] = tot.rhs
			hfile[eq.name + "/div"] = tot.div
			hfile[eq.name + "/sig"] = tot.solve()
	if comm.rank == 0:
		hfile.close()

for si in range(comm.rank, len(ids)/comm.size*comm.size, comm.size):
	entry = filedb.data[ids[si]]
	print "Reading %s" % entry.id
	try:
		d = data.read(entry, fields=["gain","polangle","tconst","boresight","cut","tod"], absdets=absdets)
		d = data.calibrate(d)
	except (IOError, OSError,errors.DataMissing) as e:
		print "Skipping [%s]" % str(e)
		output_cum(si)
		continue
	print "Computing pol tod"
	ndet, nsamp = d.tod.shape
	print ndet, nsamp

	# Estimate white noise level (rough)
	rms = white_est(d.tod[:,:50000])
	weight = (1-d.cut.to_mask())/rms[:,None]**2

	if args.filter: d.tod = highpass(d.tod, args.filter)

	# Project whole time-stream to TQU
	comps = np.zeros([3,ndet])
	comps[0] = 1
	comps[1] = np.cos(+2*d.polangle)
	comps[2] = np.sin(-2*d.polangle)
	polrhs = comps.dot(d.tod*weight)
	poldiv = np.einsum("ad,bd,di->abi",comps,comps,weight)
	poltod = array_ops.solve_multi(poldiv,polrhs,axes=[0,1])
	print "Computing az bin"
	az    = d.boresight[1]
	def build_bins(a, nbin):
		box = np.array([np.min(a),np.max(a)])
		return np.minimum(np.floor((a-box[0])/(box[1]-box[0])*nbin).astype(int),nbin-1)
	def bin_by_pix(a, pix, nbin):
		a = np.asarray(a)
		if a.ndim == 0: a = np.full(len(pix), a)
		fa = a.reshape(-1,a.shape[-1])
		fo = np.zeros([fa.shape[0],nbin])
		for i in range(len(fa)):
			fo[i] = np.bincount(pix, fa[i], minlength=nbin)
		return fo.reshape(a.shape[:-1]+(nbin,))
	# Gapfill poltod in regions with far too few hits, to
	# avoid messing up the poltod power spectrum
	mask = poldiv[0,0] < np.mean(poldiv[0,0])*0.1
	for i in range(poltod.shape[0]):
		poltod[i] = gapfill.gapfill_copy(poltod[i], rangelist.Rangelist(mask))

	# Calc phase which is equal to az while az velocity is positive
	# and 2*max(az) - az while az velocity is negative
	x = np.arange(len(az))
	az_spline = UnivariateSpline(x, az, s=1e-4)
	daz  = az_spline.derivative(1)(x)
	ddaz = az_spline.derivative(2)(x)
	phase = az.copy()
	phase[daz<0] = 2*np.max(az)-az[daz<0]
	# Bin by az and phase
	apix = build_bins(az, nbin)
	ppix = build_bins(phase, nbin)
	for i, pix in enumerate([apix, ppix]):
		tod_eq.rhs[i] += bin_by_pix(polrhs, pix, nbin)
		tod_eq.div[i] += bin_by_pix(poldiv, pix, nbin)
		acc_eq.rhs[i] += bin_by_pix(ddaz,   pix, nbin)
		acc_eq.div[i] += bin_by_pix(1,      pix, nbin)
		for j in range(ndet):
			di = d.dets[j]
			det_eq.rhs[i,di] += bin_by_pix(d.tod[j]*weight[j], pix, nbin)
			det_eq.div[i,di] += bin_by_pix(weight[j], pix, nbin)

	with h5py.File(args.odir + "/"+ entry.id, "w") as hfile:
		hfile["poltod"] = poltod
		hfile["polrhs"] = polrhs
		hfile["poldiv"] = poldiv
		hfile["az"]     = az
		hfile["daz"]    = daz
		hfile["ddaz"]   = ddaz
		hfile["phase"]  = phase

	output_cum(si)
