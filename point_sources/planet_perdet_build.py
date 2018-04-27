# Fit amplitudes and time constants to each detector in a planet tod individually
from __future__ import division, print_function
import numpy as np, sys, os, h5py
from enlib import config, pmat, mpi, errors, gapfill, utils, enmap, bench
from enlib import fft, array_ops, bunch, coordinates
from enact import filedb, actscan, actdata, cuts, filters

parser = config.ArgumentParser(os.environ["HOME"]+"./enkirc")
parser.add_argument("planet")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("-R", "--dist", type=float, default=0.2)
parser.add_argument("-v", "--verbose", action="count", default=1)
parser.add_argument("-q", "--quiet",   action="count", default=0)
parser.add_argument("-c", "--cont",    action="store_true")
args = parser.parse_args()

comm = mpi.COMM_WORLD
filedb.init()
ids  = filedb.scans[args.sel]
R    = args.dist * utils.degree
csize= 100
verbose = args.verbose - args.quiet > 0

dtype= np.float32
model_fknee = 10
model_alpha = 10
sys = "hor:"+args.planet+"/0_0"
tod_sys = config.get("tod_sys")
utils.mkdir(args.odir)
prefix = args.odir + "/"

show = bench.show if verbose else bench.dummy

def estimate_ivar(tod):
	tod -= np.mean(tod,1)[:,None]
	tod  = tod.astype(dtype)
	diff = tod[:,1:]-tod[:,:-1]
	diff = diff[:,:diff.shape[-1]/csize*csize].reshape(tod.shape[0],-1,csize)
	ivar = 1/(np.median(np.mean(diff**2,-1),-1)/2**0.5)
	return ivar

def estimate_atmosphere(tod, region_cut, srate, fknee, alpha):
	model = gapfill.gapfill_joneig(tod, region_cut, inplace=False)
	ft   = fft.rfft(model)
	freq = fft.rfftfreq(model.shape[-1])*srate
	flt  = 1/(1+(freq/fknee)**alpha)
	ft  *= flt
	fft.ifft(ft, model, normalize=True)
	return model

def build_rangedata(tod, rcut, d, ivar):
	nmax   = np.max(rcut.ranges[:,1]-rcut.ranges[:,0])
	nrange = rcut.nrange

	rdata  = bunch.Bunch()
	rdata.detmap = np.zeros(nrange,int)
	rdata.tod = np.zeros([nrange,nmax],dtype)
	rdata.pos = np.zeros([nrange,nmax,2])
	rdata.ivar= np.zeros([nrange,nmax],dtype)
	# Build our detector mapping
	for di in range(rcut.ndet):
		rdata.detmap[rcut.detmap[di]:rcut.detmap[di+1]] = di
	rdata.n = rcut.ranges[:,1]-rcut.ranges[:,0]
	# Extract our tod samples and coordinates
	for i, r in enumerate(rcut.ranges):
		di  = rdata.detmap[i]
		rn  = r[1]-r[0]
		rdata.tod[i,:rn] = tod[di,r[0]:r[1]]
		bore = d.boresight[:,r[0]:r[1]]
		mjd  = utils.ctime2mjd(bore[0])
		pos_hor = bore[1:] + d.point_offset[di,:,None]
		pos_rel = coordinates.transform(tod_sys, sys, pos_hor, time=mjd, site=d.site)
		rdata.pos[i,:rn] = pos_rel.T
		# Expand noise ivar too, including the effect of our normal data cut
		rdata.ivar[i,:rn] = ivar[di] * (1-d.cut[di:di+1,r[0]:r[1]].to_mask()[0])
	
	# Precompute our fourier space units
	rdata.freqs  = fft.rfftfreq(nmax, 1/d.srate)
	# And precompute out butterworth filter
	rdata.butter = filters.mce_filter(rdata.freqs, d.mce_fsamp, d.mce_params)
	# Store the fiducial time constants for reference
	rdata.tau    = d.tau
	# These are also nice to have
	rdata.dsens = ivar**-0.5 / d.srate**0.5
	rdata.asens = np.sum(ivar)**-0.5 / d.srate**0.5
	rdata.srate = d.srate
	rdata.dets  = d.dets
	rdata.beam  = d.beam
	rdata.id    = d.entry.id
	return rdata

def get_rangedata(id):
	entry = filedb.data[id]
	# Read the tod as usual
	with show("read"):
		d = actdata.read(entry)
	with show("calibrate"):
		# Don't apply time constant (and hence butterworth) deconvolution since we
		# will fit these ourselves
		d = actdata.calibrate(d, exclude=["autocut","tod_fourier"])
	if d.ndet == 0 or d.nsamp < 2: raise errors.DataMissing("no data in tod")
	tod = d.tod; del d.tod
	# Very simple white noise model
	with show("noise"):
		ivar  = estimate_ivar(tod)
		asens = np.sum(ivar)**-0.5 / d.srate**0.5
	with show("planet mask"):
		# Generate planet cut
		planet_cut = cuts.avoidance_cut(d.boresight, d.point_offset, d.site, args.planet, R)
	with show("atmosphere"):
		# Subtract atmospheric model
		tod -= estimate_atmosphere(tod, planet_cut, d.srate, model_fknee, model_alpha)
		tod  = tod.astype(dtype, copy=False)
	with show("extract"):
		# Should now be reasonably clean of correlated noise. Extract our range data
		rdata = build_rangedata(tod, planet_cut, d, ivar)
	return rdata

def write_rangedata(fname, rdata):
	with h5py.File(fname, "w") as hfile:
		for key in rdata:
			hfile[key] = rdata[key]

def read_rangedata(fname):
	rdata = bunch.Bunch()
	with h5py.File(fname, "r") as hfile:
		for key in hfile:
			rdata[key] = hfile[key].value
	return rdata

def make_dummy(fname, msg=""):
	with open(fname, "w") as f:
		f.write(msg + "\n")

for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	oid   = id.replace(":","_")
	ofile = prefix + "rdata_%s.hdf" % oid
	odummy= prefix + "rdata_%s.dummy" % oid
	if args.cont and (os.path.isfile(ofile) or os.path.isfile(odummy)):
		print("Skipping %s (already done)" % id)
		continue
	print("Processing %s" % id)
	try: rdata = get_rangedata(id)
	except errors.DataMissing as e:
		print("Skipping %s (%s)" % (id, e))
		make_dummy(odummy, e)
		continue
	write_rangedata(ofile, rdata)
