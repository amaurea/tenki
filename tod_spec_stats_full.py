# A more comprehensive version of tod_spec_stats. Reads a
# (potentially large) set of tods, calibrates them and gets
# their fourier transform. Then calculates a set of statistics:
# 1. A low-res spectrum for every detector. This can be used for
#    time constant characterization, but must be low-res to avoid
#    taking too much space. For 10k tods and 1k detectors, 1k freq
#    bins would take 40 GB. That's pretty big, but not unmanagable.
#    But will probably want non-even bins here. For example 2 Hz bins
#    above 10 Hz (95 bins) and 0.1 Hz bins below 10 Hz (100 bins).
#    Or for simplicity let the ranges overlap, so the bins actually
#    are equi-spaced: 100 in [0:10] and 100 in [0:200]. That gives
#    12 GB for this part.
# 2. A high-res average spectrum for each TOD. Can afford ~1k times
#    higher res. How about 0.01 Hz res? That would give 20k bins, taking
#    just 0.8 GB. What sort of average? Plain average is fast and simple,
#    but can be ruined by a single outlier. Could store quantiles, e.g.
#    median, +-1sigma max-min. Would still be within budget.
# 3. May want to encode detector covariance somehow too. Full covmats is
#    out of the picture - would be far too big. But could store something
#    like average correlation per bin. How would I measure that?
#    Calc corr (would be very noisy in such small bins). Mean of
#    off-diagonal elements. This mean would be much less noisy.
#    Would be bad for correlations of mixed sign, though. For these,
#    one could take the square first, and then subtract a noise bias (1).
#    Coudl do both. I'm worried that all those covs would be slow.
#
# To avoid needing to keep many gigs in memory, we do this in chunks
# of e.g. 250 tods, making each file just 324 MB large.

import numpy as np, argparse, h5py, os, sys
from enlib import fft, utils, errors, config, mpi, colors, bench
from enact import filedb, actdata
config.default("cut_mostly_cut",False)
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("prefix", nargs="?", default=None)
parser.add_argument("-b", "--nbin",       type=int,   default=20000)
parser.add_argument("-f", "--fmax",       type=float, default=200)
parser.add_argument("-B", "--nbin-det",   type=int,   default=100)
parser.add_argument("-Z", "--nbin-zoom",  type=int,   default=100)
parser.add_argument("-F", "--fmax-zoom",  type=float, default=10)
parser.add_argument("-C", "--chunk-size", type=int,   default=250)
parser.add_argument("--tconst",     action="store_true")
parser.add_argument("--no-autocut", action="store_true")
args = parser.parse_args()

filedb.init()
ids   = filedb.scans[args.sel]
comm  = mpi.COMM_WORLD
ntod  = len(ids)
csize = args.chunk_size
nchunk= (ntod+csize-1)/csize
dtype = np.float32

utils.mkdir(args.odir)
prefix = args.odir + "/"
if args.prefix: prefix += args.prefix + "_"

# Read a single array_info to determine max det number
array_info = actdata.read_array_info(filedb.data[ids[0]]).array_info
ndet   = array_info.ndet

def bin(a, nbin, zoom=1, return_inds=False):
	ps     = a.reshape(-1,a.shape[-1])
	binds  = np.arange(ps.shape[1])*nbin*zoom/ps.shape[1]
	mask   = binds < nbin
	ps, binds = ps[:,mask], binds[mask]
	res    = np.zeros([ps.shape[0],nbin],dtype=ps.dtype)
	hits   = np.bincount(binds)
	for i in range(ps.shape[0]):
		rhs = np.bincount(binds, ps[i])
		res[i,:len(rhs)] = (rhs/hits)[:nbin]
	res = res.reshape(a.shape[:-1]+res.shape[-1:])
	if return_inds: return res, binds
	else: return res

# Loop over chunks
for chunk in range(nchunk):
	# It's simples to just allocate the full output buffer
	# for every mpi task, even though it wastes a bit of memory
	ind1   = chunk*csize
	ind2   = min(ind1+csize,ntod)
	nctod  = ind2-ind1
	dspecs = np.zeros([nctod,ndet,args.nbin_det], dtype=dtype)
	dzooms = np.zeros([nctod,ndet,args.nbin_zoom],dtype=dtype)
	tspecs = np.zeros([5,nctod,args.nbin],dtype=dtype)
	nhits  = np.zeros([nctod,args.nbin],dtype=int)
	tcorrs = np.zeros([nctod,args.nbin],dtype=dtype)
	srates = np.zeros([nctod],dtype=dtype)
	mce_fsamps = np.zeros([nctod],dtype=dtype)
	mce_params = np.zeros([nctod,4],dtype=dtype)
	for ind in range(ind1+comm.rank, ind2, comm.size):
		i     = ind-ind1
		id    = ids[ind]
		entry = filedb.data[id]
		try:
			# Do not apply time constants. We want raw spectra so that we can
			# use them to estimate time constants ourselves.
			fields= ["array_info", "tags", "site", "mce_filter", "gain","cut","tod","boresight"]
			if args.tconst: fields.append("tconst")
			d     = actdata.read(entry, fields=fields)
			d     = actdata.calibrate(d, exclude=(["autocut"] if not args.no_autocut else []))
			if d.ndet == 0 or d.nsamp == 0: raise errors.DataMissing("empty tod")
		except (IOError, errors.DataMissing) as e:
			print "Skipped (%s)" % (e.message)
			continue
		print "Processing %s" % id
		srates[i] = d.srate
		mce_fsamps[i] = d.mce_fsamp
		mce_params[i] = d.mce_params[:4]
		# Compute the power spectrum
		d.tod = d.tod.astype(dtype)
		nsamp = d.nsamp
		srate = d.srate
		ifmax = d.srate/2
		ft    = fft.rfft(d.tod) / (nsamp*srate)**0.5
		nfreq = ft.shape[-1]
		del d.tod
		ps    = np.abs(ft)**2
		# Det specs
		zoom = int(round(ifmax/args.fmax_zoom))
		dets = actdata.split_detname(d.dets)[1]
		dspecs[i,dets] = bin(ps, args.nbin_det)
		dzooms[i,dets] = bin(ps, args.nbin_zoom, zoom=zoom)
		# Aggregate specs. First bin in small bins
		dhigh, binds = bin(ps, args.nbin, return_inds=True)
		nhits[i] = np.bincount(binds, minlength=args.nbin)
		# Then compute quantiles
		tspecs[0,i] = np.median(dhigh,0)
		tspecs[1,i] = np.percentile(dhigh,15.86553,0)
		tspecs[2,i] = np.percentile(dhigh,84.13447,0)
		tspecs[3,i] = np.min(dhigh,0)
		tspecs[4,i] = np.max(dhigh,0)
		del ps
		# Normalize ft in bins, since we want correlations
		for di in range(d.ndet):
			ft[di] /= (dhigh[di]**0.5)[binds]
		# Average correlation in bin
		sps = np.abs(np.sum(ft,0))**2
		tcorrs[i] = (bin(sps, args.nbin)-d.ndet)/(d.ndet**2-d.ndet)
		del sps, ft, d
	# Ok, we've gone through all the data in our chunk
	with bench.show("Reduce"):
		dspecs = utils.allreduce(dspecs, comm)
		dzooms = utils.allreduce(dzooms, comm)
		tspecs = utils.allreduce(tspecs, comm)
		tcorrs = utils.allreduce(tcorrs, comm)
		srates = utils.allreduce(srates, comm)
		nhits  = utils.allreduce(nhits,  comm)
		mce_fsamps = utils.allreduce(mce_fsamps, comm)
		mce_params = utils.allreduce(mce_params, comm)
	ofile  = prefix + "specs%03d.hdf" % chunk
	if comm.rank == 0:
		# Get rid of empty tods
		good   = np.where(np.any(dspecs>0,(1,2)))[0]
		if len(good) == 0:
			print "No usable tods in chunk!"
			continue
		dspecs = dspecs[good]
		dzooms = dzooms[good]
		tspecs = tspecs[:,good]
		tcorrs = tcorrs[good]
		nhits  = nhits[good]
		chunk_ids = ids[good+ind1]
		print "Writing %s" % ofile
		with h5py.File(ofile, "w") as hfile:
			hfile["dspecs"] = dspecs
			hfile["dzooms"] = dzooms
			hfile["tspecs"] = tspecs
			hfile["tcorrs"] = tcorrs
			hfile["nhits"]  = nhits
			hfile["ids"]    = chunk_ids
			hfile["srates"] = srates
			hfile["mce_fsamps"] = mce_fsamps
			hfile["mce_params"] = mce_params
	del dspecs, dzooms, tspecs, tcorrs
