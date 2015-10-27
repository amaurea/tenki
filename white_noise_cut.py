# Scans through the indicated tods and computes the ratio
# between the power at mid and high frequency to determine
# how white the white noise floor it. Cuts detectors that
# aren't white enough. Also cuts detectors that have suspiciously
# low white noise floors.

import numpy as np, argparse, h5py, os, sys, shutil
from enlib import fft, utils, enmap, errors, config, mpi
from enact import filedb, data, filters
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("-f", type=str, default="10:1,100:1")
parser.add_argument("-R", type=str, default="0.5:3")
parser.add_argument("--max-sens", type=float, default=20, help="Reject detectors more than this times more sensitive than the median at any of the indicated frequencies. Set to 0 to disable.")
parser.add_argument("--full-stats", action="store_true")
args = parser.parse_args()

comm  = mpi.COMM_WORLD
srate = 400.
fmax  = srate/2
ndet  = 32*33

utils.mkdir(args.odir)

tmp  = [[float(tok) for tok in word.split(":")] for word in args.f.split(",")]
bins = np.array([[t[0]-t[1]/2,t[0]+t[1]/2] for t in tmp])
rate = [float(w) for w in args.R.split(":")]

filedb.init()
ids = filedb.scans[args.sel].ids
ntod= len(ids)

cuts  = np.zeros([ntod,ndet],dtype=np.uint8)
stats = None
if args.full_stats: stats = np.zeros([ntod,ndet,4])
for si in range(comm.rank, ntod, comm.size):
	id    = ids[si]
	entry = filedb.data[id]
	ofile = "%s/%s.txt" % (args.odir, id)
	print "reading %s" % id
	try:
		d     = data.read(entry, fields=["gain","tconst","cut","tod","boresight"])
		d     = data.calibrate(d, nofft=True)
	except (IOError, errors.DataMissing) as e:
		print "skipping (%s)" % (e.message)
		continue
	ft    = fft.rfft(d.tod)
	ps    = np.abs(ft)**2/(d.tod.shape[1]*srate)
	inds  = bins*ps.shape[1]/fmax
	bfreqs= np.mean(bins,1)

	rms_raw = np.array([np.mean(ps[:,b[0]:b[1]],1) for b in inds]).T**0.5
	# Compute amount of deconvolution
	freqs  = np.linspace(0, d.srate/2, ft.shape[-1])
	butter = filters.butterworth_filter(freqs)
	for di, det in enumerate(d.dets):
		tconst = filters.tconst_filter(freqs, d.tau[di])
		ft[di] /= tconst*butter
	ps = np.abs(ft)**2/(d.tod.shape[1]*srate)
	rms_dec = np.array([np.mean(ps[:,b[0]:b[1]],1) for b in inds]).T**0.5

	if args.full_stats:
		stats[si,d.dets][:,:2] = rms_raw
		stats[si,d.dets][:,2:] = rms_dec
	ratio = rms_dec[:,1]/rms_dec[:,0]
	sens  = rms_dec**-2
	med_sens = np.median(sens, 0)
	cuts[si,d.dets] = ((ratio>rate[0])&(ratio<rate[1])&(np.all(sens<med_sens[None,:]*args.max_sens,1)))+1

# Reduce everything
def reduce(a):
	a2 = a.copy()
	comm.Allreduce(a,a2)
	return a2

cuts = reduce(cuts)
if args.full_stats: stats = reduce(stats)

if comm.rank == 0:
	# Output cuts as accept file
	with open(args.odir + "/accept.txt", "w") as ofile:
		for id, icut in zip(ids, cuts):
			ofile.write("%s %3d:" % (id, np.sum(icut==2)))
			for det, dcut in enumerate(icut):
				if dcut == 2:
					ofile.write(" %d" % det)
			ofile.write("\n")
	with open(args.odir + "/cut.txt", "w") as ofile:
		for id, icut in zip(ids, cuts):
			ofile.write("%s %3d:" % (id, np.sum(icut==1)))
			for det, dcut in enumerate(icut):
				if dcut == 1:
					ofile.write(" %d" % det)
			ofile.write("\n")
	if args.full_stats:
		# Output full stats
		with h5py.File(args.odir + "/stats.hdf", "w") as ofile:
			ofile["stats"] = stats
			ofile["ids"]   = ids
