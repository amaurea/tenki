import numpy as np, argparse, h5py, os, sys
from enlib import fft, utils, enmap, errors, config, mpi
from enact import filedb, data
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("sel")
parser.add_argument("ofile")
parser.add_argument("-N", type=int, default=10000)
parser.add_argument("-w", "--weight", action="store_true")
args = parser.parse_args()

comm = mpi.COMM_WORLD
srate = 400.
fmax  = srate/2
ndet = 33*32

# Use power at 10 Hz to compute weights
fref = [9.5,10.5]

# Create a global (all tods + all dets) 1-D noise spectrum. Result
# will probably have to be highpass filtered, but that can be done elsewhere
myps = np.zeros(args.N)
mynspec = 0

filedb.init()
ids = filedb.scans[args.sel]
for si in range(comm.rank, len(ids), comm.size):
	id    = ids[si]
	entry = filedb.data[id]
	print "reading %s" % id
	try:
		d     = data.read(entry)
		d     = data.calibrate(d, nofft=True)
	except (IOError, errors.DataMissing) as e:
		print "skipping (%s)" % e.message
		continue
	n     = d.tod.shape[1]
	ft    = fft.rfft(d.tod)
	del d.tod
	ps    = np.abs(ft)**2/(n*srate)
	inds  = np.linspace(0,args.N,ps.shape[1],endpoint=False).astype(int)

	for det, ps_det in zip(d.dets,ps):
		ps_bin = np.bincount(inds, ps_det, minlength=args.N)/np.bincount(inds, minlength=args.N)
		weight = 1.0
		if args.weight:
			iref = [f*n/srate for f in fref]
			weight = 1.0/np.mean(ps_det[iref[0]:iref[1]])
		myps += ps_bin*weight
		mynspec += weight

totps = myps.copy()
comm.Allreduce(myps, totps)
totnspec = comm.allreduce(mynspec)
totps /= totnspec
totfreq = np.linspace(0, fmax, args.N)

if comm.rank == 0:
	np.savetxt(args.ofile, np.array([totfreq, totps]).T, fmt="%8.3f %17.9e")
