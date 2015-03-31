import numpy as np, argparse, h5py, os, sys
from enlib import fft, utils, enmap, errors, config
from enact import filedb, data
from mpi4py import MPI
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("sel")
parser.add_argument("ofile")
parser.add_argument("-N", type=int, default=10000)
args = parser.parse_args()

comm = MPI.COMM_WORLD
srate = 400.
fmax  = srate/2
ndet = 33*32

# Create a global (all tods + all dets) 1-D noise spectrum. Result
# will probably have to be highpass filtered, but that can be done elsewhere
myps = np.zeros(args.N)

filedb.init()
ids = filedb.scans[args.sel].ids
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
		myps += ps_bin

totps = myps.copy()
comm.Allreduce(myps, totps)
totfreq = np.linspace(0, fmax, args.N)

if comm.rank == 0:
	np.savetxt(args.ofile, np.array([totfreq, totps]).T, fmt="%8.3f %17.9e")
