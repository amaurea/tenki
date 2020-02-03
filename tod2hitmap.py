import numpy as np, sys, os
from enlib import enmap, config, log, pmat, mpi, utils, scan as enscan, errors
from enact import actscan, filedb, todinfo

config.default("downsample", 1, "Factor with which to downsample the TOD")
config.default("verbosity",  1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("sel")
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("prefix",nargs="?")
args = parser.parse_args()

filedb.init()
ids = filedb.scans[args.sel]

comm  = mpi.COMM_WORLD
dtype = np.float64
area  = enmap.read_map(args.area).astype(dtype)

utils.mkdir(args.odir)
root = args.odir + "/" + (args.prefix + "_" if args.prefix else "")

# Set up logging
utils.mkdir(root + "log")
logfile   = root + "log/log%03d.txt" % comm.rank
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, file=logfile, rank=comm.rank)

L.info("Initialized")

# Loop through each scan, and compute the hits
hits = enmap.zeros((3,)+area.shape[-2:], area.wcs, dtype=dtype)
myinds = np.arange(comm.rank, len(ids), comm.size)
for ind in myinds:
	id = ids[ind]
	entry = filedb.data[id]
	try:
		scan = actscan.ACTScan(entry)
		if scan.ndet == 0 or scan.nsamp == 0:
			raise errors.DataMissing("Tod contains no valid data")
	except errors.DataMissing as e:
		L.debug("Skipped %s (%s)" % (str(id), e.args[0]))
		continue
	scan = scan[:,::config.get("downsample")]
	L.debug("Processing %s" % str(id))

	pmap = pmat.PmatMap(scan, hits)
	pcut = pmat.PmatCut(scan)
	tod  = np.full([scan.ndet, scan.nsamp], 1, dtype=dtype)
	junk = np.zeros(pcut.njunk, dtype=dtype)
	pcut.backward(tod, junk)
	pmap.backward(tod, hits)
hits = hits[0]

# Collect result
L.info("Reducing")
hits[:] = utils.allreduce(hits, comm)
# Undo effect of downsampling
hits *= config.get("downsample")

# And write it
L.info("Writing")
if comm.rank == 0:
	enmap.write_map(root + "hits.fits", hits)

L.info("Done")
