import numpy as np, time, h5py, copy, argparse, os, sys, pipes, shutil, re
from enlib import enmap, utils, pmat, fft, config, array_ops, mapmaking, nmat, errors, mpi
from enlib import log, bench, dmap, coordinates, scan as enscan, rangelist, scanutils
from enlib import pointsrcs, bunch
from enlib.cg import CG
from enact import actscan, nmat_measure, filedb, todinfo

config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("sel")
parser.add_argument("srcs")
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("--nmax", type=int, default=10)
parser.add_argument("-s", "--src", type=int, default=None, help="Only analyze given source")
parser.add_argument("-c", "--cont", action="store_true")
args = parser.parse_args()

dtype = np.float32 if config.get("map_bits") == 32 else np.float64
comm  = mpi.COMM_WORLD
tcomm = mpi.COMM_SELF
nmax  = args.nmax
ncomp = 3
isys  = "hor"

utils.mkdir(args.odir)
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, rank=comm.rank, shared=False)

# Get the source positions
srcs = np.loadtxt(args.srcs).T

# Read our area. Should be centered on 0,0
area = enmap.read_map(args.area)
area = enmap.zeros((ncomp,)+area.shape[-2:], area.wcs, dtype)

# Build the total set of tasks to do. This is
# the set of tods to examine for every src
tasks = []
filedb.init()
db = filedb.scans.select(filedb.scans[args.sel])
for si, src in enumerate(srcs.T):
	if args.src and si != args.src: continue
	for id in db["hits([%.6f,%6f])" % tuple(src[:2])]:
		if args.cont:
			bid = id.replace(":","_")
			root   = args.odir + "/src%03d_%s_" % (si,bid)
			ofile  = root + "main_map.fits"
			if os.path.isfile(ofile): continue
		tasks.append((si,id))

# Each task processes tasks independently
for ti in range(comm.rank, len(tasks), comm.size):
	si, id = tasks[ti]
	bid = id.replace(":","_")
	L.info("Processing src %3d id %s" % (si, id))
	root   = args.odir + "/src%03d_%s_" % (si,bid)
	entry  = filedb.data[id]
	osys   = "hor:%.6f_%.6f:cel/0_0:hor" % tuple(srcs[:2,si])
	try:
		scans  = [actscan.ACTScan(entry)]
		if scans[0].nsamp == 0 or scans[0].ndet == 0: raise errors.DataMissing("no data in scan")
	except errors.DataMissing as e:
		print "Skipping %s: %s" % (id, e.message)
		continue
	# Signals
	signal_cut = mapmaking.SignalCut(scans, dtype=dtype, comm=tcomm)
	signal_map = mapmaking.SignalMap(scans, area, comm=tcomm, sys=osys)
	# Weights
	weights = [mapmaking.FilterWindow(config.get("tod_window"))]
	# And equation system
	eqsys = mapmaking.Eqsys(scans, [signal_cut, signal_map], weights=weights, dtype=dtype, comm=tcomm)
	eqsys.calc_b()
	# Precons
	signal_cut.precon = mapmaking.PreconCut(signal_cut, scans)
	signal_map.precon = mapmaking.PreconMapBinned(signal_map, signal_cut, scans, weights)

	cg = CG(eqsys.A, eqsys.b, M=eqsys.M, dot=eqsys.dot)
	while cg.i < nmax:
		cg.step()
	eqsys.write(root, "map", cg.x)
	signal_map.precon.write(root)
