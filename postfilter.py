import numpy as np, os
from enlib import config, utils, mpi, enmap, dmap, mapmaking, todfilter, log, scanutils
from enact import filedb, actdata, actscan
config.default("verbosity", 2, "Verbosity")
parser = config.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("idlist")
parser.add_argument("omap")
parser.add_argument("-s", "--sys", type=str,   default="cel")
parser.add_argument(      "--daz", type=float, default=3.0)
parser.add_argument(      "--nt",  type=int,   default=10)
parser.add_argument(      "--dets",type=str,   default=0)
parser.add_argument(      "--ntod",type=int,   default=0)
parser.add_argument("-w", "--weighted", type=int, default=1)
parser.add_argument("-D", "--deslope",  type=int, default=0)
args = parser.parse_args()

comm = mpi.COMM_WORLD
filedb.init()

ids = [line.split()[0] for line in open(args.idlist,"r")]
if args.ntod: ids = ids[:args.ntod]

is_dmap = os.path.isdir(args.imap)
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, rank=comm.rank)
tshape= (720,720)

# Read in all our scans
L.info("Reading %d scans" % len(ids))
myinds = np.arange(len(ids))[comm.rank::comm.size]
myinds, myscans = scanutils.read_scans(ids, myinds, actscan.ACTScan,
		filedb.data, dets=args.dets, downsample=config.get("downsample"))
myinds = np.array(myinds, int)

# Collect scan info. This currently fails if any task has empty myinds
read_ids  = [ids[ind] for ind in utils.allgatherv(myinds, comm)]
read_ntot = len(read_ids)
L.info("Found %d tods" % read_ntot)
if read_ntot == 0:
	L.info("Giving up")
	sys.exit(1)
# Prune fully autocut scans
mydets  = [len(scan.dets) for scan in myscans]
myinds  = [ind  for ind, ndet in zip(myinds, mydets) if ndet > 0]
myscans = [scan for scan,ndet in zip(myscans,mydets) if ndet > 0]
L.info("Pruned %d fully autocut tods" % (read_ntot - comm.allreduce(len(myscans))))

# Try to get about the same amount of data for each mpi task.
# If we use distributed maps, we also try to make things as local as possible
mycosts = [s.nsamp*s.ndet for s in myscans]
if is_dmap:
	myboxes = [scanutils.calc_sky_bbox_scan(s, args.sys) for s in myscans]
	myinds, mysubs, mybbox = scanutils.distribute_scans(myinds, mycosts, myboxes, comm)
else:
	myinds = scanutils.distribute_scans(myinds, mycosts, None, comm)

L.info("Rereading shuffled scans")
del myscans # scans do take up some space, even without the tod being read in
myinds, myscans = scanutils.read_scans(ids, myinds, actscan.ACTScan,
	filedb.data, dets=args.dets, downsample=config.get("downsample"))

L.info("Reading input map")
if not is_dmap: imap = enmap.read_map(args.imap)
else:           imap = dmap.read_map(args.imap, bbox=mybbox, tshape=tshape, comm=comm)
dtype = "=f" if imap.dtype == np.float32 else "=d"
imap = imap.astype(dtype, copy=True)

L.info("Initializing signals")
signal_cut = mapmaking.SignalCut(myscans, imap.dtype, comm=comm)
if not is_dmap:
	signal_map = mapmaking.SignalMap(myscans, imap, comm=comm)
	precon     = mapmaking.PreconMapBinned(signal_map, signal_cut, myscans, [], noise=False, hits=False)
else:
	signal_map = mapmaking.SignalDmap(myscans, mysubs, imap, sys=args.sys)
	precon     = mapmaking.PreconDmapBinned(signal_map, signal_cut, myscans, [], noise=False, hits=False)

# We can now actually perform the postfilter
L.info("Postfiltering")
postfilter = mapmaking.PostPickup(myscans, signal_map, signal_cut, precon, daz=args.daz, nt=args.nt, weighted=args.weighted, deslope=args.deslope)
omap = postfilter(imap)

# And output the resulting map
L.info("Writing")
if not is_dmap:
	if comm.rank == 0:
		enmap.write_map(args.omap, omap)
else:
	omap.write(args.omap)
