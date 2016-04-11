# 1. Given a tod selector, run through them and classify them by scanning pattern.
#    A scanning pattern is defined as a bounding box in az and el.
# 2. For each scanning pattern, define a phase pixelization [ndet,{+,-},naz]
# 3. Sort tods in each scanning pattern by date. These tods collectively
#    make up an [ntod,ndet,nphase] data cube, but this would have a size of
#    roughly 1e3*1e3*1e3 = 1e9 pixels, which is a bit too big. So we will
#    output individual enmaps instead, as map_{el}_{az0}_{az1}_{pattern}_{id}.fits,
#    where id is the TOD id and pattern is the index into the list of patterns.
# 3. For each tod in a scanning pattern, read a partially calibrated TOD
#    and project it onto our map.

import numpy as np, os, h5py, sys, pipes, shutil, warnings
from enlib import config, errors, utils, log, bench, enmap, pmat, mapmaking, mpi, todfilter
from enlib.cg import CG
from enact import actdata, actscan, filedb, todinfo
warnings.filterwarnings("ignore")

config.default("downsample", 1, "Factor with which to downsample the TOD")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
config.default("map_bits", 32, "Bit-depth to use for maps and TOD")

parser = config.ArgumentParser(os.environ["HOME"]+"/.enkirc")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("prefix", nargs="?")
parser.add_argument("--tol",  type=float, default=10, help="Tolerance in arcmin for separating scanning patterns")
parser.add_argument("--daz",  type=float, default=2,  help="Pixel size in azimuth, in arcmin")
parser.add_argument("--nrow", type=int,   default=33)
parser.add_argument("--ncol", type=int,   default=32)
parser.add_argument("--nstep",type=int,   default=20)
parser.add_argument("--i0", type=int, default=None)
parser.add_argument("--i1", type=int, default=None)
parser.add_argument("-g,", "--group", type=int, default=1)
args = parser.parse_args()
filedb.init()

comm_world = mpi.COMM_WORLD
comm_group = comm_world
comm_sub   = mpi.COMM_SELF
ids  = todinfo.get_tods(args.sel, filedb.scans)
tol  = args.tol*utils.arcmin
daz  = args.daz*utils.arcmin
dtype = np.float32 if config.get("map_bits") == 32 else np.float64
tods_per_map = args.group

utils.mkdir(args.odir)
root = args.odir + "/" + (args.prefix + "_" if args.prefix else "")
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, rank=comm_world.rank, shared=False)

# Run through all tods to determine the scanning patterns
L.info("Detecting scanning patterns")
boxes = np.zeros([len(ids),2,2])
for ind in range(comm_world.rank, len(ids), comm_world.size):
	id    = ids[ind]
	entry = filedb.data[id]
	try:
		d = actdata.calibrate(actdata.read(entry, ["boresight","tconst"]))
	except errors.DataMissing as e:
		L.debug("Skipped %s (%s)" % (ids[ind], e.message))
		continue
	# Reorder from az,el to el,az
	boxes[ind] = [np.min(d.boresight[2:0:-1],1),np.max(d.boresight[2:0:-1],1)]
	L.info("%5d: %s" % (ind, id))
boxes = utils.allreduce(boxes, comm_world)

# Prune null boxes
usable = np.all(boxes!=0,(1,2))
ids, boxes = ids[usable], boxes[usable]

pattern_ids = utils.label_unique(boxes, axes=(1,2), atol=tol)
npattern = np.max(pattern_ids)+1
pboxes = np.array([utils.bounding_box(boxes[pattern_ids==pid]) for pid in xrange(npattern)])
pscans = [np.where(pattern_ids==pid)[0] for pid in xrange(npattern)]

L.info("Found %d scanning patterns" % npattern)

# Build the set of tasks we should go through. This effectively
# collapses these two loops, avoiding giving rank 0 much more
# to do than the last ranks.
tasks = []
for pid, group in enumerate(pscans):
	ngroup = (len(group)+tods_per_map-1)/tods_per_map
	for gind in range(ngroup):
		tasks.append([pid,gind,group[gind*tods_per_map:(gind+1)*tods_per_map]])

# Ok, run through each for real now
L.info("Building maps")
for pid, gind, group in tasks[comm_group.rank::comm_group.size]:
	scans = []
	for ind in group:
		id = ids[ind]
		L.info("%3d: %s" % (gind, id))
		entry = filedb.data[id]
		try:
			scan = actscan.ACTScan(entry)
			scan = scan[:,args.i0:args.i1]
			scan = scan[:,::config.get("downsample")]
			scans.append(scan)
		except errors.DataMissing as e:
			L.debug("Skipped %s (%s)" % (id, e.message))
			continue

	# Output name for this group
	proot=root + "pattern_%02d_el_%.1f_az_%.1f_%.1f_ind_%03d_" % (pid, pboxes[pid,0,0]/utils.degree,
			pboxes[pid,0,1]/utils.degree, pboxes[pid,1,1]/utils.degree, gind)
	# Record which ids went into this map
	with open(proot + "ids.txt", "w") as f:
		for ind in group:
			f.write("%s\n" % ids[ind])
	# Set up mapmaking for this group
	weights = [mapmaking.FilterWindow(config.get("tod_window"))] if config.get("tod_window") else []
	signal_cut   = mapmaking.SignalCut(scans, dtype, comm_sub)
	signal_phase = mapmaking.SignalPhase(scans, pids=[0]*len(scans), patterns=pboxes[pid:pid+1],
			array_shape=(args.nrow,args.ncol), res=daz, dtype=dtype, comm=comm_sub, cuts=signal_cut, ofmt="phase")
	signal_cut.precon   = mapmaking.PreconCut(signal_cut, scans)
	signal_phase.precon = mapmaking.PreconPhaseBinned(signal_phase, signal_cut, scans, weights)

	## Filter
	#class PickupFilter:
	#	def __init__(self, scans, pids, layout, templates):
	#		self.layout = layout
	#		self.templates = templates
	#	def __call__(self, scan, tod):

	def test_filter(scan, tod):
		return todfilter.filter_common_board(tod, scan.dets, scan.layout, name=scan.entry.id)

	eq = mapmaking.Eqsys(scans, [signal_cut, signal_phase], weights=weights, filters=[test_filter],
			dtype=dtype, comm=comm_sub)

	# Write precon
	signal_phase.precon.write(proot)
	# Solve for the given number of steps
	eq.calc_b()
	cg = CG(eq.A, eq.b, M=eq.M, dot=eq.dof.dot)
	while cg.i < args.nstep:
		with bench.mark("cg_step"):
			cg.step()
		dt = bench.stats["cg_step"]["time"].last
		if comm_sub.rank == 0:
			L.debug("CG step %5d %15.7e %6.1f %6.3f" % (cg.i, cg.err, dt, dt/max(1,len(eq.scans))))
	eq.write(proot, "map%04d" % cg.i, cg.x)

L.debug("Done")
