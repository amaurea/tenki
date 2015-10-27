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
from enlib import config, errors, utils, log, bench, enmap, pmat, map_equation, mpi
from enlib.cg import CG
from enact import data, filedb, todinfo
warnings.filterwarnings("ignore")

config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
config.default("map_bits", 32, "Bit-depth to use for maps and TOD")

parser = config.ArgumentParser(os.environ["HOME"]+"/.enkirc")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("prefix", nargs="?")
parser.add_argument("--tol",  type=float, default=1, help="Tolerance in degrees for separating scanning patterns")
parser.add_argument("--daz",  type=float, default=1, help="Pixel size in azimuth")
parser.add_argument("--nrow", type=int,   default=33)
parser.add_argument("--ncol", type=int,   default=32)
parser.add_argument("--nstep",type=int,   default=20)
parser.add_argument("-g,", "--group", type=int, default=1)
args = parser.parse_args()
filedb.init()

comm_world = mpi.COMM_WORLD
comm_group = comm_world
comm_sub   = mpi.COMM_SELF
ids  = todinfo.get_tods(args.sel, filedb.scans)
ndet = args.nrow*args.ncol
tol  = args.tol*utils.degree
daz  = args.daz*utils.arcmin
dtype = np.float32 if config.get("map_bits") == 32 else np.float64
tods_per_map = args.group

# Define our target detector ordering. This is a mapping from
# col-major to row-major
row_major = np.arange(ndet)
row, col = row_major/args.ncol, row_major%args.ncol
col_major = col*args.nrow + row

utils.mkdir(args.odir)
root = args.odir + "/" + (args.prefix + "_" if args.prefix else "")
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, rank=comm_world.rank)

# Run through all tods to determine the scanning patterns
L.info("Detecting scanning patterns")
boxes = np.zeros([len(ids),2,2])
for ind in range(comm_world.rank, len(ids), comm_world.size):
	id    = ids[ind]
	entry = filedb.data[id]
	try:
		d = data.calibrate(data.read(entry, ["boresight"]))
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
pscans = [ids[pattern_ids==pid] for pid in xrange(npattern)]

L.info("Found %d scanning patterns" % npattern)

# Build the set of tasks we should go through. This effectively
# collapses these two loops, avoiding giving rank 0 much more
# to do than the last ranks.
tasks = []
for pid, group in enumerate(pscans):
	for ind in range(0, len(group)/tods_per_map):
		tasks.append([pid,group,ind])

# Ok, run through each for real now
L.info("Building maps")
for pid, group, ind in tasks[comm_group.rank::comm_group.size]:
	scans = []
	subgroup = group[ind*tods_per_map:(ind+1)*tods_per_map]
	for id in subgroup:
		L.info("%3d: %s" % (ind, id))
		entry = filedb.data[id]
		try:
			scans.append(data.ACTScan(entry))
		except errors.DataMissing as e:
			L.debug("Skipped %s (%s)" % (id, e.message))
			continue
	# Define pixels for this tod
	az0, az1 = pboxes[pid,:,1]
	naz = np.ceil((az1-az0)/daz)
	az1 = az0 + naz*daz
	shape, wcs = enmap.geometry(pos=[[0,az0],[args.ncol*utils.degree,az1]], shape=(ndet,naz), proj="car")
	area = enmap.zeros((2,)+shape, wcs, dtype=dtype)

	prefix = root + "pattern_%02d_el_%d_az_%d_%d_ind_%03d_" % (pid, np.round(np.mean(pboxes[pid,:,0])/utils.degree),
			np.round(pboxes[pid,0,1]/utils.degree), np.round(pboxes[pid,1,1]/utils.degree), ind)
	# Record which ids went into this map
	with open(prefix + "ids.txt", "w") as f:
		for id in subgroup: f.write("%s\n"%id)

	eq = map_equation.LinearSystemAz(scans, area, ordering=col_major, comm=comm_sub)
	eq.write(prefix)
	cg = CG(eq.A, eq.b, M=eq.M, dot=eq.dof.dot)
	while cg.i < args.nstep:
		with bench.mark("cg_step"): cg.step()
		dt = bench.stats["cg_step"]["time"].last
		if comm_sub.rank == 0:
			L.debug("CG step %5d %15.7e %6.1f %6.3f" % (cg.i, cg.err, dt, dt/max(1,len(eq.scans))))
	if comm_sub.rank == 0:
		map = eq.dof.unzip(cg.x)[0]
		enmap.write_map(prefix + "map.fits", map)

L.debug("Done")
