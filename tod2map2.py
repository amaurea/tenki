import numpy as np, time, h5py, copy, argparse, os, mpi4py.MPI, sys, pipes, shutil, bunch
from enlib import enmap, utils, pmat, fft, config, array_ops, mapmaking, nmat, errors
from enlib import log, bench, dmap2 as dmap, coordinates, scan as enscan, rangelist, scanutils
from enlib.cg import CG
from enlib.source_model import SourceModel
from enact import data, nmat_measure, filedb, todinfo

config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("downsample", 1, "Factor with which to downsample the TOD")
config.default("map_precon", "bin", "Preconditioner to use for map-making")
config.default("map_eqsys",  "equ", "The coordinate system of the maps. Can be eg. 'hor', 'equ' or 'gal'.")
config.default("map_cg_nmax", 1000, "Max number of CG steps to perform in map-making")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
config.default("task_dist", "size", "How to assign scans to each mpi task. Can be 'plain' for myid:n:nproc-type assignment, 'size' for equal-total-size assignment. The optimal would be 'time', for equal total time for each, but that's not implemented currently.")
config.default("gfilter_jon", False, "Whether to enable Jon's ground filter.")
config.default("map_ptsrc_handling", "subadd", "How to handle point sources in the map. Can be 'none' for no special treatment, 'subadd' to subtract from the TOD and readd in pixel space, and 'sim' to simulate a pointsource-only TOD.")
config.default("map_ptsrc_eqsys", "cel", "Equation system the point source positions are specified in. Default is 'cel'")
config.default("map_format", "fits", "File format to use when writing maps. Can be 'fits', 'fits.gz' or 'hdf'.")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("filelist")
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("prefix",nargs="?")
parser.add_argument("-d", "--dump", type=str, default="1,2,5,10,20,50,100,200,300,400,500,600,800,1000,1200,1500,2000,3000,4000,5000,6000,8000,10000", help="CG map dump steps")
parser.add_argument("--ncomp",      type=int, default=3,  help="Number of stokes parameters")
parser.add_argument("--ndet",       type=int, default=0,  help="Max number of detectors")
parser.add_argument("--imap",       type=str,             help="Reproject this map instead of using the real TOD data. Format eqsys:filename")
parser.add_argument("--imap-op",    type=str, default='sim', help="What operation to do with imap. Can be 'sim' or 'sub'")
parser.add_argument("--dump-config", action="store_true", help="Dump the configuration file to standard output.")
parser.add_argument("--pickup-maps",  action="store_true", help="Whether to solve for pickup maps")
parser.add_argument("--nohor",       action="store_true", help="Assume that the mean of each horizontal line of the map is zero. This is useful for breaking degeneracies that come from solving for pickup and sky simultaneously")
args = parser.parse_args()

if args.dump_config:
	print config.to_str()
	sys.exit(0)

precon= config.get("map_precon")
dtype = np.float32 if config.get("map_bits") == 32 else np.float64
comm  = mpi4py.MPI.COMM_WORLD
myid  = comm.rank
nproc = comm.size
nmax  = config.get("map_cg_nmax")
ext   = config.get("map_format")
mapsys= config.get("map_eqsys")
distributed = False
tshape= (240,240)
nrow,ncol=33,32
#print "FIXME A"
#nrow,ncol=1,1
pickup_res = 2*utils.arcmin

filedb.init()
db = filedb.data
filelist = todinfo.get_tods(args.filelist, filedb.scans)

utils.mkdir(args.odir)
root = args.odir + "/" + (args.prefix + "_" if args.prefix else "")

# Dump our settings
if myid == 0:
	config.save(root + "config.txt")
	with open(root + "args.txt","w") as f:
		f.write(" ".join([pipes.quote(a) for a in sys.argv[1:]]) + "\n")
	with open(root + "env.txt","w") as f:
		for k,v in os.environ.items():
			f.write("%s: %s\n" %(k,v))
	with open(root + "ids.txt","w") as f:
		for id in filelist:
			f.write("%s\n" % id)
	shutil.copyfile(filedb.cjoin(["root","dataset","filedb"]),  root + "filedb.txt")
	try: shutil.copyfile(filedb.cjoin(["root","dataset","todinfo"]), root + "todinfo.txt")
	except IOError: pass
# Set up logging
utils.mkdir(root + "log")
logfile   = root + "log/log%03d.txt" % myid
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, file=logfile, rank=myid)
# And benchmarking
utils.mkdir(root + "bench")
benchfile = root + "bench/bench%03d.txt" % myid

def read_scans(filelist, tmpinds, db=None, ndet=0, quiet=False):
	"""Given a set of ids/files and a set of indices into that list. Try
	to read each of these scans. Returns a list of successfully read scans
	and a list of their indices."""
	myscans, myinds  = [], []
	for ind in tmpinds:
		try:
			d = enscan.read_scan(filelist[ind])
		except IOError:
			try:
				d = data.ACTScan(db[filelist[ind]])
			except errors.DataMissing as e:
				if not quiet: L.debug("Skipped %s (%s)" % (filelist[ind], e.message))
				continue
		d = d[:,::config.get("downsample")]
		if ndet > 0: d = d[:ndet]
		myscans.append(d)
		myinds.append(ind)
		if not quiet: L.debug("Read %s" % filelist[ind])
	return myscans, myinds

# Read in all our scans
L.info("Reading %d scans" % len(filelist))
myscans, myinds = read_scans(filelist, np.arange(len(filelist))[myid::nproc], db, ndet=args.ndet)

# Collect scan info
read_ids  = [filelist[ind] for ind in utils.allgatherv(myinds, comm)]
read_ntot = len(read_ids)
L.info("Found %d tods" % read_ntot)
if read_ntot == 0:
	L.info("Giving up")
	sys.exit(1)
read_ndets= utils.allgatherv([len(scan.dets) for scan in myscans], comm)
read_dets = utils.uncat(utils.allgatherv(np.concatenate([scan.dets for scan in myscans]),comm), read_ndets)
# Save accept list
if myid == 0:
	with open(root + "accept.txt", "w") as f:
		for id, dets in zip(read_ids, read_dets):
			f.write("%s %3d: " % (id, len(dets)) + " ".join([str(d) for d in dets]) + "\n")

# Try to get about the same amount of data for each mpi task,
# all at roughly the same part of the sky.
mycosts = [s.nsamp*s.ndet for s in myscans]
myboxes = [scanutils.calc_sky_bbox_scan(s, mapsys) for s in myscans]
myinds, mysubs, mybbox = scanutils.distribute_scans(myinds, mycosts, myboxes, comm)

# And reread the correct files this time. Ideally we would
# transfer this with an mpi all-to-all, but then we would
# need to serialize and unserialize lots of data, which
# would require lots of code.
L.info("Rereading shuffled scans")
myscans, myinds = read_scans(filelist, myinds, db, ndet=args.ndet)

#print "FIXME B"
#myscans[0].dets = np.arange(4)
#myscans[0].cut.clear()

L.info("Initializing signals")
signals = []
# Cuts
signal_cut = mapmaking.SignalCut(myscans, dtype, comm)
signal_cut.precon = mapmaking.PreconCut(signal_cut, myscans)
# Disabling cuts here, but using it in precon, stops convergence
# (it jumps randomly up and down between 1e-5 and 1e-2)
signals.append(signal_cut)
# Main maps
if True:
	if distributed:
		area = dmap.read_map(args.area, bbox=mybbox, tshape=tshape, comm=comm)
		area = dmap.zeros(area.geometry.aspre(args.ncomp).astype(dtype))
		signal_map = mapmaking.SignalDmap(myscans, mysubs, area, cuts=signal_cut)
		signal_map.precon = mapmaking.PreconDmapBinned(signal_map, myscans)
		if args.nohor:
			signal_map.prior = mapmaking.PriorDmapNohor(signal_map.precon.div[0,0])
	else:
		area = enmap.read_map(args.area)
		area = enmap.zeros((args.ncomp,)+area.shape[-2:], area.wcs, dtype)
		signal_map = mapmaking.SignalMap(myscans, area, comm, cuts=signal_cut)
		signal_map.precon = mapmaking.PreconMapBinned(signal_map, myscans)
		if args.nohor:
			signal_map.prior = mapmaking.PriorMapNohor(signal_map.precon.div[0,0])
	signals.append(signal_map)
# Pickup maps
if args.pickup_maps:
	# Classify scanning patterns
	patterns, mypids = scanutils.classify_scanning_patterns(myscans, comm=comm)
	L.info("Found %d scanning patterns" % len(patterns))
	signal_pickup = mapmaking.SignalPhase(myscans, mypids, patterns, (nrow,ncol), pickup_res, cuts=signal_cut, dtype=dtype, comm=comm)
	signal_pickup.precon = mapmaking.PreconPhaseBinned(signal_pickup, myscans)
	signals.append(signal_pickup)

mapmaking.write_precons(signals, root)
L.info("Initializing equation system")
eqsys = mapmaking.Eqsys(myscans, signals, dtype, comm)

#print "FIXME C"
#print eqsys.dof.n
#A = eqsys.calc_A()
#M = eqsys.calc_M()
#eqsys.calc_b()
#b = eqsys.b
#with h5py.File(root + "eqsys.hdf","w") as hfile:
#	hfile["A"] = A
#	hfile["M"] = M
#	hfile["b"] = b
#sys.exit(0)


#m = enmap.rand_gauss(area.shape, area.wcs, area.dtype)
#miwork = signal_map.prepare(m)
#mowork = signal_map.prepare(signal_map.zeros())
#p = signal_pickup.zeros()
#powork = signal_pickup.prepare(signal_pickup.zeros())
#for scan in myscans:
#	tod = np.zeros([scan.ndet, scan.nsamp], m.dtype)
#	signal_map.forward(scan, tod, miwork)
#	signal_pickup.backward(scan, tod, powork)
#signal_pickup.finish(p, powork)
#signal_pickup.precon(p)
#piwork = signal_pickup.prepare(p)
#for scan in myscans:
#	tod = np.zeros([scan.ndet, scan.nsamp], m.dtype)
#	signal_pickup.forward(scan, tod, piwork)
#	signal_map.backward(scan, tod, mowork)
#signal_map.finish(m, mowork)
#signal_map.precon(m)
#
#if comm.rank == 0:
#	enmap.write_map(root + "test.fits", m)
#sys.exit(0)


#nnocut = eqsys.dof.n - signal_cut.dof.n
#print nnocut, signal_map.dof.n, signal_pickup.dof.n
#inds = np.arange(-nnocut,-1,nnocut/30)
#inds = np.arange(-5620117-3,-5620117+3)
#eqsys.check_symmetry(inds)
#sys.exit(0)

eqsys.calc_b()
eqsys.write(root, "rhs", eqsys.b)
L.info("Computing approximate map")
x = eqsys.M(eqsys.b)
eqsys.write(root, "bin", x)

if nmax > 0:
	L.info("Solving")
	cg = CG(eqsys.A, eqsys.b, M=eqsys.M, dot=eqsys.dof.dot)
	dump_steps = [int(w) for w in args.dump.split(",")]
	while cg.i < nmax:
		with bench.mark("cg_step"):
			cg.step()
		dt = bench.stats["cg_step"]["time"].last
		L.info("CG step %5d %15.7e %6.1f %6.3f" % (cg.i, cg.err, dt, dt/max(1,len(eqsys.scans))))
		if cg.i in dump_steps or cg.i % dump_steps[-1] == 0:
			eqsys.write(root, "map%04d" % cg.i, cg.x)
		bench.stats.write(benchfile)
