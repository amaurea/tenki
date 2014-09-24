import numpy as np, time, h5py, copy, argparse, os, mpi4py.MPI, sys, pipes, shutil
from enlib import enmap, utils, pmat, fft, config, array_ops, map_equation, nmat, errors
from enlib import log, bench
from enlib.cg import CG
from enact import data, nmat_measure, filedb

config.default("filedb", "filedb.txt", "File describing the location of the TOD and their metadata")
config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("downsample", 1, "Factor with which to downsample the TOD")
config.default("map_precon", "bin", "Preconditioner to use for map-making")
config.default("map_cg_nmax", 1000, "Max number of CG steps to perform in map-making")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
config.default("task_dist", "size", "How to assign scans to each mpi task. Can be 'plain' for myid:n:nproc-type assignment, 'size' for equal-total-size assignment. The optimal would be 'time', for equal total time for each, but that's not implemented currently.")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("filelist")
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("-d", "--dump", type=int, default=10)
parser.add_argument("--ncomp",      type=int, default=3)
parser.add_argument("--ndet",       type=int, default=0)
args = parser.parse_args()

precon= config.get("map_precon")
dtype = np.float32 if config.get("map_bits") == 32 else 64
comm  = mpi4py.MPI.COMM_WORLD
myid  = comm.rank
nproc = comm.size
nmax  = config.get("map_cg_nmax")

db       = filedb.ACTdb(config.get("filedb"))
# Allow filelist to take the format filename:[slice]
toks = args.filelist.split(":")
filelist, fslice = toks[0], ":".join(toks[1:])
filelist = [line.split()[0] for line in open(filelist,"r") if line[0] != "#"]
filelist = eval("filelist"+fslice)

area = enmap.read_map(args.area)
area = enmap.zeros((args.ncomp,)+area.shape[-2:], area.wcs, dtype)
utils.mkdir(args.odir)
# Dump our settings
if myid == 0:
	config.save(args.odir + "/config.txt")
	with open(args.odir + "/args.txt","w") as f:
		f.write(" ".join([pipes.quote(a) for a in sys.argv[1:]]) + "\n")
	with open(args.odir + "/env.txt","w") as f:
		for k,v in os.environ.items():
			f.write("%s: %s\n" %(k,v))
	with open(args.odir + "/ids.txt","w") as f:
		for id in filelist:
			f.write("%s\n" % id)
	shutil.copyfile(config.get("filedb"), args.odir + "/filedb.txt")
# Set up logging
utils.mkdir(args.odir + "/log")
logfile   = args.odir + "/log/log%03d.txt" % myid
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, file=logfile, rank=myid)
# And benchmarking
utils.mkdir(args.odir + "/bench")
benchfile = args.odir + "/bench/bench%03d.txt" % myid

# Read in all our scans
L.info("Reading scans")
tmpinds    = np.arange(len(filelist))[myid::nproc]
myscans, myinds  = [], []
for ind in tmpinds:
	try:
		d = data.ACTScan(db[filelist[ind]])[:,::config.get("downsample")]
		if args.ndet > 0: d = d[:args.ndet]
		myscans.append(d)
		myinds.append(ind)
		L.debug("Read %s" % filelist[ind])
	except errors.DataMissing as e:
		L.debug("Skipped %s (%s)" % (filelist[ind], e.message))

nread = comm.allreduce(len(myscans))
L.info("Found %d tods" % nread)
if nread == 0:
	L.info("Giving up")
	sys.exit(1)

if config.get("task_dist") == "size":
	# Try to get about the same amount of data for each mpi task.
	mycosts = [s.nsamp*s.ndet for s in myscans]
	all_costs = comm.allreduce(mycosts)
	all_inds  = comm.allreduce(myinds)
	myinds_old = myinds
	myinds = [all_inds[i] for i in utils.equal_split(all_costs, nproc)[myid]]
	# And reread the correct files this time. Ideally we would
	# transfer this with an mpi all-to-all, but then we would
	# need to serialize and unserialize lots of data, which
	# would require lots of code.
	if sorted(myinds_old) != sorted(myinds):
		L.info("Rereading shuffled scans")
		myscans = []
		for ind in myinds:
			d = data.ACTScan(db[filelist[ind]])[:,::config.get("downsample")]
			if args.ndet > 0: d = d[:args.ndet]
			myscans.append(d)
			L.debug("Read %s" % filelist[ind])
	else:
		myinds = myinds_old

L.info("Building equation system")
eq  = map_equation.LinearSystemMap(myscans, area, precon=precon)
eq.write(args.odir)
bench.stats.write(benchfile)

L.info("Computing approximate map")
x  = eq.M(eq.b)
bmap = eq.dof.unzip(x)[0]
if myid == 0: enmap.write_map(args.odir + "/bin.fits", bmap)

def solve_cg(eq, nmax=1000, ofmt=None, dump_interval=10):
	cg = CG(eq.A, eq.b, M=eq.M, dot=eq.dof.dot)
	while cg.i < nmax:
		with bench.mark("cg_step"):
			cg.step()
		dt = bench.stats["cg_step"]["time"].last
		L.info("CG step %5d %15.7e %6.1f %6.3f" % (cg.i, cg.err, dt, dt/len(eq.scans)))
		xmap, xjunk = eq.dof.unzip(cg.x)
		if ofmt and cg.i % dump_interval == 0 and myid == 0:
			enmap.write_map(ofmt % cg.i, eq.dof.unzip(cg.x)[0])
		# Output benchmarking information
		bench.stats.write(benchfile)
	return cg.x

if nmax > 0:
	L.info("Solving equation")
	x = solve_cg(eq, nmax=nmax, ofmt=args.odir + "/dump%04d.fits", dump_interval=args.dump)
