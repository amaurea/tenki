import numpy as np, time, h5py, copy, argparse, os, mpi4py.MPI, sys, pipes, shutil
from enlib import enmap, utils, pmat, fft, config, array_ops, map_equation, nmat, errors
from enlib import log, bench, scan
from enlib.cg import CG
from enact import data, nmat_measure, filedb

config.default("filedb", "filedb.txt", "File describing the location of the TOD and their metadata")
config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("downsample", 1, "Factor with which to downsample the TOD")
config.default("map_precon", "bin", "Preconditioner to use for map-making")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("filelist")
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("prefix",nargs="?")
parser.add_argument("--ncomp",      type=int, default=3)
parser.add_argument("--ndet",       type=int, default=0)
args = parser.parse_args()

precon= config.get("map_precon")
dtype = np.float32 if config.get("map_bits") == 32 else np.float64
comm  = mpi4py.MPI.COMM_WORLD
myid  = comm.rank
nproc = comm.size

comm_single = comm.split(np.arange(nproc))

db       = filedb.ACTFiles(config.get("filedb"))
# Allow filelist to take the format filename:[slice]
toks = args.filelist.split(":")
filelist, fslice = toks[0], ":".join(toks[1:])
filelist = [line.split()[0] for line in open(filelist,"r") if line[0] != "#"]
filelist = eval("filelist"+fslice)

area = enmap.read_map(args.area)
area = enmap.zeros((args.ncomp,)+area.shape[-2:], area.wcs, dtype)
utils.mkdir(args.odir)
if args.prefix:
	root = args.odir + "/" + args.prefix + "_"
else:
	root = args.odir + "/"

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
	shutil.copyfile(config.get("filedb"), root + "filedb.txt")
# Set up logging
utils.mkdir(root + "log")
logfile   = root + "log/log%03d.txt" % myid
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, file=logfile, rank=myid)
# And benchmarking
utils.mkdir(root + "bench")
benchfile = root + "bench/bench%03d.txt" % myid

# Read in all our scans
L.info("Reading scans")
inds = np.arange(len(filelist))[myid::nproc]
for ind in inds:
	try:
		d = scan.read_scan(filelist[ind])
	except IOError:
		try:
			d = data.ACTScan(db[filelist[ind]])
		except errors.DataMissing as e:
			L.debug("Skipped %s (%s)" % (filelist[ind], e.message))
			continue
	d = d[:,::config.get("downsample")]
	if args.ndet > 0: d = d[:args.ndet]
	L.debug("Read %s" % filelist[ind])

	eq = map_equation.LinearSystemMap([d], area, precon=precon, comm=comm_single)
	x  = eq.M(eq.b)
	bmap = eq.dof.unzip(x)[0]
	myroot = root + "_" + filelist[ind] + "_"
	eq.write(myroot)
	enmap.write_map(myroot + "bin.fits", bmap)
