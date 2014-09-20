import numpy as np, time, h5py, copy, argparse, os, mpi4py.MPI, sys, pipes, shutil
from enlib import enmap, utils, pmat, fft, config, array_ops, map_equation, nmat, errors
from enlib import scansim
from enlib.cg import CG
from enact import data, nmat_measure, filedb

config.default("filedb", "filedb.txt", "File describing the location of the TOD and their metadata")
config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("downsample", 1, "Factor with which to downsample the TOD")
config.default("map_precon", "bin", "Preconditioner to use for map-making")
config.default("map_cg_nmax", 1000, "Max number of CG steps to perform in map-making")

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

if myid == 0: print "Reading scans"
tmpinds    = np.arange(len(filelist))[myid::nproc]
myscans, myinds  = [], []
for ind in tmpinds:
	try:
		d = data.ACTScan(db[filelist[ind]])[:,::config.get("downsample")]
		if args.ndet > 0: d = d[:args.ndet]
		myscans.append(d)
		myinds.append(ind)
		print "Read %s" % filelist[ind]
	except errors.DataMissing as e:
		print "Skipped %s (%s)" % (filelist[ind], e.message)
		pass

nread = comm.allreduce(len(myscans))
if myid == 0: print "Found %d tods" % nread
if nread == 0:
	if myid == 0: print "Giving up"
	sys.exit(1)

if myid == 0: print "Building equation system"
eq  = map_equation.LinearSystemMap(myscans, area, precon=precon)
eq.write(args.odir)

if myid == 0: print "Computing approximate map"
x  = eq.M(eq.b)
bmap = eq.dof.unzip(x)[0]
if myid == 0: enmap.write_map(args.odir + "/bin.fits", bmap)

def solve_cg(eq, nmax=1000, ofmt=None, dump_interval=10):
	cg = CG(eq.A, eq.b, M=eq.M, dot=eq.dof.dot)
	while cg.i < nmax:
		t1 = time.time()
		cg.step()
		t2 = time.time()
		if myid == 0: print "%5d %15.7e %6.3f %6.3f" % (cg.i, cg.err, t2-t1, (t2-t1)/len(eq.scans))
		xmap, xjunk = eq.dof.unzip(cg.x)
		if ofmt and cg.i % dump_interval == 0 and myid == 0:
			enmap.write_map(ofmt % cg.i, eq.dof.unzip(cg.x)[0])
	return cg.x

if nmax > 0:
	if myid == 0: print "Solving equation"
	x = solve_cg(eq, nmax=nmax, ofmt=args.odir + "/dump%04d.fits", dump_interval=args.dump)
