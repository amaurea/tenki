import numpy as np, argparse, os, sys, pipes, shutil, warnings
from enlib import utils, pmat, config, errors, mpi
from enlib import log, bench, scan, ptsrc_data
from enact import actscan, filedb, todinfo

warnings.filterwarnings("ignore")

config.default("filedb", "filedb.txt", "File describing the location of the TOD and their metadata")
config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("filelist")
parser.add_argument("srcs")
parser.add_argument("odir")
parser.add_argument("--ncomp",      type=int,   default=3)
parser.add_argument("--ndet",       type=int,   default=0)
parser.add_argument("--minsigma",   type=float, default=4)
parser.add_argument("-c", action="store_true")
args = parser.parse_args()

dtype = np.float32 if config.get("map_bits") == 32 else np.float64
comm  = mpi.COMM_WORLD
myid  = comm.rank
nproc = comm.size

filedb.init()
db = filedb.data
filelist = todinfo.get_tods(args.filelist, filedb.scans)

def compress_beam(sigma, phi):
	c,s=np.cos(phi),np.sin(phi)
	R = np.array([[c,-s],[s,c]])
	C = np.diag(sigma**-2)
	C = R.dot(C).dot(R.T)
	return np.array([C[0,0],C[1,1],C[0,1]])

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
	shutil.copyfile(filedb.cjoin(["root","dataset","filedb"]),  args.odir + "/filedb.txt")
	try: shutil.copyfile(filedb.cjoin(["root","dataset","todinfo"]), args.odir + "/todinfo.txt")
	except IOError: pass
# Set up logging
utils.mkdir(args.odir + "/log")
logfile   = args.odir + "/log/log%03d.txt" % myid
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, file=logfile, rank=myid, shared=False)
# And benchmarking
utils.mkdir(args.odir + "/bench")
benchfile = args.odir + "/bench/bench%03d.txt" % myid

# Read our point source list in the format produced by sort_sources
arcmin = np.pi/180/60
fwhm   = arcmin/(8*np.log(2))**0.5
srcs   = np.loadtxt(args.srcs).T
totsrcs= srcs.shape[1]
sigma  = np.abs(srcs[2])
pos    = srcs[[3,5]]*np.pi/180
amps   = srcs[[7,9,11]][:args.ncomp]
beam   = srcs[[19,21,23]]
ibeam  = np.array([compress_beam(b[:2]*fwhm,b[2]*np.pi/180) for b in beam.T]).T
params = np.vstack([pos,amps,ibeam]).astype(dtype)
# Eliminate insignificant sources
params = params[:,sigma>args.minsigma]
srcs   = srcs  [:,sigma>args.minsigma]

if comm.rank == 0:
	L.info("Got %d sources, keeping %d > %d sigma" % (totsrcs,params.shape[1],args.minsigma))
	np.savetxt(args.odir + "/srcs.txt", srcs.T, fmt="%12.5f")

# Our noise model is slightly different from the main noise model,
# since we assume it is white and independent between detectors,
# which is not strictly true. To minimize error, we measure the
# noise level using a method which is as close as possible to
# how we will use it later
def blockify(tod, w): return tod[:tod.size/w*w].reshape(-1,w)
def get_desloped_var(blocks):
	n   = blocks.shape[1]
	s   = np.arange(n) - 0.5*(n-1)
	ss  = np.sum(s**2)
	m   = np.mean(blocks,1)
	sd  = np.einsum("at,t->a",blocks,s)
	A   = sd/ss
	return np.mean((blocks-m[:,None]-A[:,None]*s[None,:])**2,1)
def onlyfinite(a): return a[np.isfinite(a)]

# Process each scan independently
myinds = np.arange(len(filelist))[myid::nproc]
for ind in myinds:
	ofile = args.odir + "/%s.hdf" % filelist[ind]
	if args.c and os.path.isfile(ofile): continue
	L.info("Processing %s" % filelist[ind])
	try:
		d = scan.read_scan(filelist[ind])
	except IOError:
		try:
			d = actscan.ACTScan(db[filelist[ind]])
		except errors.DataMissing as e:
			L.debug("Skipped %s (%s)" % (filelist[ind], e.message))
			continue

	try:
		# Set up pmat for this scan
		L.debug("Reading samples")
		tod   = d.get_samples().astype(dtype)
		d.noise = d.noise.update(tod, d.srate)
		L.debug("Pmats")
		Psrc  = pmat.PmatPtsrc(d, params)
		Pcut  = pmat.PmatCut(d)
	except errors.DataMissing as e:
		L.debug("Skipped %s (%s)" % (filelist[ind], e.message))
		continue

	# Measure noise
	L.debug("Noise")
	ivar = 1/np.array([np.median(onlyfinite(get_desloped_var(blockify(t,20)))) for t in tod])
	# Extract point-source relevant TOD properties
	L.debug("Extract data")
	srcdata = Psrc.extract(tod)
	# Kill detectors where the noise measured in the actually kept samples is too
	# far off from the noise measured overall. This won't work for planets!
	if False:
		L.debug("Measure white noise")
		vars, nvars = ptsrc_data.measure_mwhite(srcdata.tod, srcdata)
		ivar_src = np.sum(nvars,0)/np.sum(vars,0)
		#for v,vs in zip(ivar,ivar_src):
		#	print "%12.2f %12.2f" % (v**-0.5, vs**-0.5)
		bad = np.log(ivar/ivar_src)**2 > np.log(1.5)**2
		ivar[bad] *= 0.01

	#srcdata.ivars = d.noise.iD[-1]
	srcdata.ivars = ivar
	L.debug("Writing")
	ptsrc_data.write_srcscan(ofile, srcdata)
