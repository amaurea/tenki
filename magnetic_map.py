"""Instead of mapping pixels on the sky, creates a map in AZ for each detector
for each scanning pattern. Scanning patterns are identified by mean az, mean el,
delta az, delta el."""
import numpy as np, os, mpi4py.MPI, h5py
from enlib import config, errors, utils, log, map_equation, bench
from enlib.cg import CG
from enact import data, filedb

config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
config.default("map_cg_nmax", 1000, "Max number of CG steps to perform in map-making")
config.default("downsample", 1, "Factor with which to downsample the TOD")
config.default("map_bits", 32, "Bit-depth to use for maps and TOD")

parser = config.ArgumentParser(os.environ["HOME"]+"/.enkirc")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("prefix", nargs="?")
parser.add_argument("--pos-tol", type=float, default=2)
parser.add_argument("--amp-tol", type=float, default=0.5)
parser.add_argument("--daz",     type=float, default=1)
parser.add_argument("--ndet",    type=int,   default=0)
parser.add_argument("-d", "--dump", type=str, default="1,2,5,10,20,50,100,200,300,400,500,600,800,1000,1200,1500,2000,3000,4000,5000,6000,8000,10000", help="CG map dump steps")
args = parser.parse_args()
filedb.init()

comm = mpi4py.MPI.COMM_WORLD
db = filedb.scans[args.sel]
ids = db.ids
ndet_max = 33*32
pos_tol = args.pos_tol*utils.degree
amp_tol = args.amp_tol*utils.degree
dtype = np.float32 if config.get("map_bits") == 32 else np.float64

utils.mkdir(args.odir)
root = args.odir + "/" + (args.prefix + "_" if args.prefix else "")
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, rank=comm.rank)

azbox  = np.array([-np.pi/2,np.pi/2])
daz    = args.daz*utils.arcmin
naz    = int(np.ceil((azbox[1]-azbox[0])/daz))
nmax   = config.get("map_cg_nmax")

def mid_dev(a):
	mi = np.min(a)
	ma = np.max(a)
	return (ma+mi)/2, (ma-mi)/2

# Read in my scans
myinds_ideal  = np.arange(comm.rank, len(ids), comm.size)
myinds, myscans, myinfo = [], [], []
L.info("Reading %d scans" % len(ids))
for ind in myinds_ideal:
	try:
		entry = filedb.data[ids[ind]]
		d = data.ACTScan(entry)
	except errors.DataMissing as e:
		L.debug("Skipped %s (%s)" % (ids[ind], e.message))
		continue
	d = d[:,::config.get("downsample")]
	if args.ndet > 0: d = d[:args.ndet]
	myinds.append(ind)
	myscans.append(d)
	# Extract boresight info for each scan
	az, el = d.boresight.T[1:3]
	m_az, d_az = mid_dev(az)
	m_el, d_el = mid_dev(el)
	myinfo.append([m_az,m_el,d_az,d_el])
	L.debug("Read %s" % ids[ind])

myinds = np.array(myinds)
myinfo = np.array(myinfo)

# Reduce
inds = utils.allgatherv(myinds, comm)
info = utils.allgatherv(myinfo, comm)

# Build group mapping ind2group, which maps from index to group number. -1 indicates
# invalid group, but should not be encountered
nscan= len(inds)
ownership = np.zeros(nscan,dtype=int)
for i in xrange(nscan):
	unclaimed = np.where(ownership==0)[0]
	if len(unclaimed) == 0: break
	uinfo = info[unclaimed]
	# Find distance of every scan from the first one
	dists = utils.angdist(uinfo.T[:2], uinfo.T[:2,0])
	# Find difference in amplitudes
	adists = np.sum((uinfo[:,2:]-uinfo[0,2:])**2,1)**0.5
	group = (dists<pos_tol)&(adists<amp_tol)
	ownership[unclaimed[group]] = i+1
ind2group = np.zeros(len(ids),dtype=int)
ind2group[inds] = ownership-1
ngroup = np.max(ind2group)+1
L.info("Found %d scanning patterns" % ngroup)

mygroups = ind2group[myinds]

# Build signal cube
azmaps = np.zeros([ngroup,ndet_max,naz], dtype=dtype)

L.info("Building equation system")
eq  = map_equation.LinearSystemAz(myscans, azmaps, mygroups, azbox[0], daz, comm=comm)
eq.write(root)

L.info("Computing approximate map")
x  = eq.M(eq.b)
bmap = eq.dof.unzip(x)[0]
if comm.rank == 0:
	with h5py.File(root + "bin.hdf","w") as hfile:
		hfile["data"] = bmap
del bmap

if nmax > 0:
	L.info("Solving equation")
	dump = [int(w) for w in args.dump.split(",")]

	cg = CG(eq.A, eq.b, M=eq.M, dot=eq.dof.dot)
	while cg.i < nmax:
		with bench.mark("cg_step"):
			cg.step()
		dt = bench.stats["cg_step"]["time"].last
		L.info("CG step %5d %15.7e %6.1f %6.3f" % (cg.i, cg.err, dt, dt/max(1,len(eq.scans))))
		if comm.rank == 0 and (cg.i in dump or cg.i % dump[-1] == 0):
			map = eq.dof.unzip(cg.x)[0]
			with h5py.File(root + "map%04d.hdf" % cg.i, "w") as hfile:
				hfile["data"] = map
