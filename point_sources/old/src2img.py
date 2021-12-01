import numpy as np, argparse, mpi4py.MPI, os, h5py
from enlib import utils, config, ptsrc_data, log

config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("filelist")
parser.add_argument("srcs")
parser.add_argument("odir")
parser.add_argument("-R", "--radius", type=float, default=5.0)
parser.add_argument("-r", "--resolution", type=float, default=0.25)
args = parser.parse_args()

comm  = mpi4py.MPI.COMM_WORLD
myid  = comm.rank
nproc = comm.size

log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, rank=myid)

# Allow filelist to take the format filename:[slice]
toks = args.filelist.split(":")
filelist, fslice = toks[0], ":".join(toks[1:])
filelist = [line.split()[0] for line in open(filelist,"r") if line[0] != "#"]
filelist = eval("filelist"+fslice)

utils.mkdir(args.odir)
srcs = np.loadtxt(args.srcs).T

# create minimaps around each source
nsrc  = srcs.shape[1]
ncomp = 1
n = int(np.round(2*args.radius/args.resolution))
R = args.radius*np.pi/180/60
boxes = np.array([[s[[3,5]]-R,s[[3,5]]+R] for s in srcs.T*np.pi/180])

myinds = np.arange(len(filelist))[myid::nproc]
for ind in myinds:
	id   = os.path.basename(filelist[ind])[:-4]
	L.info("Processing %s" % id)
	maps = np.zeros([nsrc,ncomp,n,n],dtype=np.float32)
	div  = np.zeros([ncomp,nsrc,ncomp,n,n],dtype=np.float32)

	d = ptsrc_data.read_srcscan(filelist[ind])
	# Try the most significant src
	ptsrc_data.nmat_mwhite(d.tod, d)
	ptsrc_data.pmat_thumbs(-1, d.tod, maps, d.point, d.phase, boxes)

	for c in range(ncomp):
		idiv = div[0].copy(); idiv[:,c] = 1
		tod = d.tod.copy(); tod[...] = 0
		ptsrc_data.pmat_thumbs( 1, tod, idiv, d.point, d.phase, boxes)
		ptsrc_data.nmat_mwhite(tod, d, 0.0)
		ptsrc_data.pmat_thumbs(-1, tod, div[c], d.point, d.phase, boxes)
	div = np.rollaxis(div,1)
	bin = maps/div[:,0]

	# Maps is already multiplied by div, so don't overmultiply
	scale  = srcs[7,:,None,None,None]
	sm=maps*scale;sd=div*scale[:,None]**2
	mstack = np.sum(sm,0)
	dstack = np.sum(sd,0)
	bstack = mstack/dstack[0]*scale[0]

	with h5py.File(args.odir + "/maps_%s.hdf" % id,"w") as hfile:
		hfile["data"]  = bin
		hfile["stack"] = bstack
