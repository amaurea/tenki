import numpy as np, time, os, sys
from pixell import utils, enmap, mpi, bunch
from enact import filedb, files, actscan, actdata
from enlib import config, scanutils, log, coordinates, mapmaking, dmap, errors

config.default("dmap_format", "merged")
config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
config.default("tod_window", 5.0, "Number of seconds to window the tod by on each end")
config.default("eig_limit", 0.1, "Pixel condition number below which polarization is dropped to make total intensity more stable. Should be a high value for single-tod maps to avoid thin stripes with really high noise")
config.default("map_sys", "cel", "Map coordinate system")

parser = config.ArgumentParser()
parser.add_argument("sel")
parser.add_argument("template")
parser.add_argument("odir")
parser.add_argument("-c", "--cont",            action="store_true")
parser.add_argument("-D", "--distributed",     type=int,   default=1)
args = parser.parse_args()

comm       = mpi.COMM_WORLD
ncomp     = 3
down      = config.get("downsample")
sys       = config.get("map_sys")
dtype     = {32:np.float32, 64:np.float64}[config.get("map_bits")]
log_level = log.verbosity2level(config.get("verbosity"))
L         = log.init(level=log_level, rank=comm_intra.rank, shared=True)

def build_maps(scans, shape, wcs, dtype=np.float32, sys="cel", comm=None, tag=None,
		distributed=False, niter=10, my_box=None):
	if comm is None: comm = mpi.COMM_WORLD
	pre = "" if tag is None else tag + " "
	L.info(pre + "Initializing equation system")
	signal_cut  = mapmaking.SignalCut(scans, dtype=dtype, comm=comm)
	if distributed:
		# hack: my_box has descending ra, but geometry assumes ascending ra
		if my_box is not None: my_box = flip_ra(my_box)
		geo     = dmap.geometry(shape, wcs, comm=comm, dtype=dtype, bbox=my_box)
		area    = dmap.zeros(geo)
		subinds = np.zeros(len(scans),int)
		signal_sky  = mapmaking.SignalDmap(scans, subinds, area, sys=sys, name="")
	else:
		area        = enmap.zeros(shape, wcs, dtype)
		signal_sky  = mapmaking.SignalMap(scans, area, comm=comm, sys=sys, name="")
	# This stuff is distribution-agnostic
	window      = mapmaking.FilterWindow(config.get("tod_window"))
	eqsys       = mapmaking.Eqsys(scans, [signal_cut, signal_sky], weights=[window], dtype=dtype, comm=comm)
	L.info(pre + "Building RHS")
	eqsys.calc_b()
	L.info(pre + "Building preconditioner")
	signal_cut.precon = mapmaking.PreconCut(signal_cut, scans)
	if distributed:
		signal_sky.precon = mapmaking.PreconDmapBinned(signal_sky, scans, [window])
	else:
		signal_sky.precon = mapmaking.PreconMapBinned(signal_sky, scans, [window])
	L.info(pre + "Solving")
	solver = cg.CG(eqsys.A, eqsys.b, M=eqsys.M, dot=eqsys.dot)
	while solver.i < niter:
		t1 = time.time()
		solver.step()
		t2 = time.time()
		L.info(pre + "CG step %5d %15.7e %6.1f %6.3f" % (solver.i, solver.err, (t2-t1), (t2-t1)/len(scans)))
	# Ok, now that we have our map. Extract it and ivar. That's the only stuff we need from this
	map  = eqsys.dof.unzip(solver.x)[1]
	ivar = signal_sky.precon.div[0,0]
	return bunch.Bunch(map=map, ivar=ivar, tmap=tmap, signal=signal_sky)

def abort(msg, show=True):
	if show: print(msg)
	sys.exit(1)

shape, wcs = enmap.read_map_geometry(args.template)
shape = (ncomp,)+shape[-2:]
niter = config.get("map_cg_nmax")
distributed = args.distributed>0

filedb.init()
ids = filedb.scans[args.sel]

# Read in our scans (minus the actual samples)
my_inds, scans = scanutils.read_scans(db.ids, my_inds, actscan.ACTScan, db=filedb.data, downsample=down)
nread = comm.allreduce(len(my_inds))
if nread == 0: abort("No usable tods found!", comm.rank==0)
# Remove any scans that didn't get anything to read from the communicator
active = len(my_inds) > 0
comm_active = comm.Split(active, comm.rank)
if active:
	# Actually do the heavy work of building the map
	maps = build_maps(scans, shape, wcs, dtype=dtype, comm=comm_active, sys=sys,
			distributed=distributed, niter=niter)
	# And write it out
	write_maps(prefix, maps)
comm.Barrier()
if comm.rank == 0:
	print("Done")
