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
parser.add_argument("prefix", nargs="?")
parser.add_argument("-c", "--cont",         action="store_true")
parser.add_argument("-D", "--distributed",  type=int, default=1)
parser.add_argument(      "--dets",         type=str, default=0,  help="Detector slice/listfile")
parser.add_argument("-G", "--nogfit",       action="store_true")
args = parser.parse_args()

comm       = mpi.COMM_WORLD
ncomp     = 3
down      = config.get("downsample")
mapsys    = config.get("map_sys")
dtype     = {32:np.float32, 64:np.float64}[config.get("map_bits")]
log_level = log.verbosity2level(config.get("verbosity"))
L         = log.init(level=log_level, rank=comm.rank, shared=True)

def solve_joint_gain_sky(scans, shape, wcs, dtype=np.float32, sys="cel", comm=None, prefix="",
		infoprefix="", distributed=False, npass=3, niter=10, errlim=1e-30, my_box=None, write=True):
	"""Given a set of scans and a geometry, this function performs a multipass mapmaking
	where it alternates between solving for the sky and solving for the gains.
	Constant downsampling for now - consider supporting variable downsampling later"""
	if comm is None: comm = mpi.COMM_WORLD
	L.info(infoprefix + "Initializing signals")
	# Cuts
	signal_cut  = mapmaking.SignalCut(scans, dtype=dtype, comm=comm)
	# Map
	if distributed:
		# hack: my_box has descending ra, but geometry assumes ascending ra
		if my_box is not None: my_box = flip_ra(my_box)
		geo     = dmap.geometry(shape, wcs, comm=comm, dtype=dtype, bbox=my_box)
		area    = dmap.zeros(geo)
		subinds = np.zeros(len(scans),int)
		signal_sky  = mapmaking.SignalDmap(scans, subinds, area, sys=sys, name="sky")
	else:
		area        = enmap.zeros(shape, wcs, dtype)
		signal_sky  = mapmaking.SignalMap(scans, area, comm=comm, sys=sys, name="sky")
	# We need a signal for the gains too. How would this work?
	# We're solving the equation d = Sg + n, where S is the best-fit model.
	# Using CG this means that we repeatedly need to be able to apply S'N"S.
	# Two approaches for doing this:
	# 1. Precompute S and use it directoy. This makes signal-gain fast but needs us
	#    to store the full tods in memory.
	# 2. Evaluate the model for each scan for each cg step. Here we would use
	#    the other signals as inputs. This would look like:
	#    for each scan:
	#      model = zeros(scan.ndet, scan.nsamp)
	#      signals.forward(scan, tod, x)
	#      tod   = model * gains for scan
	#      scan.noise.apply(tod)
	#      res   = np.sum(model*tod,1)
	#    This has the cost of a forward for each of the other signals, a noise model,
	#    plus some simple numpy operations that could be optimized if necessary.
	# In practice only #2 scales reasonably, and since it will hopefully converge quickly,
	# the speed hit should be fine. So let's go with #2
	signals     = mapmaking.Signals([signal_cut, signal_sky], comm=comm)
	signal_gain = mapmaking.SignalGain(scans, signals, dtype=dtype, comm=comm)
	# Also set up any arbitrary weights used in the mapmaking
	window      = mapmaking.FilterWindow(config.get("tod_window"))
	weights     = [window]
	# Ok, we can finally solve! Our initial guess for the gain correction is 1,
	# meaning that the fiducial model is right
	gain = np.full(signal_gain.dof.n, 1.0, dtype)
	x0_sky, x0_gain = None, None
	for ipass in range(npass):
		ppre = prefix + "pass%d_" % (ipass+1)
		L.info(infoprefix + "Solving for sky given gains")
		sky  = solve_sky_given_gain(scans, signals, signal_gain, gain, x0=x0_sky,
				weights=weights, dtype=dtype, comm=comm, prefix=ppre, infoprefix=infoprefix + "sky ",
				niter=niter, errlim=errlim)
		if write:
			signals.write(ppre, "map", sky)
		# Make sure next iteration starts from where we left off
		x0_sky = sky
		if not args.nogfit:
			L.info(infoprefix + "Solving for gains given sky")
			x = solve_gain_given_sky(scans, signal_cut, signal_gain, sky, x0=x0_gain,
					weights=weights, dtype=dtype, comm=comm, prefix=ppre, infoprefix=infoprefix + "gain ",
					niter=niter, errlim=errlim)
			gain = x[1]
			gain = normalize_gain(gain, comm)
			if write:
				signal_gain.write(ppre, "map", gain)
			# Make sure next iteration starts from where we left off
			x0_gain= x
	# Return a bunch with enough information to unzip the solution
	return bunch.Bunch(signals=signals, signal_gain=signal_gain, sky=sky, gain=gain)

def normalize_gain(gain, comm):
	norm = comm.allreduce(np.sum(gain))/comm.allreduce(len(gain))
	return gain/norm

def eval_gains(scan, signal_gain, gain):
	# This is too intrusive. Add special method for signal_gain?
	i1,i2 = signal_gain.data[scan]
	return gain[i1:i2]

def apply_gains(scans, signal_gain, gain):
	for scan in scans:
		scan.comps_backup = scan.comps.copy()
		scan.comps *= eval_gains(scan, signal_gain, gain)[:,None]

def restore_gains(scans):
	for scan in scans:
		scan.comps = scan.comps_backup
		del scan.comps_backup

class FilterSubtractModel:
	def __init__(self, signals, x):
		self.signals = signals
		self.x       = x
	def __call__(self, scan, tod):
		wtod = np.zeros_like(tod)
		self.signals.forward(scan, wtod, self.x)
		tod -= wtod

def solve_sky_given_gain(scans, signals, signal_gain, gain, x0=None,
		weights=[], dtype=np.float32, comm=None, prefix="", infoprefix="",
		niter=10, errlim=1e-30, dump=0):
	"""Given a guess at the gain values, solve for the best-fit sky model"""
	# Apply our gain correction. Would be nice to do this by making a shallow copy of
	# scans, but this breaks us using the scans as keys in the Signal classes. In hindsight
	# that was a bad idea. So we will modify the scans in-place instead, and restore them
	# afterwards
	try:
		apply_gains(scans, signal_gain, gain)
		# If we have x0, use it to set up a noise building filter
		filters_noisebuild = []
		if x0 is not None:
			filters_noisebuild.append(FilterSubtractModel(signals, x0))
		# Build our mapmaking equation, which will take care of the solving.
		# This doesn't need to know about the gains.
		return solve_signals(scans, signals, x0=x0, weights=weights, dtype=dtype, comm=comm,
			filters_noisebuild=filters_noisebuild, prefix=prefix, infoprefix=infoprefix,
			niter=niter, errlim=errlim, dump=dump)
	finally:
		restore_gains(scans)

def solve_gain_given_sky(scans, signal_cut, signal_gain, sky, x0=None,
		weights=[], dtype=np.float32, comm=None, prefix="", infoprefix="",
		niter=10, errlim=1e-30, dump=0):
	"""Given a sky solution, solve for the best fit per-detector per-tod gains"""
	signal_gain.model = sky
	signals = mapmaking.Signals([signal_cut, signal_gain], comm=comm)
	# Set up a noise building filter
	filters_noisebuild = [FilterSubtractModel(signal_gain.signals, sky)]
	# Returns cuts, gains. Remember to extract the gain part in the end
	return solve_signals(scans, signals, x0=x0, weights=weights, dtype=dtype, comm=comm,
			filters_noisebuild=filters_noisebuild, prefix=prefix, infoprefix=infoprefix,
			niter=niter, errlim=errlim, dump=dump)

def solve_signals(scans, signals, x0=None, weights=[], filters_noisebuild=[],
		dtype=np.float32, comm=None, prefix="", infoprefix="", niter=10, errlim=1e-30, dump=0):
	"""Jointly solve the given signals"""
	eqsys = mapmaking.Eqsys2(scans, signals, weights=weights,
			filters_noisebuild=filters_noisebuild, dtype=dtype, comm=comm)
	L.info(infoprefix + "Building RHS")
	eqsys.calc_b()
	# We can now build our preconditioners. Signals are now abstract, so I had do update
	# the preconditioners with a build() function
	L.info(infoprefix + "Building preconditioner")
	signals.precon.build(scans)
	L.info(infoprefix + "Solving")
	if x0 is not None: x0 = signals.dof.zip(x0)
	solver = utils.CG(eqsys.A, eqsys.b, M=eqsys.M, dot=eqsys.dot, x0=x0)
	while solver.i < niter and solver.err > errlim:
		t1 = time.time()
		solver.step()
		t2 = time.time()
		L.info(infoprefix + "CG step %5d %15.7e %6.1f %6.3f" % (solver.i, solver.err, (t2-t1), (t2-t1)/len(scans)))
		if dump > 0 and solver.i % dump:
			eqsys.write(prefix, "map%04d" % solver.i, solver.x)
	return signals.dof.unzip(solver.x)

def write_result(prefix, result):
		result.signals.write(prefix, "map", result.sky)
		result.signal_gain.write(prefix, "map", result.gain)

def abort(msg, show=True):
	if show: print(msg)
	sys.exit(1)

shape, wcs = enmap.read_map_geometry(args.template)
shape = (ncomp,)+shape[-2:]
niter = config.get("map_cg_nmax")
distributed = args.distributed>0
prefix = args.odir + "/"
if args.prefix: prefix = prefix + args.prefix + "_"

filedb.init()
ids = filedb.scans[args.sel]

# Read in our scans (minus the actual samples)
my_inds = np.arange(len(ids))[comm.rank::comm.size]
my_inds, scans = scanutils.read_scans(ids, my_inds, actscan.ACTScan, db=filedb.data, downsample=down, dets=args.dets)
nread = comm.allreduce(len(my_inds))
if nread == 0: abort("No usable tods found!", comm.rank==0)
L.info("Found %d tods" % nread)
# Remove any scans that didn't get anything to read from the communicator
active = len(my_inds) > 0
comm_active = comm.Split(active, comm.rank)
if active:
	# Actually do the heavy work of building the map
	result = solve_joint_gain_sky(scans, shape, wcs, dtype=dtype, comm=comm_active,
			sys=mapsys, npass=5, niter=niter, write=True, prefix=prefix)
	# And write it out
	write_result(prefix, result)
comm.Barrier()
if comm.rank == 0:
	print("Done")
