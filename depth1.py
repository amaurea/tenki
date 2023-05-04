import numpy as np, time, os
from pixell import utils, enmap, mpi, bunch
from enact import filedb, files, actscan, actdata
from enlib import config, scanutils, log, coordinates, mapmaking, sampcut, cg, dmap, errors

config.default("dmap_format", "merged")
config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("downsample", 1, "Factor with which to downsample the TOD")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
config.default("tod_window", 5.0, "Number of seconds to window the tod by on each end")
config.default("eig_limit", 0.1, "Pixel condition number below which polarization is dropped to make total intensity more stable. Should be a high value for single-tod maps to avoid thin stripes with really high noise")
config.default("map_sys", "cel", "Map coordinate system")
# Use nearest neigbbor mapmaking, regardless of what's set in .enkirc. Can still be
# overridden with command line arguments
config.set("pmat_map_order", 0)

parser = config.ArgumentParser()
parser.add_argument("sel")
parser.add_argument("template")
parser.add_argument("odir")
parser.add_argument(      "--margin",          type=float, default=0.5)
parser.add_argument("-g", "--tasks-per-group", type=int,   default=1)
parser.add_argument("-c", "--cont",            action="store_true")
parser.add_argument(      "--niter",           type=int,   default=100)
parser.add_argument(      "--dets",            type=str,   default=None)
parser.add_argument("-M", "--meta-only",       action="store_true")
parser.add_argument("-D", "--distributed",     type=int,   default=1)
args = parser.parse_args()

# We default to niter = 100 due to the the cuts causing strong local artifacts
# when the gapfilling doesn't work well enough. These go away during the solution
# process, but this takes around 100 cg steps, sadly.
# I've experimented with different gapfilling, but it didn't really help.

# Each depth-1 map will have a well-defined scanning direction. If this is available
# to the user, then it owuld be much easier for them to describe the noise.
# Should measure the scanning profile as (az_min:az_max,el,t) -> cel, and then
# store it as (delta_ra(el))

comm       = mpi.COMM_WORLD
comm_intra = comm.Split(comm.rank // args.tasks_per_group, comm.rank  % args.tasks_per_group)
comm_inter = comm.Split(comm.rank  % args.tasks_per_group, comm.rank // args.tasks_per_group)
detslice   = args.dets

ncomp     = 3
niter     = args.niter
down      = config.get("downsample")
sys       = config.get("map_sys")
dtype     = {32:np.float32, 64:np.float64}[config.get("map_bits")]
log_level = log.verbosity2level(config.get("verbosity"))
# Special log format to show comm_inter and comm_intra
fmt = "%(rank)3d " + "%3d %3d" % (comm_inter.rank, comm.rank) + " %(wmins)7.2f %(mem)5.2f %(memmax)5.2f %(message)s"
L   = log.init(level=log_level, rank=comm_intra.rank, shared=True, fmt=fmt)

def print_periods(periods, db):
	pids = np.searchsorted(periods[:,0], db.data["t"])-1
	good = np.where((db.data["t"] >= periods[pids,0]) & (db.data["t"] < periods[pids,1]))[0]
	db   = db.select(good)
	pids = pids[good]
	upids, order, edges = utils.find_equal_groups_fast(pids)
	for gi, pid in enumerate(upids):
		print("Group %5d/%d %.0f %.0f dur %8.3f h" % (gi+1, len(upids), periods[pid,0], periods[pid,1], (periods[pid,1]-periods[pid,0])/3600))
		subdb = db.select(order[edges[gi]:edges[gi+1]])
		print(repr(subdb))

def split_periods(periods, maxdur):
	# How long is each period
	durs   = periods[:,1]-periods[:,0]
	# How many parts to split each into
	nsplit = utils.ceil(durs/maxdur)
	nout   = np.sum(nsplit)
	# For each output period, find which input period it
	# corresponds to
	group  = np.repeat(np.arange(len(durs)), nsplit)
	sub    = np.arange(nout)-np.repeat(utils.cumsum(nsplit),nsplit)
	t1     = periods[group,0] + sub*maxdur
	t2     = np.minimum(periods[group,0]+(sub+1)*maxdur, periods[group,1])
	return np.array([t1,t2]).T

def names2uinds(names, return_unique=False):
	"""Given a set of names, return an array where each input element has been replaced by
	a unique integer for each unique name, counting from zero"""
	order = np.argsort(names)
	uinds = np.zeros(len(names),int)
	uvals, inverse = np.unique(names[order], return_inverse=True)
	uinds[order] = inverse
	return uinds if not return_unique else (uinds, uvals)

def make_contour(box, n=100):
	(x1,y1),(x2,y2) = box
	xs = np.linspace(x1, x2, n)
	ys = np.linspace(y1, y2, n)
	return np.concatenate([
		[xs,ys*0+ys[0]],            # bottom left to bottom right
		[xs*0+xs[-1],ys],           # bottom right to top right
		[xs[::-1],ys*0+ys[-1]],     # top right to top left
		[xs*0+xs[0],ys[::-1]]],-1)  # top left to bottom left

def bounds_helper(t1, t2, az1, az2, el, sys, acenter, site):
	ipoints = make_contour([[t1,az1],[t2,az2]]) # [{t,az},:]
	zero    = ipoints[0]*0
	opoints = coordinates.transform("bore", sys, zero+acenter[:,None], time=utils.ctime2mjd(ipoints[0]), bore=[ipoints[1],zero+el,zero,zero], site=site)[::-1]
	opoints[1] = utils.unwind(opoints[1])
	return opoints # [{dec,ra},:]

def find_bounding_box(scandb, entrydb, sys="cel"):
	# all tods in group have same site. We also assume the same detector layout.
	# We need to loop in case some tods are missing pointing, though
	detpos = None
	for id in scandb.ids:
		entry    = entrydb[scandb.ids[0]]
		site     = files.read_site(entry.site)
		try: detpos = actdata.read_point_offsets(entry).point_offset
		except errors.DataMissing: continue
		break
	if detpos is None: raise errors.DataMissing("No pointing found")
	# Array center and radius in focalplane coordinates
	acenter  = np.mean(detpos,0)
	arad     = np.max(np.sum((detpos-acenter)**2,1)**0.5)
	# Find the center point of this group. We do this by transforming the array center to
	# celestial coordinates at the mid-point time of the group.
	t1 = np.min(scandb.data["t"]-0.5*scandb.data["dur"])
	t2 = np.max(scandb.data["t"]+0.5*scandb.data["dur"])
	baz, bel, waz, wel = [scandb.data[x][0]*utils.degree for x in ["baz", "bel", "waz", "wel"]]
	# We're ready to compute the bounding box. We will do that by tracing its contour
	# for the min and max el, and then merging them
	opoints1 = bounds_helper(t1, t2, baz-waz/2, baz+waz/2, bel-wel/2, sys, acenter, site)
	opoints2 = bounds_helper(t1, t2, baz-waz/2, baz+waz/2, bel+wel/2, sys, acenter, site)
	# Merge. We can unwind safely because both contours start in the same corner
	opoints  = np.concatenate([opoints1,opoints2],-1)
	opoints[1] = utils.unwind(opoints[1])
	box     = utils.bounding_box(opoints.T)
	box     = utils.widen_box(box, arad*2, relative=False) # x2 = both sides, not just total width
	box[:,1]= box[::-1,1] # descending ra
	return box

def find_scan_profile(scandb, entrydb, sys="cel",npoint=100):
	# This is a bit redundant with find_bounding_box...
	# all tods in group have same site. We also assume the same detector layout
	entry    = entrydb[scandb.ids[0]]
	site     = files.read_site(entry.site)
	detpos   = actdata.read_point_offsets(entry).point_offset
	# Array center and radius in focalplane coordinates
	acenter  = np.mean(detpos,0)
	arad     = np.max(np.sum((detpos-acenter)**2,1)**0.5)
	# Find the center point of this group. We do this by transforming the array center to
	# celestial coordinates at the mid-point time of the group.
	t = scandb.data["t"][0]
	baz, bel, waz = [scandb.data[x][0]*utils.degree for x in ["baz", "bel", "waz"]]
	# This az range won't necessarily cover all decs in the map. In fact, some decs
	# might not even be reaachable by the boresight. I'll leave those complications to the
	# user. Just continuing the slope from the end points should be good enough.
	iaz  = np.linspace(baz-waz/2,baz+waz/2,npoint)
	zero = iaz*0
	opoints = coordinates.transform("bore", sys, zero+acenter[:,None], time=utils.ctime2mjd(t), bore=[iaz,zero+bel,zero,zero], site=site)[::-1] # dec,ra
	return opoints

def build_time_rhs(scans, signal_sky, signal_cut, window, tref=0):
	# We also a time-hit map
	trhs  = signal_sky.zeros()
	twork = signal_sky.prepare(signal_sky.zeros())
	for si, scan in enumerate(scans):
		signal_sky.precompute(scan)
		# Fill a dummy tod with the time in seconds since the beginning of first tod
		t0  = utils.mjd2ctime(scan.mjd0)
		tod = np.zeros((scan.ndet, scan.nsamp), signal_sky.dtype)
		tod[:] = scan.boresight[:,0] + (t0-tref)
		# Apply white noise model
		window(scan, tod)
		scan.noise.white(tod)
		window(scan, tod)
		# Remove cut samples
		sampcut.gapfill_const(scan.cut, tod, inplace=True)
		# And accumulate
		signal_sky.backward(scan, tod, twork)
	signal_sky.finish(trhs, twork)
	del twork
	return trhs[0]

def flip_ra(box):
	res = box.copy()
	res[...,:,1] = res[...,::-1,1]
	return res

def build_maps(scans, shape, wcs, tref=0, dtype=np.float32, sys="cel", comm=None, tag=None,
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
	# Build time-map. The transform stuff is just to avoid division by zero.
	# Sadly can't just use np.maximum directly on the map, as it might not be an enmap
	L.info(pre + "Building time map")
	trhs = build_time_rhs(scans, signal_sky, signal_cut, window, tref=tref)
	tmap = trhs / signal_sky.transform(signal_sky.precon.div[0,0], lambda x: np.maximum(x, 1e-40))
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

def write_maps(prefix, data):
	data.signal.write(prefix, "map",  data.map)
	data.signal.write(prefix, "ivar", data.ivar)
	data.signal.write(prefix, "time", data.tmap)

def write_info(oname, info):
	utils.mkdir(os.path.dirname(oname))
	bunch.write(oname, info)

shape_full, wcs_full = enmap.read_map_geometry(args.template)
shape_full = (ncomp,)+shape_full[-2:]
distributed = args.distributed>0

filedb.init()
# Reject tods with missing metadata
good = np.nonzero(np.isfinite(filedb.scans.data["t"])&np.isfinite(filedb.scans.data["baz"])&np.isfinite(filedb.scans.data["bel"]))[0]
db_full = filedb.scans.select(good)
# Find ctime regions with continuous scanning, but limit them to one day
periods = scanutils.find_scan_periods(db_full, ttol=12*3600)
periods = split_periods(periods, 24*3600)
#print_periods(periods, db_full)
widen = 2*args.margin*utils.degree

# For all our ids, figure out which period they belong to
db   = db_full.select(args.sel)
pids = np.searchsorted(periods[:,0], db.data["t"])-1
# Create sub-groups for each array, since we will map them separately
atag        = np.char.replace(np.char.replace(np.char.rpartition(db.ids,".")[:,2],"ar","pa"),":","_")
aid, arrays = names2uinds(atag, return_unique=True)
narr        = len(arrays)
apids       = pids*narr + aid
# And loop over these
gvals, order, edges = utils.find_equal_groups_fast(apids)
# Loop over each such group. We will map each group
for gi in range(comm_inter.rank, len(gvals), comm_inter.size):
	apid     = gvals[gi]
	pid, aid = np.divmod(apid, narr)
	inds     = order[edges[gi]:edges[gi+1]]
	ntod     = len(inds)
	# Make a tag for this group we can use when printing progress
	tag      = "%4d/%d" % (gi+1, len(gvals))
	# Build our output prefix. Will use sub-directories to make things like ls faster
	t        = utils.floor(periods[pid,0])
	t5       = ("%05d" % t)[:5]
	prefix   = "%s/%s/depth1_%010d_%s" % (args.odir, t5, t, arrays[aid])
	meta_done = os.path.isfile(prefix + "_info.hdf")
	maps_done = os.path.isfile(prefix + ".empty") or (
		os.path.isfile(prefix + "_time.fits") and
		os.path.isfile(prefix + "_map.fits") and
		os.path.isfile(prefix + "_ivar.fits"))
	if args.cont and meta_done and (maps_done or args.meta_only): continue
	L.info("Processing %4d/%d period %4d arr %s @%.0f dur %4.2f h with %2d tods" % (gi+1, len(gvals), pid, arrays[aid], t, (periods[pid,1]-periods[pid,0])/3600, len(inds)))
	# Find the bounding box for this group, as well as this task's part of it
	try: box = find_bounding_box(db.select(inds), filedb.data, sys=sys)
	except errors.DataMissing:
		if comm_intra.rank == 0:
			L.debug("Skipping %4d/%d: No readable tods" % (gi+1, len(gvals)))
			utils.mkdir(os.path.dirname(prefix))
			with open(prefix + ".empty", "w") as ofile:
				ofile.write("\n")
		continue
	box = utils.widen_box(box, widen, relative=False)
	# Build our geometry
	shape, wcs = enmap.Geometry(shape_full, wcs_full).submap(box=box)
	# Find the scanning profile. This is useful for understanding the
	# noise properties of the maps
	profile = find_scan_profile(db.select(inds), filedb.data, sys=sys)
	# Write out our metadata
	if comm_intra.rank == 0:
		info = bunch.Bunch(profile=profile, pid=pid, period=periods[pid],
				ids=np.char.encode(db.select(inds).ids), box=box, array=arrays[aid].encode(), t=t)
		write_info(prefix + "_info.hdf", info)
	if args.cont and (args.meta_only or maps_done): continue
	# Decide which scans we should own. We do them consecutively to give each
	# task a compact area
	i1 = comm_intra.rank*ntod//comm_intra.size
	i2 = min(ntod,(comm_intra.rank+1)*ntod//comm_intra.size)
	my_inds = inds[i1:i2]
	# Read in our scans (minus the actual samples)
	my_inds, scans = scanutils.read_scans(db.ids, my_inds, actscan.ACTScan, db=filedb.data, downsample=down)
	if detslice:
		scans = [eval("scan["+detslice+"]") for scan in scans]
	nread = comm_intra.allreduce(len(my_inds))
	if nread == 0:
		# No valid tods! Make a placeholder file so we can skip past this when resuming
		if comm_intra.rank == 0:
			L.debug("Skipping %4d/%d: No readable tods" % (gi+1, len(gvals)))
			utils.mkdir(os.path.dirname(prefix))
			with open(prefix + ".empty", "w") as ofile:
				ofile.write("\n")
	# Remove any scans that didn't get anything to read from the communicator
	comm_active = comm_intra.Split(len(my_inds)>0, comm_intra.rank)
	if len(my_inds) > 0:
		# Ok, we actually have something to do.
		# Figure out what our local bounding box is
		my_box = find_bounding_box(db.select(my_inds), filedb.data, sys=sys)
		my_box = utils.widen_box(my_box, widen, relative=False)
		# Actually do the heavy work of building the map
		maps = build_maps(scans, shape, wcs, tref=t, dtype=dtype, comm=comm_active, tag=tag, sys=sys, distributed=distributed, my_box=my_box, niter=niter)
		# And write it out
		write_maps(prefix, maps)
	comm_intra.Barrier()
comm.Barrier()
if comm.rank == 0:
	print("Done")
