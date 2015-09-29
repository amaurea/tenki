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

# Default signal parameters
config.default("signal_sky_default",   "use=no,type=map,name=sky,sys=cel,prec=bin", "Default parameters for sky map")
config.default("signal_hor_default",   "use=no,type=map,name=hor,sys=hor,prec=bin", "Default parameters for ground map")
config.default("signal_sun_default",   "use=no,type=map,name=sun,sys=hor:Sun,prec=bin", "Default parameters for sun map")
config.default("signal_moon_default",  "use=no,type=map,name=moon,sys=hor:Sun,prec=bin", "Default parameters for moon map")
config.default("signal_cut_default",   "use=no,type=cut,name=cut,ofmt='{name}_{rank:03}',output=no,use=yes", "Default parameters for cut (junk) signal")
config.default("signal_scan_default",  "use=no,type=scan,name=scan,ofmt='{name}_{pid:02}_{az0:.0f}_{az1:.0f}_{el:.0f}',2way=yes,res=2,tol=0.5", "Default parameters for scan/pickup signal")
# Default filter parameters
config.default("filter_scan_default",  "use=no,name=scan,value=0,naz=8,nt=10,sky=yes", "Default parameters for scan/pickup filter")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("filelist")
parser.add_argument("odir")
parser.add_argument("prefix",nargs="?")
parser.add_argument("-d", "--dump", type=str, default="1,2,5,10,20,50,100,200,300,400,500,600,800,1000,1200,1500,2000,3000,4000,5000,6000,8000,10000", help="CG map dump steps")
parser.add_argument("--ncomp",      type=int, default=3,  help="Number of stokes parameters")
parser.add_argument("--ndet",       type=int, default=0,  help="Max number of detectors")
parser.add_argument("--dump-config", action="store_true", help="Dump the configuration file to standard output.")
parser.add_argument("-S", "--signal", action="append",    help="Signals to solve for. For example -S sky:area.fits -S scan would solve for the sky map and scan pickup maps jointly, using area.fits as the map template.")
parser.add_argument("-F", "--filter", action="append")
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
pickup_res = 2*utils.arcmin
#print "FIXME A"
#nrow,ncol=1,1
#pickup_res = 2*utils.arcmin * 100

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

def parse_desc(desc, default={}):
	res = default.copy()
	# Parse normally now that the : are out of the way
	for tok in desc.split(","):
		subtoks = tok.split("=")
		if len(subtoks) == 1:
			res["value"] = subtoks[0]
		else:
			key, val = subtoks
			res[key] = val
	return res
def setup_params(prefix, predefined, defparams):
	params = {}
	for name in predefined:
		params[name] = parse_desc(config.get("%s_%s_default" % (prefix, name)))
	argdict = vars(args)
	overrides = argdict[prefix]
	if overrides:
		for oval in overrides:
			if ":" in oval:
				name, rest = oval.split(":")
				desc = config.get("%s_%s_default" % (prefix,name)) + ",use=yes," + rest
			elif "," not in oval:
				desc = config.get("%s_%s_default" % (prefix,oval)) + ",use=yes"
			else:
				desc = "use=yes,"+oval
			param = parse_desc(desc, default=defparams)
			params[param["name"]] = param
	# Flatten to listand kill irrelevant ones
	params = [params[k] for k in params if params[k]["use"] != "no"]
	return params

######## Signal parmeters ########
signal_params = setup_params("signal", ["cut","sky","hor","sun","moon","scan"], {"use":"no", "ofmt":"{name}", "output":"yes"})
# Put the 'cut' signal first because other signals may depend on it
i = 0
for s in signal_params:
	if s["type"] == "cut": break
	i += 1
assert i < len(signal_params)
signal_params = signal_params[i:i+1] + signal_params[:i] + signal_params[i+1:]
# We only support a single distributed map
dsys = None
for sig in signal_params:
	if sig["type"] == "dmap":
		if dsys: raise ValueError("Only a single map may be distributed")
		else: dsys=sig["sys"]
	if sig["type"] in ["map", "dmap"]:
		assert "value" in sig and sig["value"] is not None and os.path.isfile(sig["value"]), "Map-type signals need a template map as argument. E.g. -S sky:foo.fits"

######## Filter parmeters ########
filter_params = setup_params("filter", ["scan"], {"use":"no"})

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

# Try to get about the same amount of data for each mpi task.
# If we use distributed maps, we also try to make things as local as possible
mycosts = [s.nsamp*s.ndet for s in myscans]
if dsys: # distributed maps
	myboxes = [scanutils.calc_sky_bbox_scan(s, dsys) for s in myscans] if dsys else None
	myinds, mysubs, mybbox = scanutils.distribute_scans(myinds, mycosts, myboxes, comm)
else:
	myinds = scanutils.distribute_scans(myinds, mycosts, None, comm)

# And reread the correct files this time. Ideally we would
# transfer this with an mpi all-to-all, but then we would
# need to serialize and unserialize lots of data, which
# would require lots of code.
L.info("Rereading shuffled scans")
myscans, myinds = read_scans(filelist, myinds, db, ndet=args.ndet)

# I would like to be able to do on-the-fly nmat computation.
# However, preconditioners depend on the noise matrix.
# Hence, to be able to do this, Eqsys initialization must go like this:#
# 1. Set up plain signals (no priors or postprocessing)
# 2. Set up plain filters (no postprocessing)
# 3. Initialize Eqsys
# 4. Call eqsys.calc_b, which updates the noise matrices
# 5. Loop through signals again, this time setting up preconditioenrs and
#    priors, which can be inserted into the signals inside Eqsys
# 6. Loop through filters again, setting up any associated signal postprocessing
# It is a good deal less tidy than what I currently have. And it will break
# as soon as a filter or signal that depends on the noise matrix appears.

# 1. Initialize filters
L.info("Initializing signals")
signals = []
for param in signal_params:
	if param["type"] == "cut":
		signal = mapmaking.SignalCut(myscans, dtype=dtype, comm=comm, name=param["name"], ofmt=param["ofmt"], output=param["output"]=="yes")
		signal_cut = signal
	elif param["type"] == "map":
		area = enmap.read_map(param["value"])
		area = enmap.zeros((args.ncomp,)+area.shape[-2:], area.wcs, dtype)
		signal = mapmaking.SignalMap(myscans, area, comm=comm, name=param["name"], ofmt=param["ofmt"], output=param["output"]=="yes")
	elif param["type"] == "dmap":
		area = dmap.read_map(param["value"], bbox=mybbox, tshape=tshape, comm=comm)
		area = dmap.zeros(area.geometry.aspre(args.ncomp).astype(dtype))
		signal = mapmaking.SignalDmap(myscans, mysubs, area, name=param["name"], ofmt=param["ofmt"], output=param["output"]=="yes")
	elif param["type"] == "scan":
		res = float(param["res"])/utils.arcmin
		tol = float(param["tol"])/utils.degree
		patterns, mypids = scanutils.classify_scanning_patterns(myscans, comm=comm, tol=tol)
		L.info("Found %d scanning patterns" % len(patterns))
		signal = mapmaking.SignalPhase(myscans, mypids, patterns, myscans[0].dgrid, res=res, dtype=dtype, comm=comm, name=param["name"], ofmt=param["ofmt"], output=param["output"]=="yes")
	else:
		raise ValueError("Unrecognized signal type '%s'" % param["type"])
	signals.append(signal)

# 2. Initialize filters
L.info("Initializing filters")
filters = []
for param in filter_params:
	if param["name"] == "scan":
		naz, nt, mode = int(param["naz"]), int(param["nt"]), int(param["value"])
		if mode == 0: continue
		filter = mapmaking.FilterPickup(naz=naz, nt=nt)
	else:
		raise ValueError("Unrecognized fitler name '%s'" % param["name"])
	filters.append(filter)

L.info("Initializing equation system")
eqsys = mapmaking.Eqsys(myscans, signals, filters=filters, dtype=dtype, comm=comm)

L.info("Initializing RHS")
eqsys.calc_b()

L.info("Initializing preconditioners")
for param, signal in zip(signal_params, signals):
	if param["type"] == "cut":
		signal.precon = mapmaking.PreconCut(signal, myscans)
	elif param["type"] == "map":
		if param["prec"] == "bin":
			signal.precon = mapmaking.PreconMapBinned(signal, signal_cut, myscans)
		else: raise ValueError("Unknown map preconditioner '%s'" % param["prec"])
		if "nohor" in param and param["nohor"] != "no":
			prior_weight = signal.precon.div[0,0]
			prior_weight /= (np.mean(prior_weight)*prior_weight.shape[-1])**0.5
			prior_weight *= float(param["nohor"])
			signal.prior = mapmaking.PriorMapNohor(prior_weight)
	elif param["type"] == "dmap":
		if param["prec"] == "bin":
			signal.precon = mapmaking.PreconDmapBinned(signal, signal_cut, myscans)
		else: raise ValueError("Unknown dmap preconditioner '%s'" % param["prec"])
		if "nohor" in param and param["nohor"] != "no":
			prior_weight  = signal.precon.div[0,0]
			prior_weight /= (dmap.sum(prior_weight)/prior_weight.size*prior_weight.shape[-1])**0.5
			prior_weight *= float(param["nohor"])
			signal.prior = mapmaking.PriorDmapNohor(prior_weight)
	elif param["type"] == "scan":
		signal.precon = mapmaking.PreconPhaseBinned(signal, signal_cut, myscans)
	else:
		raise ValueError("Unrecognized signal type '%s'" % param["type"])

L.info("Initializing postprocessors")
for param in filter_params:
	if param["name"] == "scan":
		naz, nt, mode = int(param["naz"]), int(param["nt"]), int(param["value"])
		if mode >= 2:
			for sparam, signal in zip(signal_params, signals):
				sname = sparam["name"]
				if sname in param and param[sname] == "yes":
					if sparam["type"] == "map":
						prec_ptp = mapmaking.PreconMapBinned(signal, signal_cut, myscans, noise=False, hits=False)
					elif sparam["type"] == "dmap":
						prec_ptp = mapmaking.PreconDmapBinned(signal, signal_cut, myscans, noise=False, hits=False)
					else:
						raise NotImplementedError("Scan postfiltering for '%s' signals not implemented" % sparam["type"])
					signal.post.append(mapmaking.PostPickup(myscans, signal, signal_cut, prec_ptp, naz=naz, nt=nt))
	else:
		raise ValueError("Unrecognized fitler name '%s'" % param["name"])

mapmaking.write_precons(signals, root)

#L.info("Initializing signals")
#signals = []
#for param in signal_params:
#	if param["type"] == "cut":
#		signal = mapmaking.SignalCut(myscans, dtype=dtype, comm=comm, name=param["name"], ofmt=param["ofmt"], output=param["output"]=="yes")
#		signal.precon = mapmaking.PreconCut(signal, myscans)
#		signal_cut = signal
#	elif param["type"] == "map":
#		area = enmap.read_map(param["value"])
#		area = enmap.zeros((args.ncomp,)+area.shape[-2:], area.wcs, dtype)
#		signal = mapmaking.SignalMap(myscans, area, comm=comm, name=param["name"], ofmt=param["ofmt"], output=param["output"]=="yes")
#		if param["prec"] == "bin":
#			signal.precon = mapmaking.PreconMapBinned(signal, signal_cut, myscans)
#		else: raise ValueError("Unknown map preconditioner '%s'" % param["prec"])
#		if "nohor" in param and param["nohor"] != "no":
#			prior_weight = signal.precon.div[0,0]
#			prior_weight /= (np.mean(prior_weight)*prior_weight.shape[-1])**0.5
#			prior_weight *= float(param["nohor"])
#			signal.prior = mapmaking.PriorMapNohor(prior_weight)
#	elif param["type"] == "dmap":
#		area = dmap.read_map(param["value"], bbox=mybbox, tshape=tshape, comm=comm)
#		area = dmap.zeros(area.geometry.aspre(args.ncomp).astype(dtype))
#		signal = mapmaking.SignalDmap(myscans, mysubs, area, name=param["name"], ofmt=param["ofmt"], output=param["output"]=="yes")
#		if param["prec"] == "bin":
#			signal.precon = mapmaking.PreconDmapBinned(signal, signal_cut, myscans)
#		else: raise ValueError("Unknown dmap preconditioner '%s'" % param["prec"])
#		if "nohor" in param and param["nohor"] != "no":
#			prior_weight  = signal.precon.div[0,0]
#			prior_weight /= (dmap.sum(prior_weight)/prior_weight.size*prior_weight.shape[-1])**0.5
#			prior_weight *= float(param["nohor"])
#			signal.prior = mapmaking.PriorDmapNohor(prior_weight)
#	elif param["type"] == "scan":
#		res = float(param["res"])/utils.arcmin
#		tol = float(param["tol"])/utils.degree
#		patterns, mypids = scanutils.classify_scanning_patterns(myscans, comm=comm, tol=tol)
#		L.info("Found %d scanning patterns" % len(patterns))
#		signal = mapmaking.SignalPhase(myscans, mypids, patterns, myscans[0].dgrid, res=res, dtype=dtype, comm=comm, name=param["name"], ofmt=param["ofmt"], output=param["output"]=="yes")
#		signal.precon = mapmaking.PreconPhaseBinned(signal, signal_cut, myscans)
#	else:
#		raise ValueError("Unrecognized signal type '%s'" % param["type"])
#	signals.append(signal)
#
## Initialize filters
#filters = []
#for param in filter_params:
#	if param["name"] == "scan":
#		naz, nt, mode = int(param["naz"]), int(param["nt"]), int(param["value"])
#		if mode == 0: continue
#		filter = mapmaking.FilterPickup(naz=naz, nt=nt)
#		if mode >= 2:
#			for sparam, signal in zip(signal_params, signals):
#				sname = sparam["name"]
#				if sname in param and param[sname] == "yes":
#					if sparam["type"] == "map":
#						prec_ptp = mapmaking.PreconMapBinned(signal, signal_cut, myscans, noise=False, hits=False)
#					elif sparam["type"] == "dmap":
#						prec_ptp = mapmaking.PreconDmapBinned(signal, signal_cut, myscans, noise=False, hits=False)
#					else:
#						raise NotImplementedError("Scan postfiltering for '%s' signals not implemented" % sparam["type"])
#					signal.post.append(mapmaking.PostPickup(myscans, signal, signal_cut, prec_ptp, naz=naz, nt=nt))
#	else:
#		raise ValueError("Unrecognized fitler name '%s'" % param["name"])
#	filters.append(filter)

#signal_cut = mapmaking.SignalCut(myscans, dtype, comm)
#signal_cut.precon = mapmaking.PreconCut(signal_cut, myscans)
#signals.append(signal_cut)
## Main maps
#if True:
#	if distributed:
#		area = dmap.read_map(args.area, bbox=mybbox, tshape=tshape, comm=comm)
#		area = dmap.zeros(area.geometry.aspre(args.ncomp).astype(dtype))
#		signal_map = mapmaking.SignalDmap(myscans, mysubs, area)
#		signal_map.precon = mapmaking.PreconDmapBinned(signal_map, signal_cut, myscans)
#		if args.nohor:
#			prior_weight  = signal_map.precon.div[0,0]
#			prior_weight /= (dmap.sum(prior_weight)/prior_weight.size*prior_weight.shape[-1])**0.5
#			prior_weight /= 10
#			signal_map.prior = mapmaking.PriorDmapNohor(prior_weight)
#		if args.filter_pickup >= 2:
#			prec_ptp = mapmaking.PreconDmapBinned(signal_map, signal_cut, myscans, noise=False, hits=False)
#			signal_map.post.append(mapmaking.PostPickup(myscans, signal_map, signal_cut, prec_ptp))
#	else:
#		area = enmap.read_map(args.area)
#		area = enmap.zeros((args.ncomp,)+area.shape[-2:], area.wcs, dtype)
#		signal_map = mapmaking.SignalMap(myscans, area, comm)
#		signal_map.precon = mapmaking.PreconMapBinned(signal_map, signal_cut, myscans)
#		if args.nohor:
#			prior_weight = signal_map.precon.div[0,0]
#			prior_weight /= (np.mean(prior_weight)*prior_weight.shape[-1])**0.5
#			prior_weight /= 10
#			signal_map.prior = mapmaking.PriorMapNohor(prior_weight)
#		if args.filter_pickup >= 2:
#			prec_ptp = mapmaking.PreconMapBinned(signal_map, signal_cut, myscans, noise=False, hits=False)
#			signal_map.post.append(mapmaking.PostPickup(myscans, signal_map, signal_cut, prec_ptp))
#	signals.append(signal_map)
## Pickup maps
#if args.pickup_maps:
#	# Classify scanning patterns
#	patterns, mypids = scanutils.classify_scanning_patterns(myscans, comm=comm)
#	L.info("Found %d scanning patterns" % len(patterns))
#	signal_pickup = mapmaking.SignalPhase(myscans, mypids, patterns, (nrow,ncol), pickup_res, dtype=dtype, comm=comm)
#	signal_pickup.precon = mapmaking.PreconPhaseBinned(signal_pickup, signal_cut, myscans)
#	signals.append(signal_pickup)
#if args.filter_pickup >= 1:
#	filters.append(mapmaking.FilterPickup())

#mapmaking.write_precons(signals, root)
#L.info("Initializing equation system")
#eqsys = mapmaking.Eqsys(myscans, signals, filters=filters, dtype=dtype, comm=comm)

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
		if cg.i in dump_steps or cg.i % dump_steps[-1] == 0:
			x = eqsys.postprocess(cg.x)
			eqsys.write(root, "map%04d" % cg.i, x)
		bench.stats.write(benchfile)
		L.info("CG step %5d %15.7e %6.1f %6.3f" % (cg.i, cg.err, dt, dt/max(1,len(eqsys.scans))))
