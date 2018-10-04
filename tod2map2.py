import numpy as np, time, copy, argparse, os, sys, pipes, shutil, re
from enlib import utils
with utils.nowarn(): import h5py
from enlib import enmap, pmat, fft, config, array_ops, mapmaking, nmat, errors, mpi
from enlib import log, bench, dmap, coordinates, scan as enscan, scanutils
from enlib import pointsrcs, bunch, planet9, ephemeris, parallax
from enlib.cg import CG
from enlib.source_model import SourceModel
from enact import actscan, nmat_measure, filedb, todinfo
from enact import actdata

config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("downsample", 1, "Factor with which to downsample the TOD")
config.default("hwp_resample", False, "Whether to resample the TOD to make the HWP equispaced")
config.default("map_cg_nmax", 500, "Max number of CG steps to perform in map-making")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
config.default("task_dist", "size", "How to assign scans to each mpi task. Can be 'plain' for comm.rank:n:comm.size-type assignment, 'size' for equal-total-size assignment. The optimal would be 'time', for equal total time for each, but that's not implemented currently.")
config.default("gfilter_jon", False, "Whether to enable Jon's ground filter.")
config.default("map_ptsrc_handling", "subadd", "How to handle point sources in the map. Can be 'none' for no special treatment, 'subadd' to subtract from the TOD and readd in pixel space, and 'sim' to simulate a pointsource-only TOD.")
config.default("map_ptsrc_sys", "cel", "Coordinate system the point source positions are specified in. Default is 'cel'")
config.default("map_format", "fits", "File format to use when writing maps. Can be 'fits', 'fits.gz' or 'hdf'.")
config.default("resume", 0, "Interval at which to write the internal CG information to allow for restarting. If 0, this will never be written. Also controls whether existing information on disk will be used for restarting if avialable. If negative, restart information will be written, but not used.")

# Default signal parameters
config.default("signal_sky_default",   "use=no,type=map,name=sky,sys=cel,prec=bin", "Default parameters for sky map")
config.default("signal_hor_default",   "use=no,type=map,name=hor,sys=hor,prec=bin", "Default parameters for ground map")
config.default("signal_sun_default",   "use=no,type=map,name=sun,sys=sidelobe:Sun,prec=bin,lim_Sun_min_el=0", "Default parameters for sun map")
config.default("signal_moon_default",  "use=no,type=map,name=moon,sys=sidelobe:Moon,prec=bin,lim_Moon_min_el=0", "Default parameters for moon map")
config.default("signal_jupiter_default",  "use=no,type=map,name=jupiter,sys=sidelobe:Jupiter,prec=bin,lim_Jupiter_min_el=0", "Default parameters for jupiter map")
config.default("signal_saturn_default",  "use=no,type=map,name=saturn,sys=sidelobe:Saturn,prec=bin,lim_Saturn_min_el=0", "Default parameters for saturn map")
config.default("signal_uranus_default",  "use=no,type=map,name=uranus,sys=sidelobe:Uranus,prec=bin,lim_Uranus_min_el=0", "Default parameters for uranus map")
config.default("signal_neptune_default",  "use=no,type=map,name=neptune,sys=sidelobe:Neptune,prec=bin,lim_Neptune_min_el=0", "Default parameters for neptune map")
config.default("signal_pluto_default",  "use=no,type=map,name=pluto,sys=sidelobe:Pluto,prec=bin,lim_Pluto_min_el=0", "Default parameters for pluto map")
config.default("signal_cut_default",   "use=no,type=cut,name=cut,ofmt={name}_{rank:03},output=no,use=yes", "Default parameters for cut (junk) signal")
config.default("signal_scan_default",  "use=no,type=scan,name=scan,2way=yes,res=1,tol=0.5", "Default parameters for scan/pickup signal")
#config.default("signal_scan_default",  "use=no,type=scan,name=scan,ofmt={name}_{pid:02}_{az0:.0f}_{az1:.0f}_{el:.0f},2way=yes,res=2,tol=0.5", "Default parameters for scan/pickup signal")
# Default filter parameters
config.default("filter_scan_default",  "use=no,name=scan,value=2,daz=3,nt=10,nhwp=0,weighted=0,niter=3,sky=yes", "Default parameters for scan/pickup filter")
config.default("filter_add_default",  "use=no,name=add,value=1,sys=cel,type=map,mul=+1,tmul=1,sky=yes,nopol=0", "Default parameters for map subtraction filter")
config.default("filter_sub_default",  "use=no,name=add,value=1,sys=cel,type=map,mul=-1,tmul=1,sky=yes,nopol=0", "Default parameters for map subtraction filter")
config.default("filter_src_default",   "use=no,name=src,value=1,snr=5,sys=cel,mul=1,sky=yes", "Default parameters for point source subtraction filter")
config.default("filter_buddy_default",   "use=no,name=buddy,value=1,mul=1,type=auto,sys=cel,tmul=1,sky=yes,pertod=0,nstep=200,prec=bin", "Default parameters for map subtraction filter")
config.default("filter_hwp_default",   "use=no,name=hwp,value=1", "Default parameters for hwp notch filter")
config.default("filter_common_default", "use=no,name=common,value=1", "Default parameters for blockwise common mode filter")
config.default("filter_addphase_default",  "use=no,name=addphase,value=1,mul=+1,tmul=1,sky=yes,tol=0.5", "Default parameters for phasemap subtraction filter")
config.default("filter_subphase_default",  "use=no,name=addphase,value=1,mul=-1,tmul=1,sky=yes,tol=0.5", "Default parameters for phasemap subtraction filter")
config.default("filter_fitphase_default",  "use=no,name=fitphase,value=1,mul=+1,tmul=1,sky=yes,tol=0.5,perdet=1", "Default parameters for phasemap subtraction filter")
config.default("filter_scale_default",     "use=no,name=scale,value=1,sky=yes", "Default parameters for filter that simply scale the TOD by the given value")

# Default map filter parameters
config.default("mapfilter_gauss_default", "use=no,name=gauss,value=0,cap=1e3,type=gauss,sky=yes", "Default parameters for gaussian map filter in mapmaking")

config.default("crossmap", True,  "Whether to output the crosslinking map")
config.default("icovmap",  True, "Whether to output the inverse correlation map")
config.default("icovstep",    6, "Physical degree interval between inverse correlation measurements in icovmap")
config.default("icovyskew",   1, "Number of degrees in the y direction (dec) to shift by per step in x (ra)")

config.default("tod_window", 5.0, "Number of samples to window the tod by on each end")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("filelist")
parser.add_argument("odir")
parser.add_argument("prefix",nargs="?")
parser.add_argument("-d", "--dump", type=str, default="10,50,100,250,500,700,1000,1500,2000,3000,4000,5000,6000,8000,10000", help="CG map dump steps")
parser.add_argument("--ncomp",      type=int, default=3,  help="Number of stokes parameters")
parser.add_argument("--dets",       type=str, default=0,  help="Detector slice")
parser.add_argument("--dump-config", action="store_true", help="Dump the configuration file to standard output.")
parser.add_argument("-S", "--signal",    action="append", help="Signals to solve for. For example -S sky:area.fits -S scan would solve for the sky map and scan pickup maps jointly, using area.fits as the map template.")
parser.add_argument("-F", "--filter",    action="append")
parser.add_argument("-M", "--mapfilter", action="append")
parser.add_argument("--group-tods", action="store_true")
parser.add_argument("--individual", action="store_true")
parser.add_argument("--tod-debug",  action="store_true")
parser.add_argument("--prepost",    action="store_true")
args = parser.parse_args()

if args.dump_config:
	print config.to_str()
	sys.exit(0)

dtype = np.float32 if config.get("map_bits") == 32 else np.float64
comm  = mpi.COMM_WORLD
nmax  = config.get("map_cg_nmax")
ext   = config.get("map_format")
tshape= (720,720)
#tshape= (100,100)
resume= config.get("resume")

filedb.init()
db = filedb.data
filelist = todinfo.get_tods(args.filelist, filedb.scans)
if args.group_tods:
	filelist = scanutils.get_tod_groups(filelist)

utils.mkdir(args.odir)
root = args.odir + "/" + (args.prefix + "_" if args.prefix else "")

# Dump our settings
if comm.rank == 0:
	config.save(root + "config.txt")
	with open(root + "args.txt","w") as f:
		argstring = " ".join([pipes.quote(a) for a in sys.argv[1:]])
		f.write(argstring + "\n")
		print argstring
	with open(root + "env.txt","w") as f:
		for k,v in os.environ.items():
			f.write("%s: %s\n" %(k,v))
	with open(root + "ids.txt","w") as f:
		for id in filelist:
			f.write("%s\n" % str(id))
	shutil.copyfile(filedb.cjoin(["root","dataset","filedb"]),  root + "filedb.txt")
	try: shutil.copyfile(filedb.cjoin(["root","dataset","todinfo"]), root + "todinfo.hdf")
	except IOError: pass
# Set up logging
utils.mkdir(root + "log")
logfile   = root + "log/log%03d.txt" % comm.rank
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, file=logfile, rank=comm.rank)
# And benchmarking
utils.mkdir(root + "bench")
benchfile = root + "bench/bench%03d.txt" % comm.rank

def parse_desc(desc, default={}):
	res = default.copy()
	# Parse normally now that the : are out of the way
	for tok in utils.split_outside(desc, ",", "[({", "])}"):
		subtoks = tok.split("=")
		if len(subtoks) == 1:
			res["value"] = subtoks[0]
		else:
			key, val = subtoks
			res[key] = val
	return res
def setup_params(category, predefined, defparams):
	"""Set up parameters of a given category. With the following
	convention.
	 1. If a name matches a default's name, unspecified properties use the default's
	 2. A name can be specified multiple times, e.g. -S sky:foo -S sky:bar. These
	    do not override each other - each will be used separately.
	 2. If a name that has a default is not specified manually, then a single
	    instance of that name is instantiated, with the default parameters."""
	params  = []
	argdict = vars(args)
	overrides  = argdict[category]
	counts = {}
	if overrides:
		for oval in overrides:
			m = re.match(r'([^,:]+):(.*)', oval)
			if m:
				name, rest = m.groups()
				desc = config.get("%s_%s_default" % (category,name)) + ",use=yes," + rest
			elif "," not in oval:
				desc = config.get("%s_%s_default" % (category,oval)) + ",use=yes"
			else:
				desc = "use=yes,"+oval
			param = parse_desc(desc, default=defparams)
			name = param["name"]
			if name in counts: counts[name] += 1
			else: counts[name] = 1
			param["i"] = counts[name]
			params.append(param)
	# For each predefined param, add it only if none of that name already exist
	defaults = []
	for p in predefined:
		if not p in counts:
			defaults.append(parse_desc(config.get("%s_%s_default" % (category, p))))
			defaults[-1]["i"] = 0
	params = defaults + params
	# Kill irrelevant parameters (those not in use)
	params = [p for p in params if p["use"] != "no"]
	return params
def get_effname(param): return param["name"] + (str(param["i"]) if param["i"] > 1 else "")

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
	if sig["type"] in ["dmap","fdmap"]:
		if dsys: raise ValueError("Only a single map may be distributed")
		else: dsys=sig["sys"]
	if sig["type"] in ["map", "dmap", "fmap", "fdmap"]:
		assert "value" in sig and sig["value"] is not None, "Map-type signals need a template map as argument. E.g. -S sky:foo.fits"

######## Filter parmeters ########
filter_params    = setup_params("filter", ["scan","sub"], {"use":"no"})
mapfilter_params = setup_params("mapfilter", [], {"use":"no"})

# Read in all our scans
L.info("Reading %d scans" % len(filelist))
myinds = np.arange(len(filelist))[comm.rank::comm.size]
myinds, myscans = scanutils.read_scans(filelist, myinds, actscan.ACTScan,
		db, dets=args.dets, downsample=config.get("downsample"), hwp_resample=config.get("hwp_resample"))
myinds = np.array(myinds, int)

# Collect scan info. This currently fails if any task has empty myinds
read_ids  = [filelist[ind] for ind in utils.allgatherv(myinds, comm)]
read_ntot = len(read_ids)
L.info("Found %d tods" % read_ntot)
if read_ntot == 0:
	L.info("Giving up")
	sys.exit(1)
read_ndets= utils.allgatherv([len(scan.dets) for scan in myscans], comm)
read_nsamp= utils.allgatherv([scan.cut.size-scan.cut.sum() for scan in myscans], comm)
read_dets = utils.uncat(utils.allgatherv(
	np.concatenate([scan.dets for scan in myscans]) if len(myscans) > 0 else np.zeros(0,int)
	,comm), read_ndets)
# Save accept list
if comm.rank == 0:
	with open(root + "accept.txt", "w") as f:
		for id, dets in zip(read_ids, read_dets):
			f.write("%s %3d: " % (id, len(dets)) + " ".join([str(d) for d in dets]) + "\n")
# Output autocuts
try:
	autocuts = utils.allgatherv(np.array([[cut[1:] for cut in scan.autocut] for scan in myscans]),comm)
	autokeys = [cut[0] for cut in myscans[0].autocut]
	if comm.rank == 0:
		with open(root + "autocut.txt","w") as ofile:
			ofile.write(("#%29s" + " %15s"*len(autokeys)+"\n") % (("id",)+tuple(autokeys)))
			for id, acut in zip(read_ids, autocuts):
				ofile.write(("%30s" + " %7.3f %7.3f"*len(autokeys) + "\n") % ((id,)+tuple(1e-6*acut.reshape(-1))))
except (AttributeError, IndexError):
	pass
# Output sample stats
if comm.rank == 0:
	with open(root + "samps.txt", "w") as ofile:
		ofile.write("#%29s %4s %9s\n" % ("id", "ndet", "nsamp"))
		for id, ndet, nsamp in zip(read_ids, read_ndets, read_nsamp):
			ofile.write("%30s %4d %9d\n" % (id, ndet, nsamp))
# Prune fully autocut scans, now that we have output the autocuts
mydets  = [len(scan.dets) for scan in myscans]
myinds  = [ind  for ind, ndet in zip(myinds, mydets) if ndet > 0]
myscans = [scan for scan,ndet in zip(myscans,mydets) if ndet > 0]
L.info("Pruned %d fully autocut tods" % np.sum(read_ndets==0))

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
del myscans # scans do take up some space, even without the tod being read in
myinds, myscans = scanutils.read_scans(filelist, myinds, actscan.ACTScan,
		db, dets=args.dets, downsample=config.get("downsample"), hwp_resample=config.get("hwp_resample"))

# I would like to be able to do on-the-fly nmat computation.
# However, preconditioners depend on the noise matrix.
# Hence, to be able to do this, Eqsys initialization must go like this:
# 1. Set up plain signals (no priors or postprocessing)
# 2. Set up plain filters (no postprocessing)
# 3. Initialize Eqsys
# 4. Call eqsys.calc_b, which updates the noise matrices
# 5. Loop through signals again, this time setting up preconditioenrs and
#    priors, which can be inserted into the signals inside Eqsys
# 6. Loop through filters again, setting up any associated signal postprocessing
# It is a good deal less tidy than what I currently have. And it will break
# as soon as a filter or signal that depends on the noise matrix appears.

def apply_scan_limits(scans, params):
	"""Return subset of scans which matches the limits
	defined via lim_ keys in params."""
	# We support either telescope coordinates or object coordinates for now.
	# In both cases, the function returns az, el, ra, dec
	def tele_pos(scan):
		t, az, el = np.mean(scan.boresight[::10],0)
		t = uscan.mjd0 + t / (24.*60*60)
		ra, dec = coordinates.transform("hor","cel",[[az],[el]], time=[t], site=scan.site)[:,0]
		return az, el, ra, dec
	def obj_pos(scan, obj):
		ra, dec = coordinates.ephem_pos(obj, scan.mjd0)
		az, el  = coordinates.transform("cel","hor",[[ra],[dec]],time=[scan.mjd0], site=scan.site)[:,0]
		return az, el, ra, dec
	# Parse a filter description and build a corresponding matcher function
	def build_filter(objname, fun, cname, limval):
		def f(scan):
			cdict = {"az":0,"el":1,"ra":2,"dec":3}
			if objname == "Tele": coords = tele_pos(scan)
			else: coords = obj_pos(scan, objname)
			cval =coords[cdict[cname]]
			if fun == "min": return cval >= limval
			elif fun == "max": return cval <= limval
			raise ValueError("Unknown scan filter criterion: " + fun)
		return f
	# Set up all the matchers
	filters = []
	for key in params:
		if not key.startswith("lim_"): continue
		_, objname, fun, coord = key.split("_")
		filters.append(build_filter(objname, fun, coord, float(params[key])))
	# Evaluate each scan, accepting only those that match all our matchers
	res = []
	for scan in scans:
		accept = [filter(scan) for filter in filters]
		if np.all(accept):
			res.append(scan)
	return res

def build_noise_stats(myscans, comm):
	ids    = utils.allgatherv([scan.id    for scan in myscans], comm)
	ndets  = utils.allgatherv([scan.ndet  for scan in myscans], comm)
	srates = utils.allgatherv([scan.srate for scan in myscans], comm)
	gdets  = utils.allgatherv(np.concatenate([scan.dets       for scan in myscans]), comm)
	ivars  = utils.allgatherv(np.concatenate([scan.noise.ivar for scan in myscans]), comm)
	offs   = utils.cumsum(ndets, endpoint=True)
	res    = []
	for i, id in enumerate(ids):
		o1, o2 = offs[i], offs[i+1]
		dsens = (ivars[o1:o2]*srates[i])**-0.5
		asens = (np.sum(ivars[o1:o2])*srates[i])**-0.5
		dets  = gdets[o1:o2]
		# We want sorted dets
		inds  = np.argsort(dets)
		dets, dsens = dets[inds], dsens[inds]
		line = {"id": id, "asens": asens, "dsens": dsens, "dets": dets}
		res.append(line)
	inds = np.argsort(ids)
	res = [res[ind] for ind in inds]
	return res

def write_noise_stats(fname, stats):
	with open(fname, "w") as f:
		for line in stats:
			f.write("%s %6.3f :: " % (line["id"],line["asens"]))
			f.write(" ".join(["%13s" % ("%s:%7.3f" % (d,s)) for d,s in zip(line["dets"],line["dsens"])]) + "\n")

def setup_extra_transforms(param):
	extra = []
	if "p9" in param:
		# Planet 9 search coordinate system: p9=elemfile:tref. Includes
		# both parallax and motion compensation
		toks     = param["p9"].split(":")
		elemfile = toks[0]
		tref     = float(toks[1]) if len(toks)>1 else 1380000000.0
		tref     = utils.ctime2mjd(tref)
		obj      = ephemeris.read_object(elemfile)
		p9       = planet9.MotionCompensator(obj)
		def trf(pos, time):
			# We ignore the polarization rotation for now
			opos = pos.copy()
			opos[:2] = p9.compensate(pos[:2], time, tref)
			return opos
		extra.append(trf)
	if "parallax" in param:
		# Simple parallax compensation. parallax=dist, with dist in AU
		dist = float(param["parallax"])
		def trf(pos, time):
			opos = pos.copy()
			opos[:2] = parallax.earth2sun(pos[:2], time, dist)
			return opos
		extra.append(trf)
	return extra

# UGLY HACK: Handle individual output file mode
nouter = 1
if args.individual:
	nouter = len(myscans)
	ocomm, comm = comm, mpi.COMM_SELF
	myscans_tot = myscans
	root_tot = root
for out_ind in range(nouter):
	if args.individual:
		myscans = myscans_tot[out_ind:out_ind+1]
		root = root_tot + myscans[0].entry.id + "_"

	# 1. Initialize signals
	L.info("Initializing signals")
	signals = []
	for param in signal_params:
		effname = get_effname(param)
		active_scans = apply_scan_limits(myscans, param)
		if param["type"] == "cut":
			signal = mapmaking.SignalCut(active_scans, dtype=dtype, comm=comm, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes")
			signal_cut = signal
		elif param["type"] == "map":
			area = enmap.read_map(param["value"])
			area = enmap.zeros((args.ncomp,)+area.shape[-2:], area.wcs, dtype)
			signal = mapmaking.SignalMap(active_scans, area, comm=comm, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes", sys=param["sys"], extra=setup_extra_transforms(param))
		elif param["type"] == "fmap":
			area = enmap.read_map(param["value"])
			area = enmap.zeros((args.ncomp,)+area.shape[-2:], area.wcs, dtype)
			signal = mapmaking.SignalMapFast(active_scans, area, comm=comm, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes", sys=param["sys"], extra=setup_extra_transforms(param))
		elif param["type"] == "dmap":
			area = dmap.read_map(param["value"], bbox=mybbox, tshape=tshape, comm=comm)
			area = dmap.zeros(area.geometry.aspre(args.ncomp).astype(dtype))
			signal = mapmaking.SignalDmap(active_scans, mysubs, area, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes", sys=param["sys"], extra=setup_extra_transforms(param))
		elif param["type"] == "fdmap":
			area = dmap.read_map(param["value"], bbox=mybbox, tshape=tshape, comm=comm)
			area = dmap.zeros(area.geometry.aspre(args.ncomp).astype(dtype))
			signal = mapmaking.SignalDmapFast(active_scans, mysubs, area, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes", sys=param["sys"], extra=setup_extra_transforms(param))
		elif param["type"] == "bmap":
			area = enmap.read_map(param["value"])
			area = enmap.zeros((args.ncomp,)+area.shape[-2:], area.wcs, dtype)
			signal = mapmaking.SignalMapBuddies(active_scans, area, comm=comm, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes", sys=param["sys"], extra=setup_extra_transforms(param))
		elif param["type"] == "scan":
			res = float(param["res"])*utils.arcmin
			tol = float(param["tol"])*utils.degree
			col_major = True
			patterns, mypids = scanutils.classify_scanning_patterns(active_scans, comm=comm, tol=tol)
			L.info("Found %d scanning patterns" % len(patterns))
			# Define our phase maps
			nrow, ncol = active_scans[0].dgrid
			array_dets = np.arange(nrow*ncol)
			if col_major: array_dets = array_dets.reshape(nrow,ncol).T.reshape(-1)
			det_unit   = nrow if col_major else ncol
			areas      = mapmaking.PhaseMap.zeros(patterns, array_dets, res=res, det_unit=det_unit, dtype=dtype)
			signal     = mapmaking.SignalPhase(active_scans, areas, mypids, comm, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes")
		else:
			raise ValueError("Unrecognized signal type '%s'" % param["type"])
		signals.append(signal)

	def matching_signals(params, signal_params, signals):
		for sparam, signal in zip(signal_params, signals):
			if sparam["name"] in params and params[sparam["name"]] == "yes":
				yield sparam, signal

	# 2. Initialize filters. The only complicated part here is finding the
	# corresponding signal and supporting both enmaps and dmaps.
	L.info("Initializing filters")
	filters = []
	filters2= []
	src_filters = []
	map_add_filters = []
	for param in filter_params:
		if param["name"] == "scan":
			daz, nt, mode, niter = float(param["daz"]), int(param["nt"]), int(param["value"]), int(param["niter"])
			nhwp = int(param["nhwp"])
			weighted = int(param["weighted"])
			if mode == 0: continue
			filter = mapmaking.FilterPickup(daz=daz, nt=nt, nhwp=nhwp, niter=niter)
			if mode >= 2:
				for sparam, signal in matching_signals(param, signal_params, signals):
					if sparam["type"] == "map" or sparam["type"] == "bmap":
						prec = mapmaking.PreconMapBinned(signal, signal_cut, myscans, weights=[], noise=False, hits=False)
					elif sparam["type"] == "dmap":
						prec = mapmaking.PreconDmapBinned(signal, signal_cut, myscans, weights=[], noise=False, hits=False)
					else:
						raise NotImplementedError("Scan postfiltering for '%s' signals not implemented" % sparam["type"])
					signal.post.append(mapmaking.PostPickup(myscans, signal, signal_cut, prec, daz=daz, nt=-nt, weighted=weighted>0))
		elif param["name"] == "common":
			mode = int(param["value"])
			if mode == 0: continue
			filter = mapmaking.FilterCommonBlockwise()
		elif param["name"] == "hwp":
			nmode = int(param["value"])
			filter = mapmaking.FilterHWPNotch(nmode)
		elif param["name"] == "add":
			if "map" not in param: raise ValueError("-F add/sub needs a map file to subtract. e.g. -F add:2,map=foo.fits")
			mode, sys, fname, mul = int(param["value"]), param["sys"], param["map"], float(param["mul"])
			nopol = int(param["nopol"])
			tmul = float(param["tmul"])
			if mode == 0: continue
			if param["type"] == "auto":
				param["type"] = ("dmap" if os.path.isdir(fname) else "map")
			if param["type"] == "dmap":
				# Warning: This only works if a dmap has already been initialized, and
				# has compatible coordinate system and pixelization! This is the disadvantage
				# of dmaps - they are so closely tied to a set of scans that they only work
				# in the coordinate system where the scans are reasonably local.
				m = dmap.read_map(fname, bbox=mybbox, tshape=tshape, comm=comm).astype(dtype)
				if nopol: m[1:] = 0
				filter = mapmaking.FilterAddDmap(myscans, mysubs, m, sys=sys, mul=mul, tmul=tmul)
			else:
				m = enmap.read_map(fname).astype(dtype)
				if nopol: m[1:] = 0
				filter = mapmaking.FilterAddMap(myscans, m, sys=sys, mul=mul, tmul=tmul)
			map_add_filters.append(filter)
			if mode >= 2:
				# In post mode we subtract the map that was added before each output. That's
				# why mul is -1 here
				for sparam, signal in matching_signals(param, signal_params, signals):
					assert sparam["sys"] == param["sys"]
					assert signal.area.shape[-2:] == m.shape[-2:]
					signal.post.append(mapmaking.PostAddMap(m, mul=-mul))
		elif param["name"] == "addphase" or param["name"] == "fitphase":
			if "map" not in param: raise ValueError("-F addphase/subphase/fitphase needs a phase dir to subtract. e.g. -F addphase:map=foo")
			mode, fname, mul, tmul = int(param["value"]), param["map"], float(param["mul"]), float(param["tmul"])
			tol = float(param["tol"])*utils.degree
			# Read the info file to see which scanning patterns were used in the phase dir
			phasemap = mapmaking.PhaseMap.read(fname)
			npat     = len(phasemap.patterns)
			# Find which phase map part each scan corresponds to. We get all the scan
			# boxes, and then add our existing scanning pattern boxes as references.
			# We can then just see which scans get grouped with which patterns.
			my_boxes = scanutils.get_scan_bounds(myscans)
			boxes = utils.allgatherv(my_boxes, comm)
			rank  = utils.allgatherv(np.full(len(my_boxes),comm.rank),      comm)
			boxes = np.concatenate([phasemap.patterns, boxes], 0)
			labels= utils.label_unique(boxes, axes=(1,2), atol=tol)
			if comm.rank == 0:
				print "labels"
				for b,l in zip(boxes, labels):
					print "%8.3f %8.3f %8.3f %5d" % (b[0,0]/utils.degree,b[0,1]/utils.degree,b[1,1]/utils.degree,l)
			pids  = utils.find(labels[:npat], labels[npat:])
			mypids= pids[rank==comm.rank]
			if param["name"] == "addphase":
				filter = mapmaking.FilterAddPhase(myscans, phasemap, mypids, mmul=mul, tmul=tmul)
			else:
				filter = mapmaking.FilterDeprojectPhase(myscans, phasemap, mypids, int(param["perdet"])>0, mmul=mul, tmul=tmul)
		elif param["name"] == "scale":
			value = float(param["value"])
			if value == 1: continue
			filter = mapmaking.FilterScale(value)
		elif param["name"] == "buddy":
			if "map" not in param: raise ValueError("-F buddy needs a map file to subtract. e.g. -F buddy:map=foo.fits")
			mode  = int(param["value"])
			sys   = param["sys"]
			fname = param["map"].format(id=myscans[0].entry.id)
			mul   = float(param["mul"])
			tmul  = float(param["tmul"])
			pertod= int(param["pertod"])
			nstep = int(param["nstep"])
			prec  = param["prec"]
			if mode == 0: continue
			# Two types of buddy subtraction: The one based on an input map,
			# and the one where a map is computed internally per tod. Both need
			# an input map, but for the pertod buddy, this just indicates the
			# pixelization to use for the internally generated buddy map.
			if param["type"] == "auto":
				param["type"] = ("dmap" if os.path.isdir(fname) else "map")
			if param["type"] != "dmap":
				m = enmap.read_map(fname).astype(dtype)
				if not pertod:
					filter = mapmaking.FilterBuddy(myscans, m, sys=sys, mul=-mul, tmul=tmul)
				else:
					1/0 # FIXME
					m = enmap.zeros((args.ncomp,)+m.shape[-2:], m.wcs, dtype)
					filter = mapmaking.FilterBuddyPertod(m, sys=sys, mul=-mul, tmul=tmul, nstep=nstep, prec=prec)
			else:
				# Warning: This only works if a dmap has already been initialized etc.
				m = dmap.read_map(fname, bbox=mybbox, tshape=tshape, comm=comm).astype(dtype)
				if not pertod:
					filter = mapmaking.FilterBuddyDmap(myscans, mysubs, m, sys=sys, mul=-mul, tmul=tmul)
				else:
					raise NotImplementedError("FIXME: Implement per tod buddy subtraction with dmaps")
		elif param["name"] == "src":
			if param["value"] == 0: continue
			if "params" not in param: srcs = myscans[0].pointsrcs
			else: srcs = pointsrcs.read(param["params"])
			# Restrict to chosen amplitude
			if "snr" in srcs.dtype.names:
				srcs = srcs[srcs.snr >= float(param["snr"])]
			srcparam = pointsrcs.src2param(srcs)
			srcparam = srcparam.astype(np.float64)
			filter = mapmaking.FilterAddSrcs(myscans, srcparam, sys=param["sys"], mul=-float(param["mul"]))
			src_filters.append(filter)
		else:
			raise ValueError("Unrecognized fitler name '%s'" % param["name"])
		# Add to normal filters of post-noise-model filters based on parameters
		if "postnoise" in param and int(param["postnoise"]) > 0:
			print "postnosie", param["name"]
			filters2.append(filter)
		else:
			filters.append(filter)
	# If any filters were added, append a gapfilling operation, since the filters may have
	# put large values in the gaps, and these may not be representable by our cut model
	if len(filters) > 0:
		filters.append(mapmaking.FilterGapfill())
	if len(filters2) > 0:
		filters2.append(mapmaking.FilterGapfill())

	L.info("Initializing mapfilters")
	for param in mapfilter_params:
		if param["name"] == "gauss":
			scale = float(param["value"])*utils.arcmin*utils.fwhm
			cap   = float(param["cap"])
			filter= mapmaking.MapfilterGauss(scale, cap=cap)
			for sparam, signal in matching_signals(param, signal_params, signals):
				print "adding gauss filter to " + signal.name
				signal.filters.append(filter)

	# Initialize weights. Done in a hacky manner for now. This and the above needs
	# to be reworked.
	weights = []
	if config.get("tod_window"):
		weights.append(mapmaking.FilterWindow(config.get("tod_window")))

	L.info("Initializing equation system")
	eqsys = mapmaking.Eqsys(myscans, signals, filters=filters, filters2=filters2, weights=weights, dtype=dtype, comm=comm)

	L.info("Initializing RHS")
	eqsys.calc_b()

	noise_stats = build_noise_stats(myscans, comm)
	if comm.rank == 0: write_noise_stats(root + "noise.txt", noise_stats)

	#for si, scan in enumerate(myscans):
	#	tod = np.zeros([scan.ndet, scan.nsamp], dtype)
	#	imaps  = eqsys.dof.unzip(eqsys.b)
	#	iwork = [signal.prepare(map) for signal, map in zip(eqsys.signals, imaps)]
	#	for signal, work in zip(eqsys.signals, iwork)[::-1]:
	#		signal.forward(scan, tod, work)
	#	np.savetxt("test_enki1/tod_Pb%d.txt" % si, tod[0])

	L.info("Initializing preconditioners")
	for param, signal in zip(signal_params, signals):
		# Null-preconditioner common for all types
		if "prec" in param and param["prec"] == "null":
			signal.precon = mapmaking.PreconNull()
			print "Warning: map and cut precon must have compatible units"
			continue
		if param["type"] == "cut":
			signal.precon = mapmaking.PreconCut(signal, myscans)
		elif param["type"] in ["map","bmap","fmap"]:
			prec_signal = signal if param["type"] != "bmap" else signal.get_nobuddy()
			if param["prec"] == "bin":
				signal.precon = mapmaking.PreconMapBinned(prec_signal, signal_cut, myscans, weights)
			elif param["prec"] == "jacobi":
				signal.precon = mapmaking.PreconMapBinned(prec_signal, signal_cut, myscans, weights, noise=False)
			elif param["prec"] == "hit":
				print "Warning: map and cut precon must have compatible units"
				signal.precon = mapmaking.PreconMapHitcount(prec_signal, signal_cut, myscans)
			elif param["prec"] == "tod":
				signal.precon = mapmaking.PreconMapTod(prec_signal, signal_cut, myscans, weights)
			else: raise ValueError("Unknown map preconditioner '%s'" % param["prec"])
			if "nohor" in param and param["nohor"] != "no":
				prior_weight = signal.precon.div[0,0]
				prior_weight /= (np.mean(prior_weight)*prior_weight.shape[-1])**0.5
				prior_weight *= float(param["nohor"])
				signal.prior = mapmaking.PriorMapNohor(prior_weight)
			if "unmix" in param and param["unmix"] != "no":
				signal.prior = mapmaking.PriorNorm(float(param["unmix"]))
		elif param["type"] in ["dmap","fdmap"]:
			if param["prec"] == "bin":
				signal.precon = mapmaking.PreconDmapBinned(signal, signal_cut, myscans, weights)
			elif param["prec"] == "jacobi":
				signal.precon = mapmaking.PreconDmapBinned(signal, signal_cut, myscans, weights, noise=False)
			elif param["prec"] == "hit":
				signal.precon = mapmaking.PreconDmapHitcount(signal, signal_cut, myscans)
			else: raise ValueError("Unknown dmap preconditioner '%s'" % param["prec"])
			if "nohor" in param and param["nohor"] != "no":
				prior_weight  = signal.precon.div[0,0]
				prior_weight /= (dmap.sum(prior_weight)/prior_weight.size*prior_weight.shape[-1])**0.5
				prior_weight *= float(param["nohor"])
				signal.prior = mapmaking.PriorDmapNohor(prior_weight)
			if "unmix" in param and param["unmix"] != "no":
				signal.prior = mapmaking.PriorNorm(float(param["unmix"]))
		elif param["type"] == "scan":
			signal.precon = mapmaking.PreconPhaseBinned(signal, signal_cut, myscans, weights)
		else:
			raise ValueError("Unrecognized signal type '%s'" % param["type"])

	# Can initialize extra postprocessors here, if they depend on the noise model
	# or preconditioners.
	L.info("Initializing extra postprocessors")

	L.info("Writing preconditioners")
	mapmaking.write_precons(signals, root)

	for param, signal in zip(signal_params, signals):
		if config.get("crossmap"):
			if param["type"] not in ["map","bmap","fmap","dmap"]: continue
			L.info("Computing crosslink map")
			cmap = mapmaking.calc_crosslink_map(signal, signal_cut, myscans, weights)
			signal.write(root, "crosslink", cmap)
			del cmap
		if config.get("icovmap"):
			if param["type"] not in ["map","bmap","fmap","dmap"]: continue
			L.info("Computing icov map")
			shape, wcs = signal.area.shape, signal.area.wcs
			# Use equidistant pixel spacing for robustness in non-cylindrical coordinates
			step = utils.nint(np.abs(config.get("icovstep")/wcs.wcs.cdelt[::-1]))
			pos  = np.mgrid[0.5:shape[-2]/step[-2],0.5:shape[-1]/step[-1]].reshape(2,-1).T
			if pos.size == 0:
				L.debug("Not enough pixels to compute icov for step size %f. Skipping icov" % config.get("icovstep"))
				continue
			# Apply the y skew
			yskew     = utils.nint(config.get("icovyskew")/wcs.wcs.cdelt[1])
			pos[:,0] += pos[:,1] * yskew * 1.0 / step[0]
			# Go from grid indices to pixels
			pos       = (pos*step % shape[-2:]).astype(int)
			print pos.shape, pos.dtype
			icov = mapmaking.calc_icov_map(signal, myscans, pos, weights)
			signal.write(root, "icov", icov)
			if comm.rank == 0:
				np.savetxt(root + signal.name + "_icov_pix.txt", pos, "%6d %6d")
			del icov
		if src_filters:
			if param["type"] not in ["map","bmap","fmap","dmap"]: continue
			L.info("Computing point source map")
			srcmap = mapmaking.calc_ptsrc_map(signal, signal_cut, myscans, src_filters)
			signal.write(root, "srcs", srcmap)
			del srcmap
	if map_add_filters:
		L.info("Writing added/subtracted template map")
		for fi, filter in enumerate(map_add_filters):
			map   = filter.map * filter.mul
			# It would be nice to only write the total added map, but they may be a mix
			# of dmaps and enmap, or even have different coordinate systems, so this is safer.
			# Anyway, in almost all the cases only a single map will be added
			oname = root + "added" + ("_%d"%fi if len(map_add_filters)>1 else "") + ".fits"
			if isinstance(map, dmap.Dmap):
				map.write(oname)
			elif comm.rank == 0:
				enmap.write_map(oname, map)
			del map

	L.info("Writing RHS")
	eqsys.write(root, "rhs", eqsys.b)

	L.info("Computing approximate map")
	x = eqsys.M(eqsys.b)
	eqsys.write(root, "bin", x)

	utils.mkdir(root + "cgstate")
	cgpath = root + "cgstate/cgstate%02d.hdf" % comm.rank

	if nmax > 0:
		L.info("Solving")
		cg = CG(eqsys.A, eqsys.b, M=eqsys.M, dot=eqsys.dot)
		dump_steps = [int(w) for w in args.dump.split(",")]
		# Start from saved cg info if available
		if resume > 0 and os.path.isfile(cgpath):
			cg.load(cgpath)
		assert cg.i == comm.bcast(cg.i), "Inconsistent CG step in mapmaker!"
		while cg.i < nmax:
			with bench.mark("cg_step"):
				cg.step()
			# Save cg state
			if resume != 0 and cg.i % np.abs(resume) == 0:
				cg.save(cgpath)
			dt = bench.stats["cg_step"]["time"].last
			if cg.i in dump_steps or cg.i % dump_steps[-1] == 0 or cg.i == nmax:
				if args.prepost: eqsys.write(root, "map%04d_prepost" % cg.i, cg.x)
				x = eqsys.postprocess(cg.x)
				eqsys.write(root, "map%04d" % cg.i, x)
				if args.tod_debug:
					eqsys.A(cg.x, debug_file = root + "tod_debug%04d.hdf" % cg.i)
			bench.stats.write(benchfile)
			ptime = bench.stats["M"]["time"].last
			L.info("CG step %5d %15.7e %6.1f %6.3f %6.3f" % (cg.i, cg.err, dt, dt/max(1,len(eqsys.scans)), ptime))
