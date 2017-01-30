import numpy as np, time, h5py, copy, argparse, os, sys, pipes, shutil, re
from enlib import enmap, utils, pmat, fft, config, array_ops, mapmaking, nmat, errors, mpi
from enlib import log, bench, dmap, coordinates, scan as enscan, rangelist, scanutils
from enlib import pointsrcs, bunch
from enlib.cg import CG
from enlib.source_model import SourceModel
from enact import actscan, nmat_measure, filedb, todinfo

config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("downsample", 1, "Factor with which to downsample the TOD")
config.default("map_cg_nmax", 500, "Max number of CG steps to perform in map-making")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
config.default("task_dist", "size", "How to assign scans to each mpi task. Can be 'plain' for comm.rank:n:comm.size-type assignment, 'size' for equal-total-size assignment. The optimal would be 'time', for equal total time for each, but that's not implemented currently.")
config.default("gfilter_jon", False, "Whether to enable Jon's ground filter.")
config.default("map_ptsrc_handling", "subadd", "How to handle point sources in the map. Can be 'none' for no special treatment, 'subadd' to subtract from the TOD and readd in pixel space, and 'sim' to simulate a pointsource-only TOD.")
config.default("map_ptsrc_sys", "cel", "Coordinate system the point source positions are specified in. Default is 'cel'")
config.default("map_format", "fits", "File format to use when writing maps. Can be 'fits', 'fits.gz' or 'hdf'.")

# Default signal parameters
config.default("signal_sky_default",   "use=no,type=fmap,name=sky,sys=cel,prec=bin", "Default parameters for sky map")
config.default("signal_hor_default",   "use=no,type=map,name=hor,sys=hor,prec=bin", "Default parameters for ground map")
config.default("signal_sun_default",   "use=no,type=map,name=sun,sys=hor:Sun,prec=bin,lim_Sun_min_el=0", "Default parameters for sun map")
config.default("signal_moon_default",  "use=no,type=map,name=moon,sys=hor:Moon,prec=bin,lim_Moon_min_el=0", "Default parameters for moon map")
config.default("signal_jupiter_default",  "use=no,type=map,name=jupiter,sys=hor:Jupiter,prec=bin,lim_Jupiter_min_el=0", "Default parameters for jupiter map")
config.default("signal_saturn_default",  "use=no,type=map,name=saturn,sys=hor:Saturn,prec=bin,lim_Saturn_min_el=0", "Default parameters for saturn map")
config.default("signal_uranus_default",  "use=no,type=map,name=uranus,sys=hor:Uranus,prec=bin,lim_Uranus_min_el=0", "Default parameters for uranus map")
config.default("signal_cut_default",   "use=no,type=cut,name=cut,ofmt={name}_{rank:03},output=no,use=yes", "Default parameters for cut (junk) signal")
config.default("signal_scan_default",  "use=no,type=scan,name=scan,ofmt={name}_{pid:02}_{az0:.0f}_{az1:.0f}_{el:.0f},2way=yes,res=2,tol=0.5", "Default parameters for scan/pickup signal")
# Default filter parameters
config.default("filter_scan_default",  "use=no,name=scan,value=1,naz=8,nt=10,nhwp=0,weighted=1,niter=3,sky=yes", "Default parameters for scan/pickup filter")
config.default("filter_sub_default",   "use=no,name=sub,value=1,sys=cel,type=map,mul=1,tmul=1,sky=yes", "Default parameters for map subtraction filter")
config.default("filter_src_default",   "use=no,name=src,value=1,sys=cel,mul=1,sky=yes", "Default parameters for point source subtraction filter")
config.default("filter_buddy_default",   "use=no,name=buddy,value=1,mul=1,type=map,sys=cel,tmul=1,sky=yes,pertod=0,nstep=200,prec=bin", "Default parameters for map subtraction filter")
config.default("filter_common_default", "use=no,name=common,value=1", "Default parameters for blockwise common mode filter")

config.default("tod_window", 5.0, "Number of samples to window the tod by on each end")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("filelist")
parser.add_argument("odir")
parser.add_argument("prefix",nargs="?")
parser.add_argument("-d", "--dump", type=str, default="1,2,5,10,20,50,100,200,300,400,500,600,800,1000,1200,1500,2000,3000,4000,5000,6000,8000,10000", help="CG map dump steps")
parser.add_argument("--ncomp",      type=int, default=3,  help="Number of stokes parameters")
parser.add_argument("--dets",       type=str, default=0,  help="Detector slice")
parser.add_argument("--dump-config", action="store_true", help="Dump the configuration file to standard output.")
parser.add_argument("-S", "--signal", action="append",    help="Signals to solve for. For example -S sky:area.fits -S scan would solve for the sky map and scan pickup maps jointly, using area.fits as the map template.")
parser.add_argument("-F", "--filter", action="append")
parser.add_argument("--group-tods", action="store_true")
parser.add_argument("--individual", action="store_true")
args = parser.parse_args()

if args.dump_config:
	print config.to_str()
	sys.exit(0)

dtype = np.float32 if config.get("map_bits") == 32 else np.float64
comm  = mpi.COMM_WORLD
nmax  = config.get("map_cg_nmax")
ext   = config.get("map_format")
tshape= (720,720)

filedb.init()
db = filedb.data
filelist = todinfo.get_tods(args.filelist, filedb.scans)
if args.group_tods:
	filelist = data.group_ids(filelist)

utils.mkdir(args.odir)
root = args.odir + "/" + (args.prefix + "_" if args.prefix else "")

# Dump our settings
if comm.rank == 0:
	config.save(root + "config.txt")
	with open(root + "args.txt","w") as f:
		f.write(" ".join([pipes.quote(a) for a in sys.argv[1:]]) + "\n")
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
filter_params = setup_params("filter", ["scan","sub"], {"use":"no"})

# Read in all our scans
L.info("Reading %d scans" % len(filelist))
myinds = np.arange(len(filelist))[comm.rank::comm.size]
myinds, myscans = scanutils.read_scans(filelist, myinds, actscan.ACTScan,
		db, dets=args.dets, downsample=config.get("downsample"))

# Collect scan info. This currently fails if any task has empty myinds
read_ids  = [filelist[ind] for ind in utils.allgatherv(myinds, comm)]
read_ntot = len(read_ids)
L.info("Found %d tods" % read_ntot)
if read_ntot == 0:
	L.info("Giving up")
	sys.exit(1)
read_ndets= utils.allgatherv([len(scan.dets) for scan in myscans], comm)
read_nsamp= utils.allgatherv([scan.cut.size-scan.cut.sum() for scan in myscans], comm)
read_dets = utils.uncat(utils.allgatherv(np.concatenate([scan.dets for scan in myscans]),comm), read_ndets)
# Save accept list
if comm.rank == 0:
	with open(root + "accept.txt", "w") as f:
		for id, dets in zip(read_ids, read_dets):
			f.write("%s %3d: " % (id, len(dets)) + " ".join([str(d) for d in dets]) + "\n")
# Output autocuts
autocuts = utils.allgatherv(np.array([[cut[1:] for cut in scan.autocut] for scan in myscans]),comm)
autokeys = [cut[0] for cut in myscans[0].autocut]
if comm.rank == 0:
	with open(root + "autocut.txt","w") as ofile:
		ofile.write(("#%29s" + " %15s"*len(autokeys)+"\n") % (("id",)+tuple(autokeys)))
		for id, acut in zip(read_ids, autocuts):
			ofile.write(("%30s" + " %7.3f %7.3f"*len(autokeys) + "\n") % ((id,)+tuple(1e-6*acut.reshape(-1))))
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
myinds, myscans = scanutils.read_scans(filelist, myinds, actscan.ACTScan,
		db, dets=args.dets, downsample=config.get("downsample"))

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
			signal = mapmaking.SignalMap(active_scans, area, comm=comm, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes", sys=param["sys"])
		elif param["type"] == "fmap":
			area = enmap.read_map(param["value"])
			area = enmap.zeros((args.ncomp,)+area.shape[-2:], area.wcs, dtype)
			signal = mapmaking.SignalMapFast(active_scans, area, comm=comm, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes", sys=param["sys"])
		elif param["type"] == "dmap":
			area = dmap.read_map(param["value"], bbox=mybbox, tshape=tshape, comm=comm)
			area = dmap.zeros(area.geometry.aspre(args.ncomp).astype(dtype))
			signal = mapmaking.SignalDmap(active_scans, mysubs, area, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes", sys=param["sys"])
		elif param["type"] == "fdmap":
			area = dmap.read_map(param["value"], bbox=mybbox, tshape=tshape, comm=comm)
			area = dmap.zeros(area.geometry.aspre(args.ncomp).astype(dtype))
			signal = mapmaking.SignalDmapFast(active_scans, mysubs, area, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes", sys=param["sys"])
		elif param["type"] == "bmap":
			area = enmap.read_map(param["value"])
			area = enmap.zeros((args.ncomp,)+area.shape[-2:], area.wcs, dtype)
			signal = mapmaking.SignalMapBuddies(active_scans, area, comm=comm, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes", sys=param["sys"])
		elif param["type"] == "scan":
			res = float(param["res"])*utils.arcmin
			tol = float(param["tol"])*utils.degree
			patterns, mypids = scanutils.classify_scanning_patterns(active_scans, comm=comm, tol=tol)
			L.info("Found %d scanning patterns" % len(patterns))
			signal = mapmaking.SignalPhase(active_scans, mypids, patterns, active_scans[0].dgrid, res=res, dtype=dtype, comm=comm, name=effname, ofmt=param["ofmt"], output=param["output"]=="yes")
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
	for param in filter_params:
		if param["name"] == "scan":
			naz, nt, mode, niter = int(param["naz"]), int(param["nt"]), int(param["value"]), int(param["niter"])
			nhwp = int(param["nhwp"])
			weighted = int(param["weighted"])
			if mode == 0: continue
			filter = mapmaking.FilterPickup(naz=naz, nt=nt, nhwp=nhwp, niter=niter)
			if mode >= 2:
				for sparam, signal in matching_signals(param, signal_params, signals):
					if sparam["type"] == "map" or sparam["type"] == "bmap":
						prec_ptp = mapmaking.PreconMapBinned(signal, signal_cut, myscans, weights=[], noise=False, hits=False)
					elif sparam["type"] == "dmap":
						prec_ptp = mapmaking.PreconDmapBinned(signal, signal_cut, myscans, weights=[], noise=False, hits=False)
					else:
						raise NotImplementedError("Scan postfiltering for '%s' signals not implemented" % sparam["type"])
					signal.post.append(mapmaking.PostPickup(myscans, signal, signal_cut, prec_ptp, naz=naz, nt=nt, weighted=weighted>0))
		elif param["name"] == "common":
			mode = int(param["value"])
			if mode == 0: continue
			filter = mapmaking.FilterCommonBlockwise()
		elif param["name"] == "sub":
			if "map" not in param: raise ValueError("-F sub needs a map file to subtract. e.g. -F sub:2,map=foo.fits")
			mode, sys, fname, mul = int(param["value"]), param["sys"], param["map"], float(param["mul"])
			tmul = float(param["tmul"])
			if mode == 0: continue
			if param["type"] == "dmap":
				# Warning: This only works if a dmap has already been initialized, and
				# has compatible coordinate system and pixelization! This is the disadvantage
				# of dmaps - they are so closely tied to a set of scans that they only work
				# in the coordinate system where the scans are reasonably local.
				m = dmap.read_map(fname, bbox=mybbox, tshape=tshape, comm=comm).astype(dtype)
				filter = mapmaking.FilterAddDmap(myscans, mysubs, m, sys=sys, mul=-mul, tmul=tmul)
			else:
				m = enmap.read_map(fname).astype(dtype)
				filter = mapmaking.FilterAddMap(myscans, m, sys=sys, mul=-mul, tmul=tmul)
			if mode >= 2:
				for sparam, signal in matching_signals(param, signal_params, signals):
					assert sparam["sys"] == param["sys"]
					assert signal.area.shape[-2:] == m.shape[-2:]
					signal.post.append(mapmaking.PostAddMap(m, mul=mul))
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
			if param["type"] != "dmap":
				m = enmap.read_map(fname).astype(dtype)
				if not pertod:
					filter = mapmaking.FilterBuddy(myscans, m, sys=sys, mul=-mul, tmul=tmul)
				else:
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
			if "params" not in param: params = myscans[0].pointsrcs
			else: params = pointsrcs.read(param["params"])
			params = pointsrcs.src2param(params)
			params = params.astype(np.float64)
			print "FIXME: how to handle per-source beams? Forcing to relative for now"
			params[:,5:7] = 1
			params[:,7]   = 0
			filter = mapmaking.FilterAddSrcs(myscans, params, sys=param["sys"], mul=-float(param["mul"]))
		else:
			raise ValueError("Unrecognized fitler name '%s'" % param["name"])
		filters.append(filter)
	# If any filters were added, append a gapfilling operation, since the filters may have
	# put large values in the gaps, and these may not be representable by our cut model
	if len(filters) > 0:
		filters.append(mapmaking.FilterGapfill())

	# Initialize weights. Done in a hacky manner for now. This and the above needs
	# to be reworked.
	weights = []
	if config.get("tod_window"):
		weights.append(mapmaking.FilterWindow(config.get("tod_window")))

	L.info("Initializing equation system")
	eqsys = mapmaking.Eqsys(myscans, signals, filters=filters, weights=weights, dtype=dtype, comm=comm)

	L.info("Initializing RHS")
	eqsys.calc_b()

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
				signal.precon = mapmaking.PreconMapHitcount(prec_signal, signal_cut, myscans)
			else: raise ValueError("Unknown map preconditioner '%s'" % param["prec"])
			if "nohor" in param and param["nohor"] != "no":
				prior_weight = signal.precon.div[0,0]
				prior_weight /= (np.mean(prior_weight)*prior_weight.shape[-1])**0.5
				prior_weight *= float(param["nohor"])
				signal.prior = mapmaking.PriorMapNohor(prior_weight)
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
		elif param["type"] == "scan":
			signal.precon = mapmaking.PreconPhaseBinned(signal, signal_cut, myscans, weights)
		else:
			raise ValueError("Unrecognized signal type '%s'" % param["type"])

	# Can initialize extra postprocessors here, if they depend on the noise model
	# or preconditioners.
	L.info("Initializing extra postprocessors")

	L.info("Writing preconditioners")
	mapmaking.write_precons(signals, root)

	L.info("Writing RHS")
	eqsys.write(root, "rhs", eqsys.b)

	L.info("Computing approximate map")
	x = eqsys.M(eqsys.b)
	eqsys.write(root, "bin", x)

	if nmax > 0:
		L.info("Solving")
		cg = CG(eqsys.A, eqsys.b, M=eqsys.M, dot=eqsys.dot)
		dump_steps = [int(w) for w in args.dump.split(",")]
		while cg.i < nmax:
			with bench.mark("cg_step"):
				cg.step()
			dt = bench.stats["cg_step"]["time"].last
			if cg.i in dump_steps or cg.i % dump_steps[-1] == 0:
				x = eqsys.postprocess(cg.x)
				eqsys.write(root, "map%04d" % cg.i, x)
			bench.stats.write(benchfile)
			ptime = bench.stats["M"]["time"].last
			L.info("CG step %5d %15.7e %6.1f %6.3f %6.3f" % (cg.i, cg.err, dt, dt/max(1,len(eqsys.scans)), ptime))
