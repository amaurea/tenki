import numpy as np, os, time, sys
from pixell import config, utils, mpi, colors, bench
from sogma import logging, loading, device, gutils
# for its config
from sogma.loaders import sofast
# Have to do all imports that could contain configuration before
# building config.ArgumentParser because these may define variables.
# This means we can't bail out early when called with wrong arguments;
# we have to wait for potentially slow imports first. Oh well.
parser = config.ArgumentParser("sogma")
parser.add_argument("odir")
parser.add_argument("-C", "--context", type=str, default="lat", help="Either a config.yaml, context.yaml or a telescope name, like lat, satp1, etc.. The most general is config.yaml, which works with both the sofast and soslow loaders. Using just a context only works with sofast, and must be a context that includes the preprocess database. Telescope names expand to the default preprocess context using $SOPATH, which should point to the SO top-level directory (the one that contains the metadata directory).")
parser.add_argument("-Q", "--query",   type=str, default=None)
parser.add_argument("-p", "--prefix",  type=str, default=None)
parser.add_argument("-v", "--verbose", action="count", default=1)
parser.add_argument("-q", "--quiet",   action="count", default=0)
parser.add_argument("-L", "--loader",  type=str, default="auto")
parser.add_argument("-D", "--device",  type=str, default="auto")
parser.add_argument(      "--ignore",  type=str, default="recover")
parser.add_argument("-j", "--joint",   type=str, default="full")
parser.add_argument("-s", "--split",   type=float, default=0.75, help="Split obs-groups in time to keep them smaller than this many giga-samples. Larger groups use more memory. Small groups have more edge effects and may be slower to read in.")
parser.add_argument(      "--tsplit",  type=float, default=None, help="Split obs-groups in time to keep them shorter than this many seconds. See also --split")
parser.add_argument("-c", "--cont",    action="store_true")
parser.add_argument(      "--sel",     type=str, default=":")
parser.add_argument(      "--prealloc",type=int, default=1)
args = parser.parse_args()

def setup_buffers(dev, ntot, dtype=np.float32, ndet_guess=10000):
	# These are the big ones
	ctype = utils.complex_dtype(dtype)
	ftot = ntot//2 + ndet_guess
	dev.pools["pointing"]   .empty((3, ntot), dtype=dtype)
	dev.pools["tod"]        .empty(ntot, dtype=dtype)
	dev.pools["ft" ]        .empty(ftot, dtype=ctype)
	dev.pools["fft_scratch"].empty(ftot, dtype=ctype)
	dev.pools.reset()

def simple_deriv(a, h=1, dt=1):
	ap  = device.anypy(a)
	res = ap.zeros_like(a)
	res[...,h:-h] = (a[...,2*h:]-a[...,:-2*h])/(2*h*dt)
	return res

def measure_shift(a, b, dev=None):
	if dev is None: dev = device.get_device()
	ctype = utils.complex_dtype(a.dtype)
	n  = a.shape[-1]
	nf = n//2+1
	fa = dev.pools["ft"].zeros(a.shape[:-1]+(nf,), ctype)
	fb = dev.pools["pointing"].zeros(b.shape[:-1]+(nf,), ctype)
	dev.lib.rfft(a, fa)
	dev.lib.rfft(b, fb)
	# conjugate fa in-place
	fa.imag *= -1
	# multiply in-place. Won't broadcast if b is smaller
	fb *= fa
	# And back
	c  = dev.pools["pointing"].zeros(b.shape, b.dtype, reset=False)
	dev.lib.irfft(fb, c)
	# Measure the peak
	shift = (dev.np.argmax(c, -1)+n//2)%n-n//2
	return shift

def mean_peaks(a, bsize=1000, tol=8, dev=None):
	if dev is None: dev = device.get_device()
	# First measure the typical noise level
	aa   = dev.pools["pointing"].array(a)
	aa **= 2
	std  = dev.np.median(gutils.downgrade(aa, bsize=bsize),-1)**0.5
	# Find areas higher than tol times std
	with dev.pools["pointing"].as_allocator():
		mask = a > tol*std[:,None]
		# Average value in these. Median would be cumbersome. This is hopefully fine
		mean = dev.np.sum(a*mask,-1)/dev.np.sum(mask,-1)
	return mean.copy()

def analyse_heating(data, h=40, dev=None):
	if dev is None: dev = device.get_device()
	srate = (len(data.ctime)-1)/(data.ctime[-1]-data.ctime[0])
	# Estimate the absolute acceleration and heating rate
	acc     = dev.np.abs(simple_deriv(simple_deriv(dev.np.array(data.boresight[1]), h=h, dt=1/srate), h=h, dt=1/srate)).astype(data.tod.dtype)
	heating = simple_deriv(data.tod, h=h, dt=1/srate)
	# Estimate the heating's delay relative to the acceleration
	shift   = measure_shift(acc, heating, dev=dev)/srate
	# Estimate the total amount of heating
	rate    = mean_peaks(heating, dev=dev)
	dtype   = [("detid", data.detids.dtype),("xi","d"),("eta","d"),("shift", "d"), ("rate", "d")]
	info    = np.zeros(len(data.detids), dtype).view(np.recarray)
	info.detid = data.detids
	info.xi    = data.point_offset[:,1]/utils.degree
	info.eta   = data.point_offset[:,0]/utils.degree
	info.shift = dev.get(shift)
	info.rate  = dev.get(rate)
	return info

def write_heat_info(fname, heat_info):
	np.savetxt(fname, heat_info, fmt="%s %8.3f %8.3f %8.3f %15.7e")

# I keep repeating all this boilerplate. Should figure out
# a good abstraction eventually

comm       = mpi.COMM_WORLD
mpi.install_abort_hook()
dtype      = np.float32
# Set up our device
dev = device.get_device(args.device)
# Set up our logging
verbosity  = args.verbose-args.quiet
L = logging.Logger(dev, id=comm.rank, level=verbosity, fmt="{id:3d} {t:10.6f} {mem:10.6f} {dmem_pools:10.6f} {dmem_rest:10.6f} {dmem_unknown:10.6f} {msg:s}").setdefault()
bench.set_verbose(verbosity >= 3)
L.print("Init", level=0, id=0, color=colors.lgreen)
# Benchmarking
bench.set_tfun(dev.time)
# Set up our data loader
loader  = loading.Loader(args.context, type=args.loader, dev=dev)
obsinfo = loader.query(args.query)
if len(obsinfo) == 0:
	L.print("No tods selected. Quitting", level=0, id=0, color=colors.red)
	sys.exit(1)
# Define obs-groups
joint   = loader.group_obs(obsinfo, mode=args.joint)
joint   = gutils.time_split(obsinfo, joint, maxsize=args.split*1e9, maxdur=args.tsplit)
# group selection. Sadly this can't be done with the query-level selection, as that
# happens before grouping.
jinds   = eval("list(range(%d))[%s]" % (len(joint.groups),args.sel))
joint   = gutils.select_groups(joint, jinds)
ngroup  = len(joint.groups)
if joint.joint:
	bandmsg = ",".join(joint.bands)
	if len(joint.nullbands) > 0:
		bandmsg += " null " + ",".join(joint.nullbands)
	L.print("Mapping %d tod-groups across %s with %d mpi tasks" % (ngroup, bandmsg, comm.size), level=0, id=0, color=colors.lgreen)
else:
	L.print("Mapping %d tods with %d mpi tasks" % (ngroup, comm.size), level=0, id=0, color=colors.lgreen)
prefix = args.odir + "/"
if args.prefix: prefix += args.prefix + "_"
utils.mkdir(args.odir)
# Write out our arguments
if comm.rank == 0:
	with open(prefix + "args.txt", "w") as ofile:
		ofile.write(" ".join(sys.argv) + "\n")

# Set up exception types we will ignore
if   args.ignore == "all":     etypes, load_catch = (Exception,), "all"
elif args.ignore == "missing": etypes, load_catch = (utils.DataMissing,), "expected"
elif args.ignore == "recover": etypes, load_catch = (utils.DataMissing, gutils.RecoverableError), "all"
elif args.ignore == "none":    etypes, load_catch = (), "none"
else: raise ValueError("Unrecognized error ignore setting '%s'" % str(ignore))

if args.prealloc:
	ntot_max = np.max(gutils.obs_group_size(obsinfo, joint.groups, sampranges=joint.sampranges))
	setup_buffers(dev, ntot_max)

for ind in range(comm.rank, ngroup, comm.size):
	name   = joint.names[ind]
	oname  = name.replace(":","_")
	subids = obsinfo.id[joint.groups[ind]]
	t1     = time.time()
	try:
		data  = loader.load_multi(subids, samprange=joint.sampranges[ind], catch=load_catch)
	except etypes as e:
		L.print("Skipped %d %s: %s" % (ind, name, str(e)), level=2, color=colors.red)
		continue
	if len(data.errors) > 0:
		# Partial skip
		L.print("Skipped parts %s" % str(data.errors[-1]), level=2, color=colors.red)
	t2    = time.time()

	# Whew! Finally ready to actually do our measurement
	heat_info = analyse_heating(data, dev=dev)
	t3    = time.time()
	write_heat_info("%s%s_heating.txt" % (prefix, oname), heat_info)
	t4    = time.time()

	dev.garbage_collect()
	del data
	t4    = time.time()
	L.print("Processed %d %s in %6.3f. Rd %6.3f An %6.3f Wr %6.3f" % (ind, name, t4-t1, t2-t1, t3-t2, t4-t3), level=2)
