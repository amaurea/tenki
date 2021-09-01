from __future__ import division, print_function
import numpy as np, warnings, time, copy, argparse, os, sys, pipes, shutil, re
from enlib  import config, coordinates, mapmaking, bench, scanutils, log, sampcut, dmap
from pixell import utils, enmap, pointsrcs, bunch, mpi, fft
from enact  import filedb, actdata, actscan, files, todinfo

config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("downsample", 1, "Factor with which to downsample the TOD")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
config.default("tod_window", 5.0, "Number of samples to window the tod by on each end")
config.default("eig_limit", 0.1, "Pixel condition number below which polarization is dropped to make total intensity more stable. Should be a high value for single-tod maps to avoid thin stripes with really high noise")
config.default("map_sys",  "equ", "The coordinate system of the maps. Can be eg. 'hor', 'equ' or 'gal'.")
config.default("map_dist", False, "Whether to use distributed maps")

parser = config.ArgumentParser()
parser.add_argument("sel",  help="TOD selction query")
parser.add_argument("area", help="Geometry to map")
parser.add_argument("odir", help="Output directory")
parser.add_argument("prefix", nargs="?", help="Output file name prefix")
parser.add_argument("--dets", type=str, default=0,  help="Detector slice")
args = parser.parse_args()

utils.mkdir(args.odir)
comm  = mpi.COMM_WORLD
dtype = np.float32 if config.get("map_bits") == 32 else np.float64
ncomp = 3
tsize = 720
root  = args.odir + "/" + (args.prefix + "_" if args.prefix else "")
down  = config.get("downsample")
# Set up logging
utils.mkdir(root + ".log")
logfile   = root + ".log/log%03d.txt" % comm.rank
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, file=logfile, rank=comm.rank, shared=True)
# Set up our geometry
shape, wcs = enmap.read_map_geometry(args.area)
shape = (ncomp,)+shape[-2:]
msys = config.get("map_sys")
dist = config.get("map_dist")
# Filter parameters
filter_fknee = 0.1
filter_alpha = -3

# Get our tod list
filedb.init()
ids = todinfo.get_tods(args.sel, filedb.scans)

# Dump our settings
if comm.rank == 0:
	config.save(root + "config.txt")
	with open(root + "args.txt","w") as f:
		argstring = " ".join([pipes.quote(a) for a in sys.argv[1:]])
		f.write(argstring + "\n")
		sys.stdout.write(argstring + "\n")
		sys.stdout.flush()
	with open(root + "env.txt","w") as f:
		for k,v in os.environ.items():
			f.write("%s: %s\n" %(k,v))
	with open(root + "ids.txt","w") as f:
		for id in ids:
			f.write("%s\n" % str(id))
	shutil.copyfile(filedb.cjoin(["root","dataset","filedb"]),  root + "filedb.txt")
	try: shutil.copyfile(filedb.cjoin(["root","dataset","todinfo"]), root + "todinfo.hdf")
	except (IOError, OSError): pass

# Read in all our scans
data = scanutils.read_scans_autobalance(ids, actscan.ACTScan, comm, filedb.data, dets=args.dets,
		downsample=down, sky_local=dist, osys=msys)
if data.n == 0:
	L.info("Giving up")
	sys.exit(1)

read_ids  = [ids[ind] for ind in utils.allgatherv(data.inds, comm)]
read_ndets= utils.allgatherv([len(scan.dets) for scan in data.scans], comm)
read_nsamp= utils.allgatherv([scan.cut.size-scan.cut.sum() for scan in data.scans], comm)
read_dets = utils.uncat(utils.allgatherv(
	np.concatenate([scan.dets for scan in data.scans]) if len(data.scans) > 0 else np.zeros(0,int)
	,comm), read_ndets)
if comm.rank == 0:
	# Save accept list
	with open(root + "accept.txt", "w") as f:
		for id, dets in zip(read_ids, read_dets):
			f.write("%s %3d: " % (id, len(dets)) + " ".join([str(d) for d in dets]) + "\n")
	# Save autocuts
	if data.autocuts is not None:
		with open(root + "autocut.txt","w") as ofile:
			ofile.write(("#%29s" + " %15s"*len(data.autocuts.names)+"\n") % (("id",)+tuple(data.autocuts.names)))
			for id, acut in zip(data.autocuts.ids, data.autocuts.cuts):
				ofile.write(("%30s" + " %7.3f %7.3f"*len(data.autocuts.names) + "\n") % ((id,)+tuple(1e-6*acut.reshape(-1))))
	# Save sample stats
	with open(root + "samps.txt", "w") as ofile:
		ofile.write("#%29s %4s %9s\n" % ("id", "ndet", "nsamp"))
		for id, ndet, nsamp in zip(read_ids, read_ndets, read_nsamp):
			ofile.write("%30s %4d %9d\n" % (id, ndet, nsamp))

# Reuse our signal stuff from ML mapmaking to make working with both normal and distributed
# maps easier. This also sets up our pointing matrix.
if dist:
	geo    = dmap.geometry(shape, wcs, bbox=data.bbox, tshape=(tsize,tsize), dtype=dtype, comm=comm)
	signal = mapmaking.SignalDmap(data.scans, data.subs, dmap.zeros(geo), sys=msys, name="sky")
else:
	signal = mapmaking.SignalMap(data.scans, enmap.zeros(shape, wcs, dtype), comm, sys=msys, name="sky")

# Set up our output maps
rhs = signal.zeros()
div = signal.zeros(mat=True)
hit = signal.zeros()
rhs_work = signal.prepare(rhs)
div_work = signal.prepare(div)
hit_work = signal.prepare(hit)
# Input for div calculation
div_tmp = signal.work()

# Process all our tods
L.info("Processing %d tods" % data.n)
for fi, scan in enumerate(data.scans):
	L.debug("Processing %4d/%d %s" % (data.rinds[fi]+1, data.n, ids[data.inds[fi]]))
	tod = scan.get_samples().astype(dtype)
	# Apply our filter. Just a simple one for now.
	tod -= np.mean(tod,0)
	ft   = fft.rfft(tod)
	freq = fft.rfftfreq(scan.nsamp, 1/scan.srate)
	ft  *= (1+np.maximum(freq/filter_fknee,1e-6)**filter_alpha)**-1
	fft.irfft(ft, tod, normalize=True)
	# Measure our noise properties and apply inverse variance weight
	ivar = 1/np.var(tod, 1)
	tod *= ivar[:,None]
	# Apply our cuts and accumulate
	sampcut.gapfill_const(scan.cut, tod, 0, inplace=True)
	signal.backward(scan, tod, rhs_work)

	# Build our denominator too
	for ci in range(ncomp):
		div_tmp *= 0; div_tmp[ci] = 1
		signal.forward(scan, tod, div_tmp, tmul=0)
		tod *= ivar[:,None]
		sampcut.gapfill_const(scan.cut, tod, 0, inplace=True)
		signal.backward(scan, tod, div_work[ci])

	# and hits
	tod[:] = 1
	sampcut.gapfill_const(scan.cut, tod, 0, inplace=True)
	signal.backward(scan, tod, hit_work)

L.info("Reducing")
# Everything process. Distribute as needed
del div_tmp
signal.finish(rhs, rhs_work); del rhs_work
signal.finish(div, div_work); del div_work
signal.finish(hit, hit_work); del hit_work
hit = hit[0].astype(np.int32)

L.info("Solving")
# Invert system
idiv = signal.polinv(div)
map  = signal.polmul(idiv, rhs)
del idiv

# Write result
L.info("Writing rhs"); signal.write(root, "rhs",  rhs)
L.info("Writing div"); signal.write(root, "div",  div)
L.info("Writing hit"); signal.write(root, "hits", hit)
L.info("Writing map"); signal.write(root, "map",  map)

L.info("Done")
