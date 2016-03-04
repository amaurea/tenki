import numpy as np, time, h5py, copy, argparse, os, sys, pipes, shutil, re
from enlib import enmap, utils, pmat, fft, config, array_ops, mapmaking, nmat, errors, mpi
from enlib import log, bench, dmap2 as dmap, coordinates, scan as enscan, rangelist, scanutils
from enlib import pointsrcs, bunch
from enlib.cg import CG
from enlib.source_model import SourceModel
from enact import actscan, nmat_measure, filedb, todinfo

config.default("map_bits", 32, "Bit-depth to use for maps and TOD")
config.default("downsample", 1, "Factor with which to downsample the TOD")
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
config.default("map_format", "fits", "File format to use when writing maps. Can be 'fits', 'fits.gz' or 'hdf'.")
config.default("tod_window", 5.0, "Number of samples to window the tod by on each end")

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("sel")
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("prefix",nargs="?")
parser.add_argument("--ndet",       type=int, default=0,  help="Max number of detectors")
args = parser.parse_args()
filedb.init()

utils.mkdir(args.odir)
root      = args.odir + "/" + (args.prefix + "_" if args.prefix else "")
log_level = log.verbosity2level(config.get("verbosity"))
dtype     = np.float32 if config.get("map_bits") == 32 else np.float64
area      = enmap.read_map(args.area)
comm      = mpi.COMM_WORLD
ids       = filedb.scans[args.sel].ids
L = log.init(level=log_level, rank=comm.rank)

# Set up our output map.
osig = enmap.zeros((1,)+area.shape[-2:], area.wcs, dtype)
odiv = osig*0
sig_all = np.zeros(len(ids))
sig_med = sig_all*0
div_all, div_med = sig_all*0, sig_med*0

# Read in all our scans
for ind in range(comm.rank, len(ids), comm.size):
	id = ids[ind]
	entry = filedb.data.query(id, multi=True)
	try:
		d = actscan.ACTScan(entry)
		if d.ndet == 0 or d.nsamp == 0:
			raise errors.DataMissing("Tod contains no valid data")
	except errors.DataMissing as e:
		L.debug("Skipped %s (%s)" % (str(id), e.message))
		continue
	L.debug("Read %s" % id)
	d = d[:,::config.get("downsample")]
	if args.ndet > 0: d = d[:args.ndet]
	# Get our samples
	tod = d.get_samples()
	tod -= np.mean(tod,1)[:,None]
	tod  = tod.astype(dtype)
	# Construct noise model for this tod
	winsize = int(config.get("tod_window")*d.srate)
	nmat.apply_window(tod, winsize)
	d.noise = d.noise.update(tod, d.srate)
	L.debug("Noise %s" % id)
	# Apply it, to get N"d
	d.noise.apply(tod)
	nmat.apply_window(tod, winsize)
	# Compute the variance per detector. If our noise model
	# were correct and our data were pure noise, this would be
	# N"<nn'>N" = N". But our noise model isn't totally accurate.
	vars = np.var(tod,1)
	# Project each detectors result on the sky
	tod[:] = vars[:,None]
	pmap = pmat.PmatMap(d, osig)
	pcut = pmat.PmatCut(d)
	junk = np.zeros(pcut.njunk, dtype=dtype)
	pcut.backward(tod, junk)
	pmap.backward(tod, osig)
	# Also do the fiducial noise model
	tod[:] = 1
	nmat.apply_window(tod, winsize)
	d.noise.white(tod)
	nmat.apply_window(tod, winsize)
	pcut.backward(tod, junk)
	pmap.backward(tod, odiv)
	# Collect some statistics
	sig_all[ind] = np.sum(vars)*d.nsamp
	sig_med[ind] = np.median(vars)*d.ndet*d.nsamp
	div_all[ind] = np.sum(tod)
	div_med[ind] = np.median(np.sum(tod,1))*d.ndet

# Collect result
osig[:] = utils.allreduce(osig, comm)
odiv[:] = utils.allreduce(odiv, comm)
sig_all = utils.allreduce(sig_all, comm)
sig_med = utils.allreduce(sig_med, comm)
div_all = utils.allreduce(div_all, comm)
div_med = utils.allreduce(div_med, comm)

if comm.rank == 0:
	enmap.write_map(root + "sig.fits", osig[0])
	enmap.write_map(root + "div.fits", odiv[0])
	with open(root + "stats.txt", "w") as f:
		for ind, id in enumerate(ids):
			f.write("%s %15.7e %15.7e %15.7e %15.7e\n" % (id, sig_all[ind], sig_med[ind], div_all[ind], div_med[ind]))
