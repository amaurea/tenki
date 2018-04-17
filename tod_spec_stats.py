# Reads a set of tods and extracts TOD noise power spectrum
# statistics for each tod.

from __future__ import division, print_function
import numpy as np, argparse, h5py, os, sys, shutil, time
from enlib import fft, utils, errors, config, mpi, colors
from enact import filedb, actdata, filters
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("sel")
parser.add_argument("ofile")
parser.add_argument("-b", "--bins", type=str, default="2e-2:2e-2:9999", help="Bin definitions. Comma-separated list of first,width,num. For example, 3:0.1:4 will result in the bins [[3.0:3.1],[3.1:3.2],[3.2:3.3],[3.3:3.4]].")
parser.add_argument("-B", "--bins-status", type=str, default="1e-1:1e-1:50,5:1:4,10:10:19", help="Bin definitions for progress prints in terminal.")
parser.add_argument("-C", "--chunk-size", type=int, default=4)
parser.add_argument("-n", "--nexample", type=int, default=4)
parser.add_argument("-s", "--seed", type=int, default=0)
args = parser.parse_args()

filedb.init()
ids   = filedb.scans[args.sel]
comm  = mpi.COMM_WORLD
ntod  = len(ids)
nex   = args.nexample
chunk_size = args.chunk_size * comm.size
np.random.seed(args.seed)

def parse_bin_freqs(desc):
	bfreq = []
	for ctok in desc.split(","):
		stoks = ctok.split(":")
		a = float(stoks[0])
		w = float(stoks[1])
		n = int(stoks[2])
		for i in range(n):
			bfreq.append([a+w*i,a+w*(i+1)])
	return np.array(bfreq)

def calc_bin_of_inds(bins, nf):
	res = np.full(nf,len(bins),int)
	for i, b in enumerate(bins):
		res[b[0]:b[1]] = i
	return res

def calc_bin_moments(ps, nbin, inds):
	res    = np.zeros([3,nbin])
	ndet   = ps.shape[0]
	perdet = np.zeros([nbin,ndet])
	# Get the mean in bins per detector
	hits   = np.maximum(1,np.bincount(inds, minlength=nbin)[:nbin])
	for i in range(ndet):
		perdet[:,i] = np.bincount(inds, ps[i], minlength=nbin)[:nbin]/hits
	for i in range(3):
		res[i] = np.sum(perdet**i,1)
	return res

def calc_bin_stats(ps, bin_freqs, nsamp, srate):
	nbin  = len(bin_freqs)
	bins  = (bin_freqs * nsamp / srate).astype(int)
	binds = calc_bin_of_inds(bins, ps.shape[1])
	return calc_bin_moments(ps, nbin, binds)

def get_token(mean, dev, moff=100**2, doff=0.02):
	"""Return a colored character based on the value and relative deviation."""
	if mean == 0: return colors.gray + "." + colors.reset
	lmean = np.log2(mean)
	lrdev = np.log2(dev/mean)
	letters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	cols  = [
			colors.gray,  colors.red,   colors.brown,   colors.green,
			colors.cyan,  colors.blue,  colors.purple,
			colors.lgray, colors.lred,  colors.lbrown,  colors.lgreen,
			colors.lcyan, colors.lblue, colors.lpurple,
			colors.white ]
	lind = max(0,min(len(letters)-1, int(lmean - np.log2(moff))))
	dind = max(0,min(len(cols)-1,  int(lrdev - np.log2(doff))))
	return cols[dind] + letters[lind] + colors.reset

bin_freqs = parse_bin_freqs(args.bins)
bin_freqs_status = parse_bin_freqs(args.bins_status)
nbin  = len(bin_freqs)
stats = np.zeros([3,ntod,nbin])
examples = np.zeros([ntod,nex,nbin])
nchunk= (ntod+chunk_size-1)//chunk_size

for chunk in range(nchunk):
	ind1 = chunk*chunk_size
	ind2 = min(ind1+chunk_size, ntod)
	for ind in range(ind1+comm.rank, ind2, comm.size):
		id    = ids[ind]
		entry = filedb.data[id]
		try:
			d     = actdata.read(entry, fields=["gain","tconst","cut","tod","boresight","apex"])
			d     = actdata.calibrate(d, exclude=["autocut"])
			if d.ndet == 0 or d.nsamp == 0: raise errors.DataMissing("empty tod")
		except (IOError, errors.DataMissing) as e:
			print("Skipped (%s)" % (e))
			continue
		# Compute the power spectrum
		nsamp = d.nsamp
		srate = d.srate
		ft    = fft.rfft(d.tod)
		del d.tod
		ps    = np.abs(ft)**2/(nsamp*srate)
		del ft
		# Choose a random set of detectors to be examples.
		iex = np.random.permutation(len(ps))[:nex]
		examples[ind] = ps[iex]
		# Want mean and dev between detectors. These can
		# be built from ps and ps**2
		stats[:,ind] = calc_bin_stats(ps, bin_freqs, nsamp, srate)
		n, a, a2 = calc_bin_stats(ps, bin_freqs_status, nsamp, srate)
		del ps
		amean = a/n
		adev  = (a2/n - amean**2)**0.5
		print(id + " " + "".join([get_token(me,de) for me,de in zip(amean,adev)]) + " %5.2f" % d.apex.pwv)
	tot_stats = utils.allreduce(stats, comm)
	tot_examples = utils.allreduce(examples, comm)
	if comm.rank == 0:
		mask = np.all(tot_stats[0] != 0, 1)
		n, a, a2 = tot_stats[:,mask]
		# Individual statistics
		amean = a/n
		adev  = (a2/n - amean**2)**0.5
		# Overall
		amean_tot = np.sum(a,0)/np.sum(n,0)
		adev_tot  = np.std(amean,0)
		with h5py.File(args.ofile, "w") as hfile:
			hfile["bins"]= bin_freqs
			hfile["ps"]  = amean
			hfile["dps"] = adev
			hfile["hits"]= n
			hfile["ps_tot"] = amean_tot
			hfile["dps_tot"] = adev_tot
			hfile["examples"] = examples[mask]
			hfile["ids"] = ids[mask]
			hfile["pwv"] = filedb.scans.select(ids[mask]).data["pwv"]
		print("Write")
