import numpy as np, os, scipy.ndimage
from enlib import config, utils, rangelist, mpi
from enact import filedb, actdata
parser = config.ArgumentParser(os.environ["HOME"]+"/.enkirc")
parser.add_argument("sel")
parser.add_argument("odir")
args = parser.parse_args()

filedb.init()
ids  = filedb.scans[args.sel]
comm = mpi.COMM_WORLD

utils.mkdir(args.odir)

def find_jumps(tod, bsize=100, nsigma=10, margin=50, step=50, margin_step=1000):
	ndet = tod.shape[0]
	cuts = []
	for det, det_tod in enumerate(tod):
		n = len(det_tod)
		# Compute difference tod
		dtod = det_tod[1:]-det_tod[:-1]
		nsamp= dtod.size
		# Find typical standard deviation
		nblock = int(nsamp/bsize)
		sigma = utils.medmean(np.var(dtod[:nblock*bsize].reshape(nblock,bsize),-1))**0.5
		# Look for samples that deviate too much from 0
		bad = np.abs(dtod) > sigma*nsigma
		bad = np.concatenate([bad[:1],bad])
		# Look for steps, areas where the mean level changes dramatically on each
		# side of the jump. First find the center of each bad region
		steps = bad*0
		labels, nlabel = scipy.ndimage.label(bad)
		centers = np.array(scipy.ndimage.center_of_mass(bad, labels, np.arange(nlabel+1))).astype(int)[:,0]
		# Find mean to the left and right of each bad region
		for i, pos in enumerate(centers):
			m1 = np.mean(det_tod[max(0,pos-step*3/2):max(1,pos-step/2)])
			m2 = np.mean(det_tod[min(n-2,pos+step/2):min(n-1,pos+step*3/2)])
			if np.abs(m2-m1) > sigma*nsigma:
				steps[pos] = 1
		#print centers.shape, np.sum(steps)
		# Grow each cut by a margin
		bad = scipy.ndimage.distance_transform_edt(1-bad) <= margin
		steps = scipy.ndimage.distance_transform_edt(1-steps) <= margin_step
		cuts.append(rangelist.Rangelist(bad|steps))
	return rangelist.Multirange(cuts)

def det_mask_to_cuts(mask, nsamp):
	res   = rangelist.Multirange.empty(len(mask), nsamp)
	for di, bad in enumerate(mask):
		if bad:
			res.data[di] = rangelist.Rangelist.ones(nsamp)
	return res

def find_null(tod):
	return np.all(tod==tod[:,0,None],1)

def measure_quant(tod):
	res = np.zeros(tod.shape[0], dtype=int)
	for i in range(len(tod)):
		res[i] = len(np.unique(tod[i]))
	return res

def write_cuts(ofile, cuts, array_info, dets):
	with open(ofile, "w") as f:
		f.write("""format = 'TODCuts'
format_version = 2
n_det = %d
n_samp = %d
samp_offset = 0\n""" % (array_info.ndet, cuts.shape[1]))
		for ind, det in enumerate(dets):
			rstr = " ".join(["(%d,%d)" % tuple(rn) for rn in cuts[ind].ranges])
			f.write("%4d: %s\n" % (det, rstr))

for ind in range(comm.rank, len(ids), comm.size):
	id = ids[ind]
	print id
	entry = filedb.data[id]
	# Read uncalibrated data
	d = actdata.read(entry, fields=["tod","array_info"], verbose=True)
	dmask = np.zeros(d.ndet, int)
	quant = measure_quant(d.tod)
	dmask |= quant < 1000
	cuts  = det_mask_to_cuts(dmask, d.nsamp)
	#cuts_null = find_null(d.tod)
	#cuts_jump = find_jumps(d.tod)
	#cuts = cuts_null # cuts_jump + cuts_null
	# Write cut file
	dets = np.arange(cuts.shape[0])
	write_cuts(args.odir + "/%s.cuts" % id, cuts, d.array_info, dets)
