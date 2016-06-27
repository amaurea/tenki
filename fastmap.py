# We want to be able to run this incrementally on new data.
# A simple way to do this is to identify scanning patterns using
# their actual properties rather than some global id. That way
# no awkward renumbering or communication is needed as new patterns
# are encountered.
#
# However, labeling scanning patterns individually is risky.
# Az and el have some jitter, so so direct values can't be used.
# We need to bin. There is then some risk that the jitter will
# cross bin boundaries. I'll assume scanning patterns are usually
# defined near integer values, and so use bins centered on integers.
#
# This approach may use too much memory. Our maps are getting pretty big, and
# take about half a GB per pattern for deep56, much more for boss or other
# really big patches. For deep56 alone there are 12 scanning patterns.
# That's a lot of memory. Could avoid the problem by scanning through,
# grouping by scanning pattern and then processing each by itself.

import numpy as np, os, h5py
from enlib import config, utils, mapmaking, scanutils, mpi, log, pmat, enmap, bench, fft
from enact import filedb, actscan, actdata, nmat_measure

# We want a detector-uncorrelated noise model because we will
# assume that when combining the maps later. This lets us choose
# smaller noise bins because the number of DOF is smaller.
config.default("nmat_uncorr_type", "lin")
config.default("nmat_uncorr_nbin",  4000)
config.default("nmat_uncorr_nmin",     1)

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("sel")
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("prefix", nargs="?")
parser.add_argument("-b", "--bsize",     type=float, default=1)
parser.add_argument("-v", "--verbosity", type=int,   default=1)
parser.add_argument("-d", "--dets",      type=str,   default=None)
args = parser.parse_args()

comm  = mpi.COMM_WORLD
level = log.verbosity2level(args.verbosity)
L     = log.init(level=level, rank=comm.rank)
dtype = np.float32 if config.get("map_bits") == 32 else np.float64
root  = args.odir + "/" + (args.prefix + "_" if args.prefix else "")
ncomp = 3
nbin  = config.get("nmat_uncorr_nbin")

filedb.init()
utils.mkdir(args.odir)
area  = enmap.read_map(args.area)
area  = enmap.zeros((ncomp,)+area.shape[-2:], area.wcs, dtype)

# First loop through all the tods and find out which are in which scanning
# pattern.
L.info("Scanning tods")
mypatids = {}
ids      = filedb.scans[args.sel].ids
myinds   = range(comm.rank, len(ids), comm.size)
for ind, d in scanutils.scan_iterator(ids, myinds, actscan.ACTScan, filedb.data):
	id = ids[ind]
	# Determine which scanning pattern we have
	el  = np.mean(d.boresight[:,2])
	az1, az2 = utils.minmax(d.boresight[:,1])
	pat = np.array([el,az1,az2])/utils.degree
	pat = tuple(np.round(pat/args.bsize)*args.bsize)
	# And record it
	if pat not in mypatids:
		mypatids[pat] = []
	mypatids[pat].append(id)

# Get list of all patterns
L.info("Gathering pattern lists")
mypats = mypatids.keys()
if len(mypats) == 0: mypats = np.zeros([0,3])
pats   = utils.allgatherv(mypats, comm)
pats   = list(set(sorted([tuple(pat) for pat in pats])))
# Collect ids for each pattern
patids = {}
for pat in pats:
	pids = mypatids[pat] if pat in mypatids else []
	pids = comm.allreduce(list(pids))
	patids[pat] = pids

if comm.rank == 0:
	L.info("Found %d patterns" % len(pats))
	for i, pat in enumerate(pats):
		L.debug("%2d: el %s az %s %s" % (i, pat[0], pat[1], pat[2]))

# Now process each pattern, one by one. This
# Means we only have to keep one in memory at a time.
if comm.rank == 0:
	tot_rhs = area*0
	tot_hits = area[0]*0
for i, pat in enumerate(pats):
	L.info("Processing pattern %d" % i)
	# Loop through our indices in this pattern
	pids   = patids[pat]
	myinds = range(comm.rank, len(pids), comm.size)
	rhs    = area*0
	hits   = area[0]*0
	nscan  = 0
	srate  = 0
	speed  = 0
	site   = {}
	inspec = np.zeros(nbin)
	offset = np.zeros(2)
	for ind, d in scanutils.scan_iterator(pids, myinds, actscan.ACTScan, filedb.data,
			dets=args.dets, downsample=config.get("downsample")):
		id = pids[ind]
		with bench.mark("pbuild"):
			# Build pointing matrices
			pmap = pmat.PmatMap(d, area)
			pcut = pmat.PmatCut(d)
		with bench.mark("tod"):
			# Get tod
			tod  = d.get_samples()
			tod -= np.mean(tod,1)[:,None]
			tod  = tod.astype(dtype)
			junk = np.zeros(pcut.njunk, dtype=dtype)
		with bench.mark("nmat"):
			# Build noise model
			ft = fft.rfft(tod) * tod.shape[1]**-0.5
			nmat = nmat_measure.detvecs_simple(ft, d.srate)
			del ft
		with bench.mark("rhs"):
			# Calc rhs, accumulating into pattern total
			nmat.apply(tod)
			pcut.backward(tod, junk)
			pmap.backward(tod, rhs)
		with bench.mark("hitmap"):
			# Calc hitcount map, accumulating into pattern total
			myhits = area*0
			tod[:] = 1
			pcut.backward(tod, junk)
			pmap.backward(tod, myhits)
			myhits = myhits[0]
			hits  += myhits
		del tod
		# Get the mean noise power spectrum
		myspec = np.mean(nmat.iD,1)
		inspec+= myspec * np.sum(myhits) # weight in total avg
		bins   = nmat.bins
		srate  = d.srate
		site   = dict(d.site)
		speed  = np.median(np.abs(d.boresight[1:,1]-d.boresight[:-1,1])[::10])/utils.degree*d.srate
		offset+= np.mean(d.offsets,0)[2:0:-1]
		nscan += 1
		del myhits, d
	
	# Ok, we're done with all tods for this pattern. Collect our
	# result.
	rhs    = utils.allreduce(rhs,    comm)
	hits   = utils.allreduce(hits,   comm)
	inspec = utils.allreduce(inspec, comm)
	inspec/= np.sum(hits)
	srate  = comm.allreduce(srate, op=mpi.MAX)
	speed  = comm.allreduce(speed, op=mpi.MAX)
	site   = [w for w in comm.allgather(site) if len(w) > 0][0]
	nscan  = comm.allreduce(nscan)
	offset = utils.allreduce(offset, comm)/nscan

	# And output
	if comm.rank == 0:
		# Output our files
		proot = root + "el_%s_az_%s_%s" % tuple([str(w) for w in pat])
		enmap.write_map(proot + "_rhs.fits",  rhs)
		enmap.write_map(proot + "_hits.fits", hits)
		with h5py.File(proot + "_info.hdf","w") as hfile:
			hfile["inspec"] = inspec
			hfile["srate"]  = srate
			hfile["ids"]    = np.array(pids)
			hfile["hits"]   = hits
			hfile["pattern"]= pat
			hfile["offset"] = offset/utils.degree # el,az offset in deg
			for k,v in site.items():
				hfile["site/"+k] = v
			for k,v in hits.wcs.to_header().items():
				hfile["wcs/"+k] = v
		tot_rhs  += rhs
		tot_hits += hits
	del rhs, hits

# Output the total rhs too, in case we want to plot it
if comm.rank == 0:
	enmap.write_map(root + "tot_rhs.fits",  tot_rhs)
	enmap.write_map(root + "tot_hits.fits", tot_hits)
