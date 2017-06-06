import numpy as np, os, h5py
from enlib import utils, config, mpi, errors
from enact import filedb, actdata
parser = config.ArgumentParser(os.environ["HOME"] + "./enkirc")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("prefix", nargs="?", default=None)
parser.add_argument("-C", "--chunk-size",   type=int,   default=250)
parser.add_argument("-S", "--segment-size", type=int,   default=24000, help="Length of segments over which to compute statistics. Must be a multiple of nmed and nrms.")
parser.add_argument("-m", "--nmed",         type=int,   default=3)
parser.add_argument("-s", "--nrms",         type=str,   default="800:40:10", help="size of sub-segments on which stats are comuted directly")
parser.add_argument("-c", "--cont", action="store_true")
args = parser.parse_args()

# segsize of 900 and nrms of 3 makes us sensitive to signals on timescales
# of 3/4 s (1.25 Hz). That puts us in the 1/f part of the spectrum for most
# detectors. We want some 1/f in there to confirm that the detector really
# sees the atmosphere.
filedb.init()
ids   = filedb.scans[args.sel]
comm  = mpi.COMM_WORLD
ntod  = len(ids)
nmed       = args.nmed
nrms       = [int(w) for w in args.nrms.split(":")]
nstat      = len(nrms)
seg_size   = args.segment_size
chunk_size = args.chunk_size
nchunk= (ntod+chunk_size-1)/chunk_size
dtype = np.float32

utils.mkdir(args.odir)
prefix = args.odir + "/"
if args.prefix: prefix += args.prefix + "_"

# Read a single layout to determine max det number
layout = actdata.read_layout(filedb.data[ids[0]]).layout
ndet   = layout.ndet

# Loop over chunks of tods
for chunk in range(nchunk):
	ofile = prefix + "stats%03d.hdf" % chunk
	if args.cont and os.path.exists(ofile):
		print "Skipped chunk %2d/%d (done)" % (chunk+1,nchunk)
		continue
	if comm.rank == 0:
		print "Processing chunk %2d/%d" % (chunk+1,nchunk)
	# We can't allocate the full buffer, since we don't know the
	# length of each tod a priori
	ind1   = chunk*chunk_size
	ind2   = min(ind1+chunk_size,ntod)
	cntod  = ind2-ind1
	lens   = np.zeros(cntod,int)
	mystats= []
	myinds = []
	for ind in range(ind1+comm.rank, ind2, comm.size):
		i     = ind-ind1
		id    = ids[ind]
		entry = filedb.data[id]
		try:
			# Get completely raw tods. No cut gapfilling. No gains. Output
			# will be in ADC units.
			d = actdata.read(entry, fields=["tod"])
			if d.ndet == 0 or d.nsamp == 0: raise errors.DataMissing("empty tod")
		except (IOError, errors.DataMissing) as e:
			print "Skipped %s [%3d/%d] (%s)" % (id, i+1, ind2-ind1, e.message)
			continue
		print "Processing %s [%3d/%d]" % (id, i+1, ind2-ind1)
		# Get rid of non-data bits
		d.tod /= 128
		nsamp = d.nsamp
		# Find the number of segments in this tod. We only want whole segments
		nseg  = nsamp/seg_size
		if nseg < 1:
			print "Skipped %s: To short tod" % id
			continue
		tod   = d.tod[:,:nseg*seg_size]
		stat  = np.zeros([2,nstat,ndet,nseg],dtype=dtype)
		for si in range(nstat):
			sub  = tod.reshape(ndet,nseg,-1,nmed,nrms[si])
			rmss = np.median(np.std(sub,-1),-1)
			stat[0,si] = np.mean(rmss,-1)
			stat[1,si] = np.std(rmss,-1)
		lens[i] = nseg
		mystats.append(stat)
		myinds.append(i)
		del d, tod, sub
	# Collect everybody's lengths
	lens = utils.allreduce(lens, comm)
	offs = utils.cumsum(lens, endpoint=True)
	# Allocate output stat buffer. This is a bit inefficient, since
	# only really the root should need to do this. But the stat arrays
	# aren't that big.
	stats = np.zeros([2,nstat,ndet,offs[-1]],dtype=dtype)
	for li, gi in enumerate(myinds):
		stats[:,:,:,offs[gi]:offs[gi+1]] = mystats[li]
	del mystats
	stats = utils.allreduce(stats, comm)
	# And output
	if comm.rank == 0:
		print "Writing %s" % ofile
		with h5py.File(ofile, "w") as hfile:
			hfile["stats"]= stats
			hfile["lens"] = lens
			hfile["ids"]  = ids[ind1:ind2]
			hfile["csize"]= chunk_size
		del stats
