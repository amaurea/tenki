import numpy as np, os, h5py
from enlib import utils, mpi, config, errors, fft, array_ops
from enact import filedb, actdata
parser = config.ArgumentParser(os.environ["HOME"]+"/.enkirc")
parser.add_argument("sel")
parser.add_argument("ofile")
parser.add_argument("--freqs", type=str, default="0,1,10,200")
parser.add_argument("--nbins", type=str, default="20,18,2")
parser.add_argument("--delay", type=int, default=0)
args = parser.parse_args()

filedb.init()
comm = mpi.COMM_WORLD
dtype = np.float32
delay = args.delay
# Group into ar1+ar2+... groups
ids = filedb.scans[args.sel]
times = np.array([float(id[:id.index(".")]) for id in ids])
labels = utils.label_unique(times, rtol=0, atol=10)
nlabel = np.max(labels)+1
# We want to be consistent with how many tods are
# grouped together, so measure the largest group,
# and ignore any smaller group
nsub = np.max(np.bincount(labels))

# Define our bins
freqs = np.array([float(f) for f in args.freqs.split(",")])
nbins = np.array([int(n)   for n in args.nbins.split(",")])
fbins = []
for i in range(len(nbins)):
	subfreqs = np.linspace(freqs[i],freqs[i+1],nbins[i]+1,endpoint=True)
	fbins.append(np.array([subfreqs[:-1],subfreqs[1:]]).T)
fbins = np.concatenate(fbins,0)
nbin  = len(fbins)
del freqs, nbins

corr = None
pos  = None

for ind in range(comm.rank, nlabel, comm.size):
	id = ids[labels==ind]
	order = np.argsort([i[-1] for i in id])
	id = id[order]
	if len(id) != nsub: continue
	print "Processing " + "+".join(id)
	entries = [filedb.data[i] for i in id]
	try:
		d = actdata.read(entries)
		d = actdata.calibrate(d, exclude=["autocut"])
	except errors.DataMissing as e:
		print "Skipping %s: %s" % (id,str(e))
		continue
	tod = d.tod.astype(dtype)
	del d.tod
	ft = fft.rfft(tod)
	del tod

	if corr is None:
		ndet = d.array_info.ndet
		corr = np.zeros((nbin,ndet,ndet),dtype=dtype)
		hits = np.zeros((nbin,ndet,ndet),dtype=int)
		var  = np.zeros((nbin,ndet))
		pos  = np.zeros((ndet,2))
		doff = actdata.read(entries, fields=["point_offsets"])
		pos[doff.dets] = doff.point_template

	# Fourier-sample bins
	ibins = (fbins/(d.srate/2)*ft.shape[1]).astype(int)
	
	# Measure each bin
	for bi, bin in enumerate(ibins):
		ft_b = ft[:,bin[0]:bin[1]]
		norm = np.mean(np.abs(ft_b))
		ft_b /= norm
		nsamp = bin[1]-bin[0]-delay
		if ft_b.size == 0: continue
		cov_b  = array_ops.measure_cov(ft_b, delay)
		var_b  = np.diag(cov_b)
		corr_b = cov_b / var_b[:,None]**0.5 / var_b[None,:]**0.5
		mask = np.isfinite(corr_b)
		corr_b[~mask] = 0
		for di, det in enumerate(d.dets):
			corr[bi,det,d.dets] += corr_b[di]*nsamp
			hits[bi,det,d.dets] += mask[di]*nsamp
			var[bi,d.dets] += var_b*nsamp*norm
	del ft

if comm.rank == 0: print "Reducing"
corr = utils.allreduce(corr, comm)
print "B", np.sum(corr**2)
hits = utils.allreduce(hits, comm)
var  = utils.allreduce(var,  comm)

if comm.rank == 0:
	# Reduce to hit subset
	mask = np.diag(np.sum(hits,0))>0
	print np.sum(mask)
	corr = corr[:,mask][:,:,mask]
	print "C", np.sum(corr**2)
	hits = hits[:,mask][:,:,mask]
	var  = var[:,mask]
	pos  = pos[mask]
	dets = np.where(mask)
	# Normalize
	corr /= hits
	var  /= np.einsum("iaa->ia", hits)
	print "Writing to %s" % args.ofile
	with h5py.File(args.ofile, "w") as hfile:
		hfile["corr"] = corr
		hfile["var"]  = var
		hfile["hits"] = hits
		hfile["pos"]  = pos
		hfile["dets"] = dets
		hfile["bins"] = fbins
	print "Done"
