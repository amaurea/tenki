import numpy as np, argparse, time, os, zipfile, h5py, bunch
from mpi4py import MPI
from enlib import utils, fft, nmat, errors, config, bench, array_ops, pmat, enmap
from enact import filedb, todinfo, data, nmat_measure

config.default("filedb", "filedb.txt", "File describing the location of the TOD and their metadata")
config.default("todinfo", "todinfo.txt", "File describing location of the TOD id lists")
parser = config.ArgumentParser(os.environ["HOME"]+"/.enkirc")
parser.add_argument("filelist")
parser.add_argument("odir")
parser.add_argument("-m", "--model", default="jon")
parser.add_argument("-c", "--resume", action="store_true")
parser.add_argument("-s", "--spikecut", type=int, default=1)
parser.add_argument("-C", "--covtest", action="store_true")
parser.add_argument("--imap",       type=str,             help="Reproject this map instead of using the real TOD data. Format eqsys:filename")
args = parser.parse_args()

comm = MPI.COMM_WORLD
myid, nproc = comm.rank, comm.size
model = args.model
shared = True

utils.mkdir(args.odir)

db       = filedb.ACTFiles(config.get("filedb"))
filelist = todinfo.get_tods(args.filelist, config.get("todinfo"))
myinds   = range(len(filelist))[myid::nproc]
n        = len(filelist)

# Optinal input map to subtract
imap = None
if args.imap:
	toks = args.imap.split(":")
	imap_sys, fname = ":".join(toks[:-1]), toks[-1]
	imap = bunch.Bunch(sys=imap_sys or None, map=enmap.read_map(fname))

for i in myinds:
	id    = filelist[i]
	entry = db[id]
	ofile = "%s/%s.hdf" % (args.odir, id)
	if os.path.isfile(ofile) and args.resume: continue
	t=[]; t.append(time.time())
	try:
		fields = ["gain","tconst","cut","tod","boresight", "noise_cut"]
		if args.spikecut: fields.append("spikes")
		if args.imap: fields += ["polangle","point_offsets","site"]
		d  = data.read(entry, fields)                ; t.append(time.time())
		d  = data.calibrate(d)                       ; t.append(time.time())
		if args.imap:
			# Make a full scan object, so we can perform pointing projection
			# operations
			d.noise = None
			scan = data.ACTScan(entry, d=d)
			imap.map = imap.map.astype(d.tod.dtype, copy=False)
			pmap = pmat.PmatMap(scan, imap.map, sys=imap.sys)
			# Subtract input map from tod inplace
			pmap.forward(d.tod, imap.map, tmul=1, mmul=-1)
			utils.deslope(d.tod, w=8, inplace=True)
		ft = fft.rfft(d.tod) * d.tod.shape[1]**-0.5  ; t.append(time.time())
		spikes = d.spikes[:2].T if args.spikecut else None
		if model == "old":
			noise = nmat_measure.detvecs_old(ft, d.srate, d.dets)
		elif model == "jon":
			di = np.where(d.dets==20)[0]
			noise = nmat_measure.detvecs_jon(ft, d.srate, d.dets, shared, cut_bins=spikes)
		elif model == "simple":
			noise = nmat_measure.detvecs_simple(ft, d.srate, d.dets)
		elif model == "joint":
			noise = nmat_measure.detvecs_joint(ft, d.srate, d.dets, cut_bins=spikes)
		t.append(time.time())
		with h5py.File("%s/%s.hdf" % (args.odir, id),"w") as hfile:
			nmat.write_nmat(hfile, noise)                ; t.append(time.time())
			if args.covtest:
				# Measure full cov per bin
				ndet = ft.shape[0]
				bins = np.minimum((noise.bins*ft.shape[1]/noise.bins[-1,1]).astype(int),ft.shape[1]-1)
				nbin = len(bins)
				cov_full = np.zeros([nbin,ndet,ndet])
				for bi, b in enumerate(bins):
					print "A", bi, b, np.mean(np.abs(ft[0,b[0]:b[1]])**2)**0.5/20
					cov_full[bi]  = nmat_measure.measure_cov(ft[:,b[0]:b[1]])
				cov_model= noise.covs
				# Compute total noise and correlated noise per detector for the two
				pow_full_tot  = np.einsum("bii->bi",cov_full)
				pow_model_tot = np.einsum("bii->bi",cov_model)
				pow_full_ucorr = 1/np.einsum("bii->bi",array_ops.eigpow(cov_full,-1))
				pow_model_ucorr = 1/np.einsum("bii->bi",array_ops.eigpow(cov_model,-1))
				corr_full = cov_full/(pow_full_tot[:,:,None]*pow_full_tot[:,None,:])**0.5
				corr_model = cov_model/(pow_model_tot[:,:,None]*pow_model_tot[:,None,:])**0.5
				# And write
				hfile["full/corr"]  = corr_full
				hfile["full/tpow"] = pow_full_tot
				hfile["full/upow"] = pow_full_ucorr
				hfile["model/corr"]  = corr_model
				hfile["model/tpow"] = pow_model_tot
				hfile["model/upow"] = pow_model_ucorr
		t = np.array(t)
		dt= t[1:]-t[:-1]
	except (errors.DataMissing, ValueError, AssertionError, np.linalg.LinAlgError) as e:
		print "%3d/%d %25s skip (%s)" % (i+1,n,id, e.message)
		#print entry
		#raise
		continue
	except zipfile.BadZipfile:
		print "%d/%d %25s bad zip" % (i+1,n,id)
		continue
	print ("%3d/%d %25s" + " %6.3f"*len(dt)) % tuple([i+1,n,id]+list(dt))
