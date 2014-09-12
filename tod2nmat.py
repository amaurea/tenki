import numpy as np, argparse, time, os
from mpi4py import MPI
from enlib import utils, fft, nmat, errors, config
from enact import filedb, data, nmat_measure

config.default("filedb", "filedb.txt", "File describing the location of the TOD and their metadata")
parser = config.ArgumentParser(os.environ["HOME"]+"/.enkirc")
parser.add_argument("filelists", nargs="+")
parser.add_argument("odir")
parser.add_argument("-m", "--model", default="jon")
parser.add_argument("-c", "--resume", action="store_true")
args = parser.parse_args()

comm = MPI.COMM_WORLD
myid, nproc = comm.rank, comm.size
model = args.model

utils.mkdir(args.odir)

db       = filedb.ACTdb(config.get("filedb"))
filelist = [line.split()[0] for filelist in args.filelists for line in open(filelist,"r") if line[0] != "#"]
myinds   = range(len(filelist))[myid::nproc]
n        = len(filelist)

for i in myinds:
	id    = filelist[i]
	entry = db[id]
	ofile = "%s/%s.hdf" % (args.odir, id)
	if os.path.isfile(ofile) and args.resume: continue
	t=[]; t.append(time.time())
	try:
		d  = data.read(entry, ["gain","tconst","cut","tod","boresight"])   ; t.append(time.time())
		d  = data.calibrate(d)                                             ; t.append(time.time())
	except errors.DataMissing as e:
		print "%3d/%d %25s skip (%s)" % (i+1,n,id, e.message)
		continue
	ft = fft.rfft(d.tod) * d.tod.shape[1]**-0.5                          ; t.append(time.time())
	if model == "old":
		noise = nmat_measure.detvecs_old(ft, d.srate, d.dets)
	elif model == "jon":
		noise = nmat_measure.detvecs_jon(ft, d.srate, d.dets)
	elif model == "simple":
		noise = nmat_measure.detvecs_simple(ft, d.srate, d.dets)
	t.append(time.time())
	nmat.write_nmat("%s/%s.hdf" % (args.odir, id), noise)                ; t.append(time.time())
	t = np.array(t)
	dt= t[1:]-t[:-1]
	print ("%3d/%d %25s" + " %6.3f"*len(dt)) % tuple([i+1,n,id]+list(dt))
