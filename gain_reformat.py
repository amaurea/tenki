import numpy as np, argparse, glob, h5py, os, re, sys, shutil
from enlib import utils, sampcut, flagrange, mpi
from enact import files
parser = argparse.ArgumentParser()
parser.add_argument("iglobs", nargs="+")
parser.add_argument("ofile")
args = parser.parse_args()

comm = mpi.COMM_WORLD

ifiles, ids = [], []
for iglob in args.iglobs:
	for fname in glob.glob(iglob):
		base = os.path.basename(fname)
		m = re.match(r"(\d\d\d\d\d\d\d\d\d\d\.\d\d\d\d\d\d\d\d\d\d.ar\d)\.cal", base)
		if not m: continue
		ifiles.append(fname)
		ids.append(m.group(1))
ids  = np.asarray(ids)
inds = np.argsort(ids)
# Apply the sort
ids  = ids[inds]
ifiles = [ifiles[ind] for ind in inds]
# Look for duplicates
dups = np.where(ids[1:]==ids[:-1])[0]
if len(dups) > 0:
	# Get the first example
	if comm.rank == 0:
		print "Duplicate ids in input: " + ", ".join([ifiles[dups[0]],ifiles[dups[1]]])
	comm.Finalize()
	sys.exit(1)

nfile = len(ifiles)
# Make tmp directory for output
tmpdir  = args.ofile + ".tmp"
tmpfmt  = tmpdir + "/cut%03d.hdf"
utils.mkdir(tmpdir)

# Process each of our blocks
with h5py.File(tmpfmt % comm.rank, "w") as hfile:
	for j, i in enumerate(range(comm.rank*nfile/comm.size, (comm.rank+1)*nfile/comm.size)):
		ifile, id = ifiles[i], ids[i]
		progress = min(comm.rank + j*comm.size, nfile-1)
		print "%5d/%d %5.1f%% %s" % (progress+1, nfile, 100.0*(progress+1)/nfile, id)
		dets, gain = files.read_gain(ifile)
		dtype = [("det_uid","i"),("cal","f")]
		res = np.zeros(len(dets),dtype)
		res["det_uid"] = dets
		res["cal"] = gain
		hfile[id] = res

# Then concatenate them into the final file
comm.Barrier()
if comm.rank == 0:
	print "Reducing"
	i = 0
	with h5py.File(args.ofile, "w") as ohdf:
		for rank in range(comm.size):
			tmpfile = tmpfmt % rank
			with h5py.File(tmpfile, "r") as ihdf:
				for key in ihdf.keys():
					ihdf.copy(key, ohdf)
					print "%5d/%d %5.1f%% %s" % (i+1, nfile, 100.0*(i+1)/nfile, id)
					i += 1
	print "Done"
	# Remove our tmp files
	shutil.rmtree(tmpdir)
