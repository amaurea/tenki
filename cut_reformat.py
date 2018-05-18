import numpy as np, argparse, glob, h5py, os, re, sys, shutil
from enlib import utils, sampcut, flagrange, mpi
from enact import files
parser = argparse.ArgumentParser()
parser.add_argument("iglobs", nargs="+")
parser.add_argument("ofile")
parser.add_argument("-m", "--mode", type=str, default="permissive")
args = parser.parse_args()

comm = mpi.COMM_WORLD

ifiles, ids = [], []
for iglob in args.iglobs:
	for fname in glob.glob(iglob):
		base = os.path.basename(fname)
		m = re.match(r"(\d\d\d\d\d\d\d\d\d\d\.\d\d\d\d\d\d\d\d\d\d.ar\d)\.cuts", base)
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

permissive = args.mode == "permissive"

# Process each of our blocks
nplus, nminus = 0,0
with h5py.File(tmpfmt % comm.rank, "w") as hfile:
	for j, i in enumerate(range(comm.rank*nfile/comm.size, (comm.rank+1)*nfile/comm.size)):
		ifile, id = ifiles[i], ids[i]
		progress = min(comm.rank + j*comm.size, nfile-1)
		dets, cuts, offset = files.read_cut(ifile, permissive=permissive)
		#_, moo, _ = files.read_cut(ifile, permissive=not permissive)
		#diff = (cuts.size-cuts.sum())-(moo.size-moo.sum())
		#if diff > 0: nplus  += diff
		#else:        nminus -= diff
		#print "%5d/%d %5.1f%% %s %6dM %6dM %6dM" % (progress+1, nfile, 100.0*(progress+1)/nfile, id, diff/1e6, nplus/1e6, nminus/1e6)
		print "%5d/%d %5.1f%% %s" % (progress+1, nfile, 100.0*(progress+1)/nfile, id)
		flags= flagrange.from_sampcut(cuts, dets=dets, sample_offset=offset)
		flags.write(hfile, group=id)

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
