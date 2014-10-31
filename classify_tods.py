import numpy as np, mpi4py.MPI, os
from enlib import config, utils, coordinates, targets
from enact import filedb, files

config.default("filedb", "filedb.txt", "File describing the location of the TOD and their metadata")
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("filelist")
args = parser.parse_args()

comm  = mpi4py.MPI.COMM_WORLD

db    = filedb.ACTFiles(config.get("filedb"))
# Allow filelist to take the format filename:[slice]
toks = args.filelist.split(":")
filelist, fslice = toks[0], ":".join(toks[1:])
filelist = [line.split()[0] for line in open(filelist,"r") if line[0] != "#"]
filelist = eval("filelist"+fslice)

for ind in range(comm.rank, len(filelist), comm.size):
	id    = filelist[ind]
	entry = db[id]

	site  = files.read_site(entry.site)
	bore  = files.read_boresight(entry.tod)[0][:,::100]
	bore[0] = utils.ctime2mjd(bore[0])
	bore[1:3] = coordinates.transform("hor","equ",bore[1:3]*utils.degree,time=bore[0],site=site)

	targdb= targets.TargetDB(entry.targets)
	obj   = targdb.match(bore.T)
	name  = obj.name if obj else "misc"

	print "%s %s" % (id, name)
