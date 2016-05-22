import numpy as np, os, h5py
from enlib import config, mpi, coordinates, utils, errors, tagdb
from enact import filedb, actdata, todinfo
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("tagfile")
parser.add_argument("sel", nargs="?", default="")
parser.add_argument("ofile")
args = parser.parse_args()

file_db = filedb.ACTFiles()
scan_db = tagdb.read(args.tagfile)
comm    = mpi.COMM_WORLD

scan_db = scan_db[args.sel]
ids     = scan_db.ids
stats = []
for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	entry = file_db[id]
	try:
		stats.append(todinfo.build_tod_stats(entry))
	except errors.DataMissing as e:
		print "Skipping %s (%s)" % (id, e.message)
		continue
	print id
stats = todinfo.merge_tod_stats(stats)

if comm.rank == 0: print "Reducing"
comm.Barrier()
for key in stats:
	stats[key] = utils.allgatherv(stats[key],comm)

# Sort by id and move id index last
inds = np.argsort(stats["id"])
for key in stats:
	stats[key] = utils.moveaxis(stats[key][inds],0,-1)
stat_db = todinfo.Todinfo(stats)
# Merge with original tags
stat_db += scan_db

if comm.rank == 0:
	print "Writing"
	stat_db.write(args.ofile)
	print "Done"
