import numpy as np, os, h5py
from enlib import config, mpi, coordinates, utils, errors, tagdb
from enact import filedb, actdata, todinfo
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("tagfile")
parser.add_argument("sel", nargs="?", default="")
parser.add_argument("ofile")
args = parser.parse_args()

file_db = filedb.setup_filedb()
scan_db = todinfo.read(args.tagfile, vars={"root":config.get("root")})
comm    = mpi.COMM_WORLD

# We want to process *all* tods, not just selected ones. Could also
# achieve this by adding /all to the selector.
scan_db = scan_db.select(scan_db.query(args.sel, apply_default_query=False))
ids     = scan_db.ids

stats = []
for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	entry = file_db[id]
	try:
		stats.append(todinfo.build_tod_stats(entry))
	except (errors.DataMissing,AttributeError) as e:
		print "Skipping %s (%s)" % (id, e.message)
		continue
	print "%3d %4d/%d %5.1f%% %s" % (comm.rank, ind+1, len(ids), (ind+1)/float(len(ids))*100, id)
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
# Merge with original tags. Rightmost overrides for overridable fields.
# For tags, we get the union. This means that stat_db can't override incorrect
# tags in scan_db, just add to them.
stat_db = scan_db + stat_db

if comm.rank == 0:
	print "Writing"
	stat_db.write(args.ofile)
	print "Done"
