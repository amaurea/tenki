import numpy as np, os
from enlib import config, mpi, utils, enmap, fastweight
from enact import filedb, actdata
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("sel")
parser.add_argument("template")
parser.add_argument("omap")
parser.add_argument("--daz",        type=float, default=0.7)
parser.add_argument("--azdown",     type=int,   default=100)
parser.add_argument("--chunk-size", type=int,   default=100)
parser.add_argument("--rad",        type=float, default=0.7)
parser.add_argument("-W", "--weight", type=str, default="det")
args = parser.parse_args()

filedb.init()
ids = filedb.scans[args.sel]
db  = filedb.scans.select(ids)

comm  = mpi.COMM_WORLD
shape, wcs = enmap.read_map_geometry(args.template)

# We assume that site and pointing offsets are the same for all tods,
# so get them based on the first one
entry = filedb.data[ids[0]]
site  = actdata.read(entry, ["site"]).site

omap = fastweight.fastweight(shape, wcs, db, weight=args.weight, array_rad=args.rad*utils.degree,
		comm=comm, dtype=np.float64, daz=args.daz*utils.degree, chunk_size=args.chunk_size,
		site=site, verbose=True)

if comm.rank == 0:
	enmap.write_map(args.omap, omap)
