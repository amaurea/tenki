import numpy as np
from enlib import config, utils, fastweight, mpi, enmap
from enact import filedb, actdata

parser = config.ArgumentParser()
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("-n", "--nsplit",type=int,   default=4)
parser.add_argument("-R", "--rad",   type=float, default=0.7)
parser.add_argument("-r", "--res",   type=float, default=0.5)
parser.add_argument("-b", "--block", type=str,   default="day")
args = parser.parse_args()

filedb.init()
ids    = filedb.scans[args.sel]
db     = filedb.scans.select(ids)
ntod   = len(db)
nsplit = args.nsplit
comm   = mpi.COMM_WORLD

# We assume that site and pointing offsets are the same for all tods,
# so get them based on the first one
entry = filedb.data[ids[0]]
site  = actdata.read(entry, ["site"]).site

# Determine the bounding box of our selected data
bounds    = db.data["bounds"].reshape(2,-1).copy()
bounds[0] = utils.rewind(bounds[0], bounds[0,0], 360)
box = utils.widen_box(utils.bounding_box(bounds.T), args.rad, relative=False)
waz, wel = box[1]-box[0]
# Use fullsky horizontally if we wrap too far
if waz <= 180:
	shape, wcs = enmap.geometry(pos=box[:,::-1]*utils.degree, res=args.res*utils.degree, proj="car", ref=(0,0))
else:
	shape, wcs = enmap.fullsky_geometry(res=args.res*utils.degree)
	y1, y2 = np.sort(enmap.sky2pix(shape, wcs, [box[:,0]*utils.degree,[0,0]])[:,0].astype(int))
	shape, wcs = enmap.slice_geometry(shape, wcs, (slice(y1,y2),slice(None)))

# Split the data into our building blocks
toks = args.block.split(":")
block_mode = toks[0]
block_size = int(toks[1]) if len(toks) > 1 else 1

if   block_mode == "tod": bid = np.arange(ntod)
elif block_mode == "day": bid = db.data["jon"]
else: raise ValueError("Unrecognized block mode '%s'" % block_mode)
bid = (bid//block_size).astype(int)
u, inds = np.unique(bid, return_inverse=True)
nblock  = len(u)
print "nblock", nblock
block_inds = [[] for i in range(nblock)]
for i, ind in enumerate(inds):
	block_inds[ind].append(i)

# Get the hitmap for each block
hits = enmap.zeros((nblock,)+shape, wcs)
for bi, binds in enumerate(block_inds):
	hits[bi] = fastweight.fastweight(shape, wcs, db.select(binds), array_rad=args.rad*utils.degree, site=site)

# Find the typical hit level
ref    = np.median(hits[hits>0])
# Find the number of blocks each pixel is hit by
nb_hits= np.sum(hits>ref*0.2,0)
nb_ref = np.median(nb_hits[nb_hits>0])
# We will only try to optimize regions that are hit a
# usable fraction of the time
mask   = (nb_hits > nb_ref*0.2)&(nb_hits>2*nsplit)

enmap.write_map("nb_hits.fits", nb_hits)
enmap.write_map("mask.fits", mask.astype(np.float32))

# Perform the split
split_hits = enmap.zeros((nsplit,)+shape, wcs)




#
#shape, wcs = enmap.read_map_geometry(args.template)
#
## We assume that site and pointing offsets are the same for all tods,
## so get them based on the first one
#entry = filedb.data[ids[0]]
#site  = actdata.read(entry, ["site"]).site
#
#omap = fastweight.fastweight(shape, wcs, db, weight=args.weight, array_rad=args.rad*utils.degree,
#		comm=comm, dtype=np.float64, daz=args.daz*utils.degree, chunk_size=args.chunk_size,
#		site=site, verbose=True)
#
#if comm.rank == 0:
#	enmap.write_map(args.omap, omap)
