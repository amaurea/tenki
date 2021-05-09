from enlib import config
parser = config.ArgumentParser()
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("-n", "--nsplit",type=int,   default=4)
parser.add_argument("-R", "--rad",   type=float, default=0.7)
parser.add_argument("-r", "--res",   type=float, default=0.5)
parser.add_argument("-b", "--block", type=str,   default="day")
parser.add_argument("-O", "--nopt",  type=int,   default=2000)
parser.add_argument("-m", "--mode",  type=str,   default="crosslink")
parser.add_argument("-w", "--weight",type=str,   default="plain")
parser.add_argument("--opt-mode",    type=str,   default="linear")
parser.add_argument("--constraint",  type=str,   default=None)
args = parser.parse_args()
import numpy as np, sys, glob, re, os
from enlib import utils
with utils.nowarn():
	from enlib import fastweight, enmap
	from enact import filedb, actdata

filedb.init()
ids    = filedb.scans[args.sel]
db     = filedb.scans.select(ids)
ntod   = len(db)
nsplit = args.nsplit
nopt   = args.nopt if nsplit > 1 else 0
optimize_subsets = (args.mode == "crosslink" or args.mode=="scanpat")
utils.mkdir(args.odir)

# Determine which arrays we have. We can't process arrays independently,
# as they in principle have correlated noise. But we also want to distinguish
# between them
pre, _, anames = np.char.rpartition(ids,".").T
if args.mode == "crosslink":
	# Treat rising and setting as separate arrays"
	rise = utils.rewind(db.data["baz"],0,360) > 0
	anames[rise]  = np.char.add(anames[rise], "r")
	anames[~rise] = np.char.add(anames[~rise],"s")
elif args.mode == "scanpat":
	# Treat each scanning pattern as a different array
	patterns = np.array([db.data["baz"],db.data["bel"],db.data["waz"]]).T
	pids     = utils.label_unique(patterns, axes=(1,), atol=1.0)
	npat     = np.max(pids)+1
	for pid in range(npat):
		anames[pids==pid] = np.char.add(anames[pids==pid], "p%d" % pid)

def ids2ctimes(ids): return np.char.partition(ids,".").T[0].astype(int)
def fix_aname(aname): return aname.replace("ar","pa").replace(":","_")
anames = np.array([fix_aname(aname) for aname in anames])
arrays, ais, nper = np.unique(anames, return_counts=True, return_inverse=True)
narray = len(arrays)
ctime  = ids2ctimes(pre)
sys.stderr.write("found arrays " + " ".join(arrays) + "\n")

# Get our block splitting parameters
toks = args.block.split(":")
block_mode = toks[0]
block_size = float(toks[1]) if len(toks) > 1 else 1

def calc_ndig(a): return int(np.log10(a))+1 if a > 0 else 1
def atolist(a): return ",".join(["%d" % v for v in a])
def calc_overlap_matrix(split_ids):
	flat = [[id.split(".")[0] for id in ids] for split in split_ids for ids in split]
	return [[len(set(fa)&set(fb)) for fb in flat] for fa in flat]
def format_overlap_matrix(mat):
	res = ""
	for row in mat:
		for col in row:
			res += " %4d" % col
		res += "\n"
	return res

def read_existing(dirname):
	# Read an existing split set, returning [(array,splits),(array,splits),...],
	# where splits = [ids1, ids2, ...].
	work = {}
	for fname in glob.glob(dirname + "/ids*.txt"):
		m = re.match(r"ids_([^_]+)_set(\d+)\.txt", os.path.basename(fname))
		if not m: continue
		array = m.group(1)
		split = int(m.group(2))
		ids   = np.loadtxt(fname, usecols=(0,), dtype="S")
		if array not in work: work[array] = {}
		work[array][split] = ids
	# Check that we have a consistent number of splits
	shit = np.bincount([split for key in work for split in work[key]])
	if np.any(shit != shit[0]): raise ValueError("Inconsistent number of splits in input directory")
	nsplit = len(shit)
	# Reformat to lists insted of dicts for the splits
	aset = {array:[work[array][split] for split in range(nsplit)] for array in work}
	return aset

def match_existing(aset, ctimes):
	# Given an aset as returned by read_existing, convert TOD ids to ctimes and match them against
	# our existing ctime list. Returns an [nctime] array containing the id of the input split
	# it belongs to. If any id is present in aset but not in ctimes, an IndexError is raised.
	# Unrestricted ctimes will be set to -1.
	ind_ownership = np.full(len(ctimes),-1,int)
	for array in aset:
		for split, ids in enumerate(aset[array]):
			my_ctimes = ids2ctimes(ids)
			my_inds   = utils.find(ctimes, my_ctimes)
			ind_ownership[my_inds] = split
	return ind_ownership

def get_block_ownership(ind_ownership, block_inds):
	# Given information of which split already owns which ctime index,
	# return the corresponding information at the block level. Returns
	# ValueError if a block has conflicting ownership
	block_ownership = np.full(len(block_inds),-1,int)
	for bi, ablock in enumerate(block_inds):
		bown = []
		for inds in ablock:
			bown.append(ind_ownership[inds])
		bown = np.concatenate(bown)
		# Filter out unowned markers
		bown = bown[bown>=0]
		if len(bown) == 0:
			block_ownership[bi] = -1
		else:
			if np.any(bown != bown[0]): raise ValueError("Inconsistent ownership for block %d: %s" % (bi, str(np.unique(bown))))
			block_ownership[bi] = bown[0]
	return block_ownership

# Peform the actual splitting. The goal is to get a
# [nblock,narray][{tod_index}] list defining the blocks
# (with tod_index being an index into the original db)
if   block_mode == "tod":
	# TOD as the building block
	_, bid_raw = np.unique(ctime, return_inverse=True)
elif block_mode == "day":
	bid_raw = db.data["jon"]
else: raise ValueError("Unrecognized block mode '%s'" % block_mode)
bid_raw = (bid_raw//block_size).astype(int)
# We know know which block each tod belongs to. But some of these will
# be empty, so we want to prune those, This results in the proper block id
u, bid, ucounts = np.unique(bid_raw, return_counts=True, return_inverse=True)
nblock  = len(u)
block_inds = [[[] for ai in range(narray)] for bi in range(nblock)]
block_size = np.zeros(nblock,int)
for i, bi in enumerate(bid):
	block_inds[bi][ais[i]].append(i)
	block_size[bi] += 1
block_inds = np.array(block_inds)
# Sort from biggest to smallest, to aid greedy algorithm
block_order = np.argsort(block_size)[::-1]  # [nblock]
block_inds  = block_inds[block_order]       # [nblock,narr][{tod_indices}]
block_size  = block_size[block_order]       # [nblock]

# Apply any external constraints
if args.constraint:
	aset            = read_existing(args.constraint)
	ind_ownership   = match_existing(aset, ctime)
	block_ownership = get_block_ownership(ind_ownership, block_inds)
else:
	block_ownership = np.full(len(block_inds),-1,int)
fixed_blocks = np.where(block_ownership>=0)[0]
free_blocks  = np.where(block_ownership<0)[0]
nfixed = len(fixed_blocks)
nfree  = len(free_blocks)

sys.stderr.write("splitting %d:[%s] tods into %d splits via %d blocks%s" % (
	ntod, atolist(nper), nsplit, nblock, (" with %d:%d free:fixed" % (nfree,nfixed)) if nfixed > 0 else "") + "\n")

# We assume that site and pointing offsets are the same for all tods,
# so get them based on the first one
entry = filedb.data[ids[0]]
site  = actdata.read(entry, ["site"]).site

# Determine the bounding box of our selected data
bounds    = db.data["bounds"].reshape(2,-1).copy()
bounds[0] = utils.rewind(bounds[0], bounds[0,0], 360)
box = utils.widen_box(utils.bounding_box(bounds.T), 4*args.rad, relative=False)
waz, wel = box[1]-box[0]
# Use fullsky horizontally if we wrap too far
if waz <= 180:
	shape, wcs = enmap.geometry(pos=box[:,::-1]*utils.degree, res=args.res*utils.degree, proj="car", ref=(0,0))
else:
	shape, wcs = enmap.fullsky_geometry(res=args.res*utils.degree)
	y1, y2 = np.sort(enmap.sky2pix(shape, wcs, [box[:,1]*utils.degree,[0,0]])[0].astype(int))
	shape, wcs = enmap.slice_geometry(shape, wcs, (slice(y1,y2),slice(None)))

sys.stderr.write("using %s workspace with resolution %.2f deg" % (str(shape), args.res) + "\n")

# Get the hitmap for each block
hits = enmap.zeros((nblock,narray)+shape, wcs)
ndig = calc_ndig(nblock)
sys.stderr.write("estimating hitmap for block %*d/%d" % (ndig,0,nblock))
for bi in range(nblock):
	for ai in range(narray):
		block_db = db.select(block_inds[bi,ai])
		hits[bi,ai] = fastweight.fastweight(shape, wcs, block_db, array_rad=args.rad*utils.degree, site=site, weight=args.weight)
	sys.stderr.write("%s%*d/%d" % ("\b"*(1+2*ndig),ndig,bi+1,nblock))
sys.stderr.write("\n")

# Build a mask for the region of interest per array
mask           = enmap.zeros((narray,)+shape, wcs, bool)
nblock_per_pix = enmap.zeros((narray,)+shape, wcs, np.int)
nblock_lim     = np.zeros(narray)
for ai in range(narray):
	ahits   = hits[:,ai]
	ref     = np.median(ahits[ahits>0])
	nblock_per_pix[ai] = np.sum(ahits>ref*0.2,0)
	nblock_ref         = np.median(nblock_per_pix[ai][nblock_per_pix[ai]>0])
	nblock_lim[ai]     = min(2*nsplit, nblock_ref*0.2)
	mask[ai]           = nblock_per_pix[ai] > nblock_lim[ai]
sys.stderr.write("[%s] pixels hit by at least [%s] blocks\n" % (
	atolist(np.sum(mask,(-2,-1))), atolist(nblock_lim)))

def calc_delta_score(split_hits, bhits, mask):
	# fractional improvement is (split_hits + bhits)/split_hits -1 = bhits/split_hits
	# This will often lead to division by zero. That is not catastrophic, but loses
	# the ability to distinguish between multiple cases that would all fill in empty pixels.
	# So we cap the ratio to a large number.
	with utils.nowarn():
		ratio = bhits/split_hits
		ratio[np.isnan(ratio)] = 0
		ratio = np.minimum(ratio, 1000)
	return np.sum(ratio[:,mask],-1)

# Perform the split. Can't use the traditional greedy
# bucket algorithm where one always allocates to the
# emptiest one, because there are now npix ways to be
# empty. could allocate to the one that makes the biggest
# relative increase. That way filling in holes will always
# be prioritized, while increasing already high areas counts less.
#target = np.sum(hits,0)/nsplit
split_hits   = enmap.zeros((nsplit,narray)+shape, wcs)
split_blocks = [[] for i in range(nsplit)]
split_fixed  = [np.where(block_ownership==i)[0] for i in range(nsplit)]
ndig_free  = calc_ndig(nfree)
ndig_fixed = calc_ndig(nfixed)
if nfixed > 0:
	sys.stderr.write("allocating fixed block %*d/%d" % (ndig_fixed,0,nfixed))
	for i, bi in enumerate(fixed_blocks):
		split_hits[block_ownership[bi]] += hits[bi]
		sys.stderr.write("%s%*d/%d" % ("\b"*(1+2*ndig_fixed),ndig_fixed,i+1,nfixed))
	sys.stderr.write("\n")
	sys.stderr.write("allocating free block %*d/%d" % (ndig_free,0,nfree))
else:
	sys.stderr.write("allocating block %*d/%d" % (ndig_free,0,nfree))
for i, bi in enumerate(free_blocks):
	bhits = hits[bi]
	score = calc_delta_score(split_hits, bhits, mask)
	best  = np.argmax(score)
	split_hits[best] += bhits
	split_blocks[best].append(bi)
	sys.stderr.write("%s%*d/%d" % ("\b"*(1+2*ndig_free),ndig_free,i+1,nfree))
sys.stderr.write("\n")

nswap  = 0
odig   = calc_ndig(nopt)
if   args.opt_mode == "linear":
	opt_order = np.arange(nopt)%max(1,nfree)
elif args.opt_mode == "random":
	opt_order = np.random.randint(0, nfree, nopt)
sys.stderr.write("optimizing %*d/%d [%*d]" % (odig, 0, nopt, odig, 0))
for oi, i in enumerate(opt_order):
	bi    = free_blocks[i]
	bhits = hits[bi]
	# Which split is this block currently in? This could be sped up with a set, but
	# will probably be fast enough anyway
	for scur in range(nsplit):
		if bi in split_blocks[scur]:
			break
	else: raise AssertionError("block not in any splits!")
	# Simulate removing and readding
	# Find the score decrease from removing it from this split
	split_hits[scur] = np.maximum(0, split_hits[scur] - bhits)
	split_blocks[scur].remove(bi)
	score = calc_delta_score(split_hits, bhits, mask)
	best  = np.argmax(score)
	split_hits[best] += bhits
	split_blocks[best].append(bi)
	if scur != best: nswap += 1
	sys.stderr.write("%s%*d/%d [%*d]" % ("\b"*(3*odig+4),odig,oi+1,nopt,odig,nswap))
sys.stderr.write("\n")

# Merge the fixed and free groups
split_blocks = [np.concatenate([
		np.array(split_blocks[i],int),
		np.array(split_fixed[i], int),
	]) for i in range(nsplit)]

# We now know which split each block belongs to. Use this and the block
# definitions to extract the ids that go into each split.

split_ids = [[[] for ai in range(narray)] for bi in range(nsplit)]
for si in range(nsplit):
	for ai in range(narray):
		for bi in split_blocks[si]:
			split_ids[si][ai] += list(ids[block_inds[bi,ai]])

if optimize_subsets:
	# If we are optimizing for crosslinking support we want to output information
	# both about individual scanning directions and the overall ones. We do that
	# by first handling the individual ones, and then combining them
	for i in range(nsplit):
		for ai, aname in enumerate(arrays):
			enmap.write_map(args.odir + "/hits_%s_set%d.fits" % (aname, i), split_hits[i,ai])
			enmap.write_map(args.odir + "/hits_masked_%s_set%d.fits" % (aname,i), split_hits[i,ai]*mask[ai])
			with open(args.odir + "/ids_%s_set%d.txt" % (aname,i), "w") as f:
				for id in sorted(split_ids[i][ai]):
					f.write(id + "\n")
	sys.stderr.write("split stats by rise vs. set\n")
	stats = ""
	for ai, aname in enumerate(arrays):
		stats += (" %-5s %s" % ("ntod",aname) + " ".join(["%7d" % len(split_ids[i][ai]) for i in range(nsplit)]) + "\n")
	for name, op in [("min",np.min),("max",np.max),("mean",np.mean)]:
		for ai, aname in enumerate(arrays):
			stats += (" %-5s %s" % (name,aname) + " ".join(["%7.1f" % op(split_hits[i,ai][mask[ai]]) for i in range(nsplit)]) + "\n")
	sys.stderr.write(stats)
	with open(args.odir + "/stats_rs.txt", "w") as f:
		f.write(stats)
	# Combine rising and setting versions of each array
	if   args.mode == "crosslink": patdig = 1
	elif args.mode == "scanpat":   patdig = calc_ndig(npat)+1
	arrays_comb = sorted(list(set([a[:-patdig] for a in arrays])))
	split_hits_comb = enmap.zeros((nsplit,len(arrays_comb))+shape, wcs)
	split_ids_comb = [[[] for a in arrays_comb] for bi in range(nsplit)]
	mask_comb      = enmap.zeros((len(arrays_comb),)+shape, wcs, bool)
	for an, anew in enumerate(arrays_comb):
		for ao, aold in enumerate(arrays):
			if not aold[:-patdig] == anew: continue
			for i in range(nsplit):
				split_hits_comb[i,an] += split_hits[i,ao]
				split_ids_comb[i][an] += split_ids[i][ao]
			mask_comb[an] |= mask[ao]
	arrays, split_hits, split_ids, mask = arrays_comb, split_hits_comb, split_ids_comb, mask_comb

sys.stderr.write("split stats\n")
stats = ""
for ai, aname in enumerate(arrays):
	stats += (" %-5s %s" % ("ntod",aname) + " ".join(["%7d" % len(split_ids[i][ai]) for i in range(nsplit)]) + "\n")
for name, op in [("min",np.min),("max",np.max),("mean",np.mean)]:
	for ai, aname in enumerate(arrays):
		stats += (" %-5s %s" % (name,aname) + " ".join(["%7.1f" % op(split_hits[i,ai][mask[ai]]) for i in range(nsplit)]) + "\n")
sys.stderr.write(stats)
with open(args.odir + "/stats.txt", "w") as f:
	f.write(stats)

# Print overlap matrix
sys.stderr.write("overlap matrix\n")
omat    = calc_overlap_matrix(split_ids)
overlap = format_overlap_matrix(omat)
sys.stderr.write(overlap)
with open(args.odir + "/overlap.txt", "w") as f:
	f.write(overlap)

# Get the ids that go into each split
for i in range(nsplit):
	for ai, aname in enumerate(arrays):
		with open(args.odir + "/ids_%s_set%d.txt" % (aname,i), "w") as f:
			for id in sorted(split_ids[i][ai]):
				f.write(id + "\n")
		enmap.write_map(args.odir + "/hits_%s_set%d.fits" % (aname, i), split_hits[i,ai])
		enmap.write_map(args.odir + "/hits_masked_%s_set%d.fits" % (aname,i), split_hits[i,ai]*mask[ai])
enmap.write_map(args.odir + "/nblock.fits", nblock_per_pix)
enmap.write_map(args.odir + "/mask.fits", mask.astype(np.int16))
