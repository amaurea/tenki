import numpy as np, os
from enlib import config, mpi, utils, coordinates, enmap, bench, pmat
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
ntod  = len(ids)
daz   = args.daz*utils.arcmin
rad   = args.rad*utils.degree
csize = args.chunk_size
nchunk= (ntod+csize-1)/csize
dtype = np.float64
nt    = 8

shape, wcs = enmap.read_map_geometry(args.template)

# We assume that site and pointing offsets are the same for all tods,
# so get them based on the first one
entry = filedb.data[ids[0]]
site  = actdata.read(entry, ["site"]).site

# Get the boresight bounds for each TOD
mids   = np.array([db.data["t"],db.data["az"],db.data["el"]])
widths = np.array([db.data["dur"],db.data["waz"],db.data["wel"]])
box    = np.array([mids-widths/2,mids+widths/2])
box[:,1:] *= utils.degree
ndets  = db.data["ndet"]

# Set up our output map
omap = enmap.zeros(shape, wcs, dtype)

def upscale(pos, factor):
	if factor == 1: return pos
	pix = np.arange(pos.shape[-1]*factor)/float(factor)
	return utils.interpol(pos, pix[None], mask_nan=False, order=1)
def fixx(yx, nphi):
	yx    = np.array(yx)
	yx[1] = utils.unwind(yx[1], nphi)
	return yx

def get_pix_ranges(shape, wcs, horbox, daz, nt=4, azdown=1, ndet=1.0):
	(t1,t2),(az1,az2),el = horbox[:,0], horbox[:,1], np.mean(horbox[:,2])
	nphi = np.abs(utils.nint(360/wcs.wcs.cdelt[0]))
	# Find the pixel coordinates of first az sweep
	naz  = utils.nint(np.abs(az2-az1)/daz)/azdown
	if naz <= 0: return None, None
	ahor = np.zeros([3,naz])
	ahor[0] = utils.ctime2mjd(t1)
	ahor[1] = np.linspace(az1,az2,naz)
	ahor[2] = el
	acel    = coordinates.transform("hor","cel",ahor[1:],time=ahor[0],site=site)
	y, x1   = upscale(fixx(utils.nint(enmap.sky2pix(shape, wcs, acel[::-1])),nphi),azdown)
	# Reduce to unique y values
	_, uinds, hits = np.unique(y, return_index=True, return_counts=True)
	y, x1 = y[uinds], x1[uinds]
	# Find the pixel coordinates of time drift
	thor = np.zeros([3,nt])
	thor[0] = utils.ctime2mjd(np.linspace(t1,t2,nt))
	thor[1] = az1
	thor[2] = el
	tcel    = coordinates.transform("hor","cel",thor[1:],time=thor[0],site=site)
	_, tx   = utils.nint(fixx(enmap.sky2pix(shape, wcs, tcel[::-1]),nphi))
	x2 = x1 + tx[-1]-tx[0]
	x1, x2  = np.minimum(x1,x2), np.maximum(x1,x2)
	pix_ranges = np.concatenate([y[:,None],x1[:,None],x2[:,None]],1)
	# Weight per pixel in pix ranges. If ndet=1 this corresponds to
	# telescope time per output pixel
	weights = (t2-t1)/(naz*azdown)/(x2-x1)*ndet * hits
	return pix_ranges, weights

def add_weight(omap, pix_ranges, weights, nphi=0, method="fortran"):
	if   method == "fortran": add_weight_fortran(omap, pix_ranges, weights, nphi)
	elif method == "python":  add_weight_python (omap, pix_ranges, weights, nphi)
	else: raise ValueError
def add_weight_python(omap, pix_ranges, weights, nphi=0):
	# This function is a candidate for implementation in fortran
	for (y,x1,x2), w in zip(pix_ranges, weights):
		omap[y,max(0,x1):min(omap.shape[1],x2)] += w
def add_weight_fortran(omap, pix_ranges, weights, nphi=0):
	core = pmat.get_core(dtype)
	core.add_rows(omap.T, pix_ranges[:,0], pix_ranges[:,1:].T, weights, nphi)

enmap.extent_model = ["intermediate"]
def smooth_tophat(map, rad):
	# Will use flat sky approximation here. It's not a good approximation for
	# our big maps, but this doesn't need to be accurate anyway
	ny,nx = map.shape[-2:]
	refy, refx = ny/2,nx/2
	pos   = map.posmap()
	pos[0] -= pos[0,refy,refx]
	pos[1] -= pos[1,refy,refx]
	r2     = np.sum(pos**2,0)
	kernel = (r2 < rad**2).astype(dtype) / (np.pi*rad**2) / map.size**0.5 * map.area()
	kernel = np.roll(kernel,-refy,0)
	kernel = np.roll(kernel,-refx,1)
	res = enmap.ifft(enmap.fft(map)*np.conj(enmap.fft(kernel))).real
	return res

nphi = np.abs(utils.nint(360/wcs.wcs.cdelt[0]))
for chunk in range(comm.rank, nchunk, comm.size):
	i1 = chunk*csize
	i2 = min((chunk+1)*csize, ntod)
	# Split the hits into horizontal pixel ranges
	pix_ranges, weights = [], []
	with bench.mark("get"):
		for i in range(i1,i2):
			weight = ndets[i] if args.weight == "det" else 1000.0
			pr, w = get_pix_ranges(shape, wcs, box[:,:,i], daz, nt, azdown=args.azdown, ndet=weight)
			if pr is None: continue
			pix_ranges.append(pr)
			weights.append(w)
		if len(pix_ranges) == 0: continue
		pix_ranges = np.concatenate(pix_ranges, 0)
		weights    = np.concatenate(weights, 0)
	with bench.mark("add"):
		add_weight(omap, pix_ranges, weights, nphi)
	print "%4d %4d %7.4f %7.4f" % (chunk, comm.rank, bench.stats.get("get"), bench.stats.get("add"))

# Combine weights
omap = utils.allreduce(omap, comm)

# Change unit from seconds per pixel to seconds per square acmin
if comm.rank == 0:
	pixarea = omap.pixsizemap() / utils.arcmin**2
	omap   /= pixarea
	omap[~np.isfinite(omap)] = 0
	omap    = smooth_tophat(omap, rad)
	omap[omap<1e-3] = 0
	enmap.write_map(args.omap, omap)
