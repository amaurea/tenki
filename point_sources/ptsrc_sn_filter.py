import numpy as np, argparse, os
from enlib import enmap, utils, powspec, jointmap, bunch, mpi
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("sel",  nargs="?", default=None)
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("-m", "--mode",   type=str,   default="pstrc")
parser.add_argument("-s", "--scale",  type=float, default=1.0)
parser.add_argument("-t", "--tsize",  type=int,   default=360)
parser.add_argument("-p", "--pad",    type=int,   default=60)
parser.add_argument("-c", "--cont",   action="store_true")
args = parser.parse_args()

config  = jointmap.read_config(args.config)
mapinfo = jointmap.Mapset(config, args.sel)
tsize   = args.tsize # pixels
pad     = args.pad   # pixels
dtype   = np.float64
ncomp   = 1
comm    = mpi.COMM_WORLD
utils.mkdir(args.odir)

# Get the set of bounding boxes, after normalizing them
boxes  = np.sort(np.array([d.box for d in mapinfo.datasets]),-2)

# Read the cmb power spectrum, which is an effective noise
# component. T-only
cl_path = os.path.join(os.path.dirname(args.config),config.cl_background)
cl_bg   = powspec.read_spectrum(cl_path)[0,0]

def overlaps_any(box, refboxes):
	rdec, rra = utils.moveaxis(refboxes - box[0,:], 2,0)
	wdec, wra = box[1]   - box[0]
	rra -= np.floor(rra[:,0,None]/(2*np.pi)+0.5)*(2*np.pi)
	for i in range(-1,2):
		nra = rra + i*(2*np.pi)
		if np.any((nra[:,1]>0)&(nra[:,0]<wra)&(rdec[:,1]>0)&(rdec[:,0]<wdec)): return True
	return False

def parse_bounds(bstr):
	res  = []
	toks = bstr.strip().split(",")
	if len(toks) != 2: return None
	for tok in toks:
		sub = tok.split(":")
		if len(sub) != 2: return None
		res.append([float(s)*utils.degree for s in sub])
	return np.array(res).T

def get_filtered_tile(mapinfo, box, mode="ptsrc", scale=1.0, dump_dir=None, verbose=False):
	if not overlaps_any(box, boxes): return None
	mapset = mapinfo.read(box, pad=pad, dtype=dtype, verbose=verbose)
	if mapset is None: return None
	jointmap.sanitize_maps(mapset)
	jointmap.build_noise_model(mapset)
	if len(mapset.datasets) == 0: return None
	jointmap.setup_background_cmb(mapset, cl_bg)
	jointmap.setup_beams(mapset)
	if   mode == "ptsrc":
		jointmap.setup_profiles_ptsrc(mapset)
	elif mode == "sz":
		jointmap.setup_profiles_sz(mapset, scale)
	else: raise ValueError("Unknown filter mode '%s'" % mode)

	signal_filter = jointmap.SignalFilter(mapset)
	rhs     = signal_filter.calc_rhs()
	mu      = signal_filter.calc_mu(rhs, dump_dir=dump_dir, verbose=verbose)
	alpha   = signal_filter.calc_alpha(mu)
	dalpha  = signal_filter.calc_dalpha_empirical(alpha)
	with utils.nowarn(): snmap = alpha/dalpha
	snmap[~np.isfinite(snmap)] = 0
	res = bunch.Bunch(snmap=snmap, alpha=alpha, dalpha=dalpha, mu=mu, rhs=rhs)
	return res


# We have two modes, depending on what args.area is.
# 1. area is an enmap. Will loop over tiles in that area, and output padded tiles
#    to output directory
# 2. area is a dec1:dec2,ra1:ra2 bounding box. Will process that area as a single
#    tile, and output it and debugging info to output directory
bounds = parse_bounds(args.area)
if bounds is None:
	# Tiled, so read geometry
	shape, wcs = jointmap.read_geometry(args.area)
	shape  = shape[-2:]
	tshape = np.array([args.tsize,args.tsize])
	ntile  = np.floor((shape[-2:]+tshape-1)/tshape).astype(int)
	tyx    = [(y,x) for y in range(ntile[0]-1,-1,-1) for x in range(ntile[1])]
	for i in range(comm.rank, len(tyx), comm.size):
		y, x = tyx[i]
		ofile = args.odir + "/padtile%(y)03d_%(x)03d.fits" % {"y":y,"x":x}
		if args.cont and os.path.isfile(ofile):
			print "%3d skipping %3d %3d (already done)" % (comm.rank, y, x)
			continue
		print "%3d processing %3d %3d" % (comm.rank, y, x)
		tpos = np.array(tyx[i])
		pbox = np.array([tpos*tshape,np.minimum((tpos+1)*tshape,shape[-2:])])
		box  = enmap.pix2sky(shape, wcs, pbox.T).T
		res  = get_filtered_tile(mapinfo, box, args.mode, args.scale, verbose=False)
		omap = res.snmap if res is not None else jointmap.make_dummy_tile(shape, wcs, box, pad=pad, dtype=dtype).map
		enmap.write_map(ofile, omap)
else:
	# Single arbitrary tile
	if not overlaps_any(bounds, boxes):
		print "No data in selected region"
	else:
		res = get_filtered_tile(mapinfo, bounds, args.mode, args.scale, args.odir, verbose=True)
		enmap.write_map(args.odir + "/snmap.fits", res.snmap)
		enmap.write_map(args.odir + "/alpha.fits", res.alpha)
		enmap.write_map(args.odir + "/dalpha.fits", res.dalpha)
