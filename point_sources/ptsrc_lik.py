import numpy as np, argparse, os, time
from enlib import enmap, utils, powspec, jointmap, bunch, mpi
from scipy import interpolate, ndimage
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("sel",  nargs="?", default=None)
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("-s", "--signals",   type=str,   default="ptsrc,sz")
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
signals = args.signals.split(",")
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

def spec_2d_to_1d(spec2d):
	l2d = spec2d.modlmap()
	dl  = l2d[0,1]*1.2
	lmax= np.max(l2d)
	pix = (l2d.reshape(-1)/dl).astype(int)
	spec1d = np.bincount(pix, spec2d.reshape(-1))/np.bincount(pix)
	spec1d[~np.isfinite(spec1d)] = 1e-20
	spec1d = np.maximum(spec1d, 1e-30)
	spline =interpolate.splrep(np.arange(len(spec1d))*dl,np.log(spec1d))
	return np.exp(interpolate.splev(np.arange(0, lmax), spline))


def eval_tile(mapinfo, box, signals=["ptsrc","sz"], dump_dir=None, verbose=False):
	if not overlaps_any(box, boxes): return None
	# Read the data and set up the noise model
	if verbose: print "Reading data"
	mapset = mapinfo.read(box, pad=pad, dtype=dtype, verbose=verbose)
	if mapset is None: return None
	if verbose: print "Sanitizing"
	jointmap.sanitize_maps(mapset)
	if verbose: print "Building noise model"
	jointmap.build_noise_model(mapset)
	if len(mapset.datasets) == 0: return None
	jointmap.setup_beams(mapset)
	jointmap.setup_background_cmb(mapset, cl_bg)

	# Analyze this tile
	finder = jointmap.SourceSZFinder(mapset)
	cand_classes = finder.find_candidates(10)
	for cands in cand_classes[1:]:
		#cands = bunch.Bunch(pix=[[200,500]], pos=[[1,1]], sn=[0], npix=[0], n=1)
		for ci in range(cands.n)[3:]:
			lik   = finder.get_likelihood(cands.pix[ci], cands.name)
			ml    = lik.maximize(verbose=True)
			stats = lik.explore(ml.x, verbose=True, nsamp=50)
			stats.posterior = ml.posterior
			print "Final stats"
			print lik.format_sample(stats)
			print "cand %2d had S/N %5.1f. Fid pos: %6.2f %6.2f. Assuming 10 uK noise: %6.2f" % (ci, cands.sn[ci], cands.pos[ci,0]/utils.degree, cands.pos[ci,1]/utils.degree, cands.sn[ci]*10)

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
		osuff = "_padtile%(y)03d_%(x)03d.fits" % {"y":y,"x":x}
		tags = [s.replace(":","_") for s in signals]
		nok  = sum([os.path.isfile(args.odir + "/" + tag + osuff) for tag in tags])
		if args.cont and nok == len(tags):
			print "%3d skipping %3d %3d (already done)" % (comm.rank, y, x)
			continue
		print "%3d processing %3d %3d" % (comm.rank, y, x)
		tpos = np.array(tyx[i])
		pbox = np.array([tpos*tshape,np.minimum((tpos+1)*tshape,shape[-2:])])
		box  = enmap.pix2sky(shape, wcs, pbox.T).T
		res  = eval_tile(mapinfo, box, signals, verbose=False)
		for si, tag in enumerate(tags):
			if res is not None:
				snmap = res.signals[si].snmap
			else:
				snmap = jointmap.make_dummy_tile(shape, wcs, box, pad=pad, dtype=dtype).map
			enmap.write_map(args.odir + "/" + tag + osuff, snmap)
else:
	# Single arbitrary tile
	if not overlaps_any(bounds, boxes):
		print "No data in selected region"
	else:
		res = eval_tile(mapinfo, bounds, signals, verbose=True)
		for i, sig in enumerate(res.signals):
			enmap.write_map(args.odir + "/%s_snmap.fits"  % sig.name, sig.snmap)
			enmap.write_map(args.odir + "/%s_alpha.fits"  % sig.name, sig.alpha)
			enmap.write_map(args.odir + "/%s_dalpha.fits" % sig.name, sig.dalpha)
