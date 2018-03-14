import numpy as np, argparse, os, time, sys
from astropy.io import fits
from astropy import table
from enlib import enmap, utils, powspec, jointmap, bunch, mpi, memory
from scipy import interpolate, ndimage
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("sel",  nargs="?", default=None)
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("-s", "--signals",   type=str,   default="ptsrc,sz")
parser.add_argument("-t", "--tsize",     type=int,   default=360)
parser.add_argument("-p", "--pad",       type=int,   default=60)
parser.add_argument("-a", "--apod-edge", type=int,   default=30)
parser.add_argument("-S", "--nsigma",    type=float, default=4)
parser.add_argument("-l", "--lmin",      type=float, default=1000)
parser.add_argument("--debug-tile",      type=str,   default=None)
parser.add_argument("--nmax",            type=int,   default=None)
parser.add_argument("-v", "--verbose",               action="count", default=3)
parser.add_argument("-q", "--quiet",                 action="count", default=0)
parser.add_argument("-c", "--cont",                  action="store_true")
parser.add_argument("-P", "--npass",     type=int,   default=3)
parser.add_argument("--output-full-model",           action="store_true")
args = parser.parse_args()

config  = jointmap.read_config(args.config)
mapinfo = jointmap.Mapset(config, args.sel)
tsize   = args.tsize # pixels
pad     = args.pad   # pixels
dtype   = np.float64
ncomp   = 1
comm    = mpi.COMM_WORLD
signals = args.signals.split(",")
verbosity = args.verbose - args.quiet
highpass_alpha = 3
utils.mkdir(args.odir)

debug_tile = None if args.debug_tile is None else [int(w) for w in args.debug_tile.split(",")]

# Get the set of bounding boxes, after normalizing them
boxes  = np.sort(np.array([d.box for d in mapinfo.datasets]),-2)

# Read the cmb power spectrum, which is an effective noise
# component. T-only
cl_path = os.path.join(os.path.dirname(args.config),config.cl_background)
cl_bg   = powspec.read_spectrum(cl_path)[0,0]

# Read our mask. High-resolution and per-dataset masks would be handled otherwise.
# This approach only works for low-resolution masks
if   config.mask_mode == "none": mask = None
elif config.mask_mode == "lowres":
	mask_path = os.path.join(os.path.dirname(args.config), config.mask)
	mask = enmap.read_map(mask_path).preflat[0]
else: raise ValueError("Unrecognized mask mode '%s'" % config.mask_mode)

def overlaps_any(box, refboxes):
	box = np.sort(box, 0)
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

#def make_dummy_tile(mapinfo, box, signals):
#	# This is pretty hacky...
#	dset = mapinfo.datasets[0]
#	pbox = jointmap.calc_pbox(dset.shape, dset.wcs, box)
#	pbox[0] -= pad
#	pbox[1] += pad
#	psize = pbox[1]-pbox[0]
#	ffpad = np.array([fft.fft_len(s, direction="above")-s for s in psize])
#	pbox[1] += ffpad
#	# Ok, we can construct the empty tile now
#	shape, wcs      = enmap.slice_geometry(dset.shape, dset.wcs, (slice(pbox[0,0],pbox[1,0]),slice(pbox[0,1],pbox[1,1])))
#	empty_map       = enmap.zeros(shape, wcs)

def eval_tile(mapinfo, box, signals=["ptsrc","sz"], dump_dir=None, verbosity=1):
	if not overlaps_any(box, boxes): return None
	# Read the data and set up the noise model
	if verbosity >= 2: print "Reading data"
	mapset = mapinfo.read(box, pad=pad, dtype=dtype, ncomp=1, verbose=verbosity>=3)
	if mapset is None: return None
	if verbosity >= 2: print "Sanitizing"
	jointmap.sanitize_maps(mapset, apod_edge=args.apod_edge)
	jointmap.setup_mask_common_lowres(mapset, mask)
	if verbosity >= 2: print "Building noise model"
	jointmap.build_noise_model(mapset)
	if args.lmin > 0: jointmap.downweight_lowl(mapset, args.lmin, highpass_alpha)
	#if len(mapset.datasets) == 0: return None
	jointmap.setup_beams(mapset)
	jointmap.setup_background_cmb(mapset, cl_bg)

	#for i,d in enumerate(mapset.datasets):
	#	for j, s in enumerate(d.splits):
	#		enmap.write_map("test_map_%02d_%02d.fits" % (i,j), s.data.map)

	# Analyze this tile. It's best to loop manually, as that lets us
	# output maps gradually
	finder  = jointmap.SourceSZFinder3(mapset, snmin=args.nsigma, npass=args.npass)
	info    = finder.analyze(verbosity=verbosity-2)
	info.ffpad = mapset.ffpad
	return info

def output_tile(prefix, info):
	shape = info.model.shape[-2:]
	ffpad_slice = (Ellipsis,slice(0,shape[0]-info.ffpad[0]),slice(0,shape[1]-info.ffpad[1]))
	for name, snmap in info.snmaps:
		enmap.write_map(prefix + name + "_snmap.fits", snmap[ffpad_slice])
	for name, snmap in info.snresid:
		enmap.write_map(prefix + name + "_snresid.fits", snmap[ffpad_slice])
	if not args.output_full_model:
		if len(info.model) > 0: model = info.model[0]
		else: model = enmap.zeros(info.model.shape[-2:], info.model.wcs, info.model.dtype)
		enmap.write_map(prefix + "model.fits", model[ffpad_slice])
	else:
		enmap.write_map(prefix + "model.fits", info.model[ffpad_slice])
	# Output total catalogue
	apod_slice  = (slice(args.apod_edge,-args.apod_edge), slice(args.apod_edge,-args.apod_edge))
	shape, wcs = enmap.slice_geometry(info.model.shape, info.model.wcs, ffpad_slice[-2:])
	shape, wcs = enmap.slice_geometry(shape, wcs, apod_slice)
	box        = enmap.box(shape, wcs)
	jointmap.write_catalogue(prefix + "catalogue.fits", info.catalogue, box)

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
		if debug_tile is not None and not (y == debug_tile[0] and x == debug_tile[1]):
			continue
		prefix  = args.odir + "/padtile%(y)03d_%(x)03d_" % {"y":y,"x":x}
		if args.cont and os.path.isfile(prefix + "catalogue.fits"):
			if verbosity >= 1:
				print "%3d skipping %3d %3d (already done)" % (comm.rank, y, x)
			continue
		if verbosity >= 1:
			print "%3d processing %3d %3d" % (comm.rank, y, x)
		sys.stdout.flush()
		t1   = time.time()
		tpos = np.array(tyx[i])
		pbox = np.array([tpos*tshape,np.minimum((tpos+1)*tshape,shape[-2:])])
		box  = enmap.pix2sky(shape, wcs, pbox.T).T
		try:
			info = eval_tile(mapinfo, box, signals, verbosity=verbosity)
			output_tile(prefix, info)
		except (np.linalg.LinAlgError, MemoryError) as e:
			print "%3d error while processing %3d %3d: '%s'. Skipping" % (comm.rank, y, x, e.message)
			raise
			continue
		t2   = time.time()
		if verbosity >= 1:
			print "%3d processed %3d %3d in %7.1f max-mem %7.3f" % (comm.rank, y, x, t2-t1, memory.max()/1024.**3)
else:
	# Single arbitrary tile
	if not overlaps_any(bounds, boxes):
		if verbosity >= 1:
			print "No data in selected region"
	else:
		info = eval_tile(mapinfo, bounds, signals, verbosity=verbosity)
		output_tile(args.odir + "/", info)
