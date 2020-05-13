from __future__ import division, print_function
import numpy as np, argparse, os
from enlib import enmap, utils, powspec, jointmap, bunch, mpi
from scipy import interpolate
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("sel",  nargs="?", default=None)
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("-t", "--tsize",  type=int,   default=480)
parser.add_argument("-p", "--pad",    type=int,   default=240)
parser.add_argument("-C", "--ncomp",  type=int,   default=3)
parser.add_argument("-B", "--obeam",  type=str,   default=None)
parser.add_argument("-c", "--cont",    action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-W", "--wiener",  action="store_true")
parser.add_argument("-m", "--mask",   type=str,   default=None)
parser.add_argument("--filter-mode",  type=str,   default="weight")
parser.add_argument("--cg-tol",       type=float, default=1e-4)
parser.add_argument(      "--detrend",type=int,   default=1)
args = parser.parse_args()

config  = jointmap.read_config(args.config)
mapinfo = jointmap.Mapset(config, args.sel)
tsize   = args.tsize # pixels
pad     = args.pad   # pixels
dtype   = np.float64
ncomp   = args.ncomp
comm    = mpi.COMM_WORLD
utils.mkdir(args.odir)

# Get the set of bounding boxes, after normalizing them
boxes  = np.sort(np.array([d.box for d in mapinfo.datasets]),-2)

# Read the cmb power spectrum, which is an effective noise
# component. T-only
#cl_path = os.path.join(os.path.dirname(args.config),config.cl_background)
#cl_bg   = powspec.read_spectrum(cl_path)[0,0]

def overlaps_any(box, refboxes):
	rdec, rra = utils.moveaxis(refboxes - box[0,:], 2,0)
	wdec, wra = np.abs(box[1]   - box[0])
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

def mapdiag(map):
	if map.ndim < 4: return map
	elif map.ndim == 4: enmap.samewcs(np.einsum("iiyx->iyx",map),map)
	else: raise NotImplementedError

def get_coadded_tile(mapinfo, box, obeam=None, ncomp=1, dump_dir=None, verbose=False):
	if not overlaps_any(box, boxes): return None
	mapset = mapinfo.read(box, pad=pad, dtype=dtype, verbose=verbose, ncomp=ncomp)
	if mapset is None: return None
	if all([d.insufficient for d in mapset.datasets]): return None
	jointmap.sanitize_maps(mapset, detrend=args.detrend)
	jointmap.build_noise_model(mapset)
	if len(mapset.datasets) == 0: return None
	if all([d.insufficient for d in mapset.datasets]): return None
	jointmap.setup_beams(mapset)
	jointmap.setup_target_beam(mapset, obeam)
	jointmap.setup_filter(mapset, mode=args.filter_mode)
	jointmap.setup_background_spectrum(mapset)
	mask    = jointmap.get_mask_insufficient(mapset)
	if args.wiener: coadder = jointmap.Wiener(mapset)
	else:           coadder = jointmap.Coadder(mapset)
	rhs     = coadder.calc_rhs()
	if dump_dir:
		enmap.write_map(dump_dir + "/rhs.fits", rhs)
		enmap.write_map(dump_dir + "/ps_rhs.fits", np.abs(enmap.fft(rhs.preflat[0]))**2)
		with open(dump_dir + "/names.txt", "w") as nfile:
			for name in coadder.names:
				nfile.write(name + "\n")
		ls, weights = coadder.calc_debug_weights()
		np.savetxt(dump_dir + "/weights_1d.txt", np.concatenate([
			ls[None], weights.reshape(-1, weights.shape[-1])],0).T, fmt="%15.7e")
		ls, noisespecs = coadder.calc_debug_noise()
		np.savetxt(dump_dir + "/noisespecs_1d.txt", np.concatenate([
			ls[None], noisespecs.reshape(-1, noisespecs.shape[-1])],0).T, fmt="%15.7e")
	map     = coadder.calc_map(rhs, dump_dir=dump_dir, verbose=verbose, cg_tol=args.cg_tol)#, maxiter=1)
	if dump_dir:
		enmap.write_map(dump_dir + "/ps_map.fits", np.abs(enmap.fft(mapdiag(map)))**2)
	div     = coadder.tot_div
	#C       = 1/mapset.datasets[0].iN
	res = bunch.Bunch(rhs=rhs*mask, map=map*mask, div=div*mask)#, C=C)
	#res = bunch.Bunch(rhs=rhs, map=map, div=div)#, C=C)
	return res

if args.obeam:
	try: obeam = jointmap.read_beam(("fwhm",float(args.obeam)))
	except ValueError: obeam = jointmap.read_beam(("transfun", args.obeam))
else: obeam = None

# We have two modes, depending on what args.area is.
# 1. area is an enmap. Will loop over tiles in that area, and output padded tiles
#    to output directory
# 2. area is a dec1:dec2,ra1:ra2 bounding box. Will process that area as a single
#    tile, and output it and debugging info to output directory
bounds = parse_bounds(args.area)
if bounds is None:
	# Tiled, so read geometry
	shape, wcs = jointmap.read_geometry(args.area)
	tshape = np.array([args.tsize,args.tsize])
	ntile  = np.floor((shape[-2:]+tshape-1)/tshape).astype(int)
	tyx    = [(y,x) for y in range(ntile[0]-1,-1,-1) for x in range(ntile[1])]
	for i in range(comm.rank, len(tyx), comm.size):
		y, x = tyx[i]
		ofile_map = args.odir + "/map_padtile/tile%(y)03d_%(x)03d.fits" % {"y":y,"x":x}
		ofile_div = args.odir + "/div_padtile/tile%(y)03d_%(x)03d.fits" % {"y":y,"x":x}
		utils.mkdir(os.path.dirname(ofile_map))
		utils.mkdir(os.path.dirname(ofile_div))
		if args.cont and os.path.isfile(ofile_map):
			print("%3d skipping %3d %3d (already done)" % (comm.rank, y, x))
			continue
		print("%3d processing %3d %3d" % (comm.rank, y, x))
		tpos = np.array(tyx[i])
		pbox = np.array([tpos*tshape,np.minimum((tpos+1)*tshape,shape[-2:])])
		box  = enmap.pix2sky(shape, wcs, pbox.T).T
		res  = get_coadded_tile(mapinfo, box, obeam=obeam, ncomp=args.ncomp, verbose=args.verbose)
		if res is None: res = jointmap.make_dummy_tile((args.ncomp,)+shape[-2:], wcs, box, pad=pad, dtype=dtype)
		enmap.write_map(ofile_map, res.map)
		enmap.write_map(ofile_div, res.div)
else:
	# Single arbitrary tile
	if not overlaps_any(bounds, boxes):
		print("No data in selected region")
	else:
		res = get_coadded_tile(mapinfo, bounds, obeam=obeam, ncomp=args.ncomp, dump_dir=args.odir, verbose=args.verbose)
		enmap.write_map(args.odir + "/map.fits", res.map)
		enmap.write_map(args.odir + "/div.fits", res.div)
		#enmap.write_map(args.odir + "/C.fits",   res.C)
