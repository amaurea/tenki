import numpy as np, argparse, os
from enlib import enmap, utils, powspec, jointmap, bunch, mpi
from scipy import interpolate
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
	acorr_harm = signal_filter.calc_alpha_cov_harmonic()
	acorr   = jointmap.map_ifft(acorr_harm+0j)
	acorr  /= acorr[0,0]
	filter_harm = signal_filter.calc_filter_harmonic()
	filter = jointmap.map_ifft(filter_harm+0j)
	filter/= filter[0,0]
	sims = signal_filter.sim()
	sims2= signal_filter.sim2()
	r = np.arange(0, 300, 0.1)
	sz_r  = np.array([r,jointmap.sz_rad_projected(r, scale)])
	Q  = enmap.samewcs(signal_filter.Q[0]*signal_filter.B[0],signal_filter.m[0])
	n  = Q.shape[-1]
	l  = Q.modlmap()[0,:n/2]
	y  = Q[0,:n/2]
	y /= y[0]
	l_out  = np.arange(np.max(l))
	spline =interpolate.splrep(l,np.log(y))
	sz_l   = np.exp(interpolate.splev(l_out, spline))
	#sz_l  = jointmap.calc_profile_sz(150.0, scale)
	#sz_l /= sz_l[0]
	tmp   = enmap.samewcs(jointmap.map_ifft(signal_filter.Q[0]+0j),signal_filter.m[0])
	tmp  /= tmp[0,0]
	#enmap.write_map("test.fits", tmp)
	# Noise spectrum
	ispec2d = enmap.zeros(signal_filter.shape, signal_filter.wcs, signal_filter.dtype)
	for i in range(signal_filter.nmap):
		ispec2d += signal_filter.iN[i]*np.mean(signal_filter.H[i])**2
	spec1d = 1/spec_2d_to_1d(ispec2d)
	cmb1d  = spec_2d_to_1d(signal_filter.mapset.S)
	alpha_simple, Ntot = signal_filter.calc_alpha_simple()
	dalpha_simple = signal_filter.calc_dalpha_empirical(alpha)
	snmap_simple  = alpha_simple/dalpha_simple
	ntot_simple   = spec_2d_to_1d(Ntot)

	sz_r2 = np.array([tmp.posmap()[1,0], tmp[0]])
	sz_r2[0] /= utils.arcmin
	sz_r2[0] -= sz_r2[0,0]
	sz_r2[0]  = np.abs(sz_r2[0])
	sz_r2[1] /= sz_r2[1,0]

	with utils.nowarn(): snmap = alpha/dalpha
	snmap[~np.isfinite(snmap)] = 0
	res = bunch.Bunch(snmap=snmap, alpha=alpha, dalpha=dalpha, mu=mu, rhs=rhs, acorr=acorr, filter=filter, iNs=[d.iN for d in mapset.datasets], sims=sims, sims2=sims2, sz_r=sz_r, sz_r2=sz_r2, sz_l=sz_l, nspec=spec1d, cmb=cmb1d, snmap_simple=snmap_simple, ntot_simple=ntot_simple)
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
		enmap.write_map(args.odir + "/snmap.fits",  res.snmap)
		enmap.write_map(args.odir + "/alpha.fits",  res.alpha)
		enmap.write_map(args.odir + "/dalpha.fits", res.dalpha)
		enmap.write_map(args.odir + "/acorr.fits",  np.fft.fftshift(res.acorr))
		enmap.write_map(args.odir + "/filter.fits",  np.fft.fftshift(res.filter))
		for i, iN in enumerate(res.iNs):
			enmap.write_map(args.odir + "/iN_%02d.fits" % i, np.fft.fftshift(iN))
		for i, sim in enumerate(res.sims):
			enmap.write_map(args.odir + "/sim_%02d.fits" % i, sim)
		for i, sim in enumerate(res.sims2):
			enmap.write_map(args.odir + "/sim2_%02d.fits" % i, sim)
		np.savetxt(args.odir + "/sz_r.txt", res.sz_r.T, fmt="%15.7e")
		np.savetxt(args.odir + "/sz_l.txt", res.sz_l, fmt="%15.7e")
		np.savetxt(args.odir + "/sz_r2.txt", res.sz_r2.T, fmt="%15.7e")
		np.savetxt(args.odir + "/nspec.txt", res.nspec)
		np.savetxt(args.odir + "/cmb.txt", res.cmb)
		np.savetxt(args.odir + "/ntot_simple.txt", res.ntot_simple)
		enmap.write_map(args.odir + "/snmap_simple.fits", res.snmap_simple)
