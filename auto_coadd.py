import numpy as np, argparse, os, imp, time
from enlib import enmap, retile, utils, bunch, cg, fft, mpi
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("sel")
parser.add_argument("template", nargs="?")
parser.add_argument("odir")
# Let's use a 4x4 degree test patch with 1 degree of padding on each side and
# another 1 degree of apodization
parser.add_argument("-b", "--box",        type=str,   default="-5:-1,36:32")
parser.add_argument("-t", "--tilesize",   type=int,   default=480)
parser.add_argument("-p", "--pad",        type=int,   default=240)
parser.add_argument("-a", "--apod-val",   type=float, default=2e-1)
parser.add_argument("-A", "--apod-alpha", type=float, default=5)
parser.add_argument("-E", "--apod-edge",  type=float, default=120)
parser.add_argument(      "--kxrad",      type=float, default=20)
parser.add_argument(      "--kx-ymax-scale", type=float, default=1)
parser.add_argument(      "--highpass",   type=float, default=200)
parser.add_argument(      "--cg-tol",     type=float, default=1e-6)
parser.add_argument(      "--max-ps",     type=float, default=0)
parser.add_argument("-F", "--filter",     action="store_true")
parser.add_argument("-c", "--cont",       action="store_true")
parser.add_argument("-v", "--verbose",    action="store_true")
args = parser.parse_args()

enmap.extent_model.append("intermediate")

sel   = "&".join(["(" + w + ")" for w in utils.split_outside(args.sel, ",")])
dtype = np.float32
comm  = mpi.COMM_WORLD
utils.mkdir(args.odir)

# Set up configuration
config = imp.load_source("config", args.config)

# Set up boolean arrays for querys
all_tags = set()
for dataset in config.datasets:
	all_tags |= dataset.tags
flags = {flag: np.array([flag in dataset.tags for dataset in config.datasets],bool) for flag in all_tags}
# Extract the relevant datasets
selinds  = np.where(eval(sel, flags))[0]
datasets = [config.datasets[i] for i in selinds]

def read_geometry(fname):
	if os.path.isdir(fname):
		geo = retile.read_tileset_geometry(fname + "/tile%(y)03d_%(x)03d.fits")
		return geo.shape, geo.wcs
	else:
		return enmap.read_map_geometry(fname)

def medloop(a, nloop=6, lim=0):
	for i in range(nloop):
		mask = a>lim
		if np.sum(mask) == 0: return lim
		lim = np.median(a[mask])
	return lim

# Make all paths relative to us instead of the config file
cdir = os.path.dirname(args.config)
# In tiled input format, all input maps have the same geometry
for dataset in datasets:
	for split in dataset.splits:
		split.map = os.path.join(cdir, split.map)
		split.div = os.path.join(cdir, split.div)

# Read the geometry from all the datasets. Also make paths relative to us instead
# of the config file
for dataset in datasets:
	shape, wcs = read_geometry(dataset.splits[0].map)
	dataset.shape = shape
	dataset.wcs   = wcs

def setup_beam(params, nl=50000):
	l = np.arange(nl).astype(float)
	if params[0] == "fwhm":
		sigma = params[1]*utils.fwhm*utils.arcmin
		return -0.5*l**2*sigma**2
	elif params[0] == "transfun":
		res   = np.zeros(nl)
		bdata = np.loadtxt(params[1])[:,1]
		ndata = len(bdata)
		res[:ndata] = np.log(bdata)
		# Fit gaussian that would reach res[ndata-1] by that point. We do this
		# because we want well-defined values for every l, to allow us to divide later
		# -0.5*(ndata-1)**2*sigma**2 = res[ndata-1] => sigma**2 = -2*res[ndata-1]/(ndata-1)**2
		sigma2 = -2*res[ndata-1]/(ndata-1)**2
		res[ndata:] = -0.5*l[ndata:]**2*sigma2
		return res
	else: raise ValueError("beam type '%s' not implemented" % type)

def beam_ratio(beam1, beam2): return beam1 - beam2

def beam_size(beam):
	return np.where(beam < np.exp(-1))[0]

def eval_beam(beam, l, raw=False):
	res = utils.interpol(beam, l[None], order=1, mask_nan=False)
	if not raw: res = np.exp(res)
	return res

# Set up our beams
for dataset in datasets:
	dataset.beam = setup_beam(dataset.beam_params)
# Get the reference beam, which is the biggest beam at each l. That way
# we never try to deconvolve sub-beam scales.

ref_beam = datasets[0].beam
for dataset in datasets[1:]:
	ref_beam = np.maximum(ref_beam, dataset.beam)

## Get the reference beam
#ref_beam = np.min([dataset.beam for dataset in datasets])

def butter(f, f0, alpha):
	if f0 <= 0: return f*0+1
	with utils.nowarn():
		return 1/(1 + (np.abs(f)/f0)**alpha)

def smooth_pix(map, pixrad):
	fmap  = enmap.fft(map)
	ky = np.fft.fftfreq(map.shape[-2])
	kx = np.fft.fftfreq(map.shape[-1])
	kr2   = ky[:,None]**2+kx[None,:]**2
	fmap *= np.exp(-0.5*kr2*pixrad**2)
	map   = enmap.ifft(fmap).real
	return map

def read_map(fname, pbox, cname, write_cache=True, read_cache=True):
	if os.path.isdir(fname):
		fname = fname + "/tile%(y)03d_%(x)03d.fits"
	if read_cache and os.path.isfile(cname):
		map = enmap.read_map(cname).astype(dtype)
	else:
		map = retile.read_area(fname, pbox).astype(dtype)
		if write_cache: enmap.write_map(cname, map)
	#if map.ndim == 3: map = map[:1]
	return map

def map_fft(x): return enmap.fft(x)
def map_ifft(x): return enmap.ifft(x).real

def calc_pbox(shape, wcs, box, n=10):
	nphi = utils.nint(360/wcs.wcs.cdelt[1])
	dec = np.linspace(box[0,0],box[1,0],n)
	ra  = np.linspace(box[0,1],box[1,1],n)
	y   = enmap.sky2pix(shape, wcs, [dec,dec*0+box[0,1]])[0]
	x   = enmap.sky2pix(shape, wcs, [ra*0+box[0,0],ra])[1]
	x   = utils.unwind(x, nphi)
	pbox = np.array([
		[np.min(y),np.min(x)],
		[np.max(y),np.max(x)]])
	xm1 = np.mean(pbox[:,1])
	xm2 = utils.rewind(xm1, shape[-1]/2, nphi)
	pbox[:,1] += xm2-xm1
	pbox = utils.nint(pbox)
	return pbox

def make_dummy_tile(shape, wcs, box, pad=0):
	pbox = calc_pbox(shape, wcs, box)
	if pad:
		pbox[0] -= pad
		pbox[1] += pad
	shape2, wcs2 = enmap.slice_wcs(shape, wcs, (slice(pbox[0,0],pbox[1,0]),slice(pbox[0,1],pbox[1,1])))
	shape2 = tuple(pbox[1]-pbox[0])
	map = enmap.zeros(shape2, wcs2, dtype)
	div = enmap.zeros(shape2[-2:], wcs2, dtype)
	return bunch.Bunch(map=map, div=div)

times = np.zeros(5)

def read_data(datasets, box, odir, pad=0, verbose=False, read_cache=False,
		write_cache=False, div_max=100, div_unhit=1e-7, map_max=1e8):
	odatasets = []
	for dataset in datasets:
		dataset = dataset.copy()
		pbox = calc_pbox(dataset.shape, dataset.wcs, box)
		#pbox = np.round(enmap.sky2pix(dataset.shape, dataset.wcs, box.T).T).astype(int)
		pbox[0] -= pad
		pbox[1] += pad
		psize = pbox[1]-pbox[0]
		ffpad = np.array([fft.fft_len(s, direction="above")-s for s in psize])
		pbox[1] += ffpad

		dataset.pbox = pbox
		osplits = []
		for split in dataset.splits:
			split = split.copy()
			if verbose: print "Reading %s" % split.map
			try:
				map = read_map(split.map, pbox, odir + "/" + os.path.basename(split.map), read_cache=read_cache, write_cache=write_cache)
				div = read_map(split.div, pbox, odir + "/" + os.path.basename(split.div), read_cache=read_cache, write_cache=write_cache).preflat[0]
			except IOError: continue
			map *= dataset.gain
			div *= dataset.gain**-2
			# Sanitize div and map, so that they don't contain unreasonable
			# values. After this, the rest of the code doesn't need to worry
			# about that.
			div[~np.isfinite(div)] = 0
			map[~np.isfinite(map)] = 0
			div = np.maximum(0,np.minimum(div_max,div))
			div[div<div_unhit] = 0
			map = np.maximum(-map_max,np.minimum(map_max,map))

			if np.any(div>0): ref_val = np.mean(div[div>0])*args.apod_val
			else: ref_val = 1.0
			apod  = np.minimum(div/ref_val,1.0)**args.apod_alpha
			apod  = apod.apod(args.apod_edge)
			#opre  = odir + "/" + os.path.basename(split.map)[:-5]
			#enmap.write_map(opre + "_apod.fits", apod)
			#enmap.write_map(opre + "_amap.fits", apod*map)
			#enmap.write_map(opre + "_adiv.fits", apod*div)
			div  *= apod
			if np.all(div==0): continue
			split.data = bunch.Bunch(map=map, div=div, H=div**0.5, empty=np.all(div>0))
			osplits.append(split)
		if len(osplits) < 2: continue
		dataset.splits = osplits
		odatasets.append(dataset)
	return odatasets, ffpad

def coadd_tile_data(datasets, box, odir, ps_smoothing=10, pad=0, ref_beam=None,
		cg_tol=1e-6, dump=False, verbose=False, read_cache=False, write_cache=False,
		div_max_tol=100, div_div_tol=1e-10):
	# Load data for this box for each dataset
	datasets, ffpad = read_data(datasets, box, odir, pad=pad,
			verbose=verbose, read_cache=read_cache, write_cache=write_cache)
	# We might not find any data
	if len(datasets) == 0: return None
	# Find the smallest beam size of the datasets
	bmin = np.min([beam_size(dataset.beam) for dataset in datasets])

	# Subtract mean map from each split to get noise maps. Our noise
	# model is HNH, where H is div**0.5 and N is the mean 2d noise spectrum
	# after some smoothing
	rhs, tot_div = None, None
	tot_iN, tot_udiv = None, 0
	for dataset in datasets:
		nsplit = 0
		dset_map, dset_div = None, None
		for split in dataset.splits:
			if dset_map is None:
				dset_map = split.data.map*0
				dset_div = split.data.div*0
			dset_map += split.data.map * split.data.div
			dset_div += split.data.div
		# Form the mean map for this dataset
		dset_map[:,dset_div>0] /= dset_div[dset_div>0]
		if tot_div is None: tot_div = dset_div*0
		tot_div += dset_div
		tshape, twcs, tdtype = dset_map.shape, dset_div.wcs, dset_div.dtype
		# Then use it to build the diff maps and noise spectra
		dset_ps = None
		for split in dataset.splits:
			if split.data.empty: continue
			diff  = split.data.map - dset_map
			wdiff = diff * split.data.H
			# What is the healthy area of wdiff? Wdiff should have variance
			# 1 or above. This tells us how to upweight the power spectrum
			# to take into account missing regions of the diff map.
			ndown = 10
			wvar  = enmap.downgrade(wdiff**2,ndown)
			goodfrac = np.sum(wvar > 1e-3)/float(wvar.size)
			if goodfrac < 0.1: goodfrac = 0
			#opre  = odir + "/" + os.path.basename(split.map)[:-5]
			#enmap.write_map(opre + "_diff.fits", diff)
			#enmap.write_map(opre + "_wdiff.fits", wdiff)
			#enmap.write_map(opre + "_wvar.fits", wvar)
			ps    = np.abs(map_fft(wdiff))**2
			#enmap.write_map(opre + "_ps1.fits", ps)
			# correct for unhit areas, which can't be whitened
			#print "A", dataset.name, np.median(ps[ps>0]), medloop(ps), goodfrac
			with utils.nowarn():
				ps   /= goodfrac
			#print "B", dataset.name, np.median(ps[ps>0]), medloop(ps), goodfrac
			#enmap.write_map(opre + "_ps2.fits", ps)
			#enmap.write_map(opre + "_ps2d.fits", ps)
			if dset_ps is None: dset_ps = enmap.zeros(ps.shape, ps.wcs, ps.dtype)
			dset_ps += ps
			nsplit += 1
		if nsplit < 2: continue
		# With n splits, mean map has var 1/n, so diff has var (1-1/n) + (n-1)/n = 2*(n-1)/n
		# Hence tot-ps has var 2*(n-1)
		dset_ps /= 2*(nsplit-1)
		#enmap.write_map(opre + "_ps2d_tot.fits", dset_ps)
		dset_ps  = smooth_pix(dset_ps, ps_smoothing)
		#enmap.write_map(opre + "_ps2d_smooth.fits", dset_ps)
		if np.all(np.isfinite(dset_ps)):
			# Super-low values of the spectrum are not realistic. These appear
			# due to beam/pixel smoothing in the planck maps. This will be
			# mostly taken care of when processing the beams, as long as we don't
			# let them get too small
			dset_ps = np.maximum(dset_ps, 1e-7)
			# Optionally cap the max dset_ps, this is mostly to speed up convergence
			if args.max_ps:
				dset_ps = np.minimum(dset_ps, args.max_ps)

			# Our fourier-space inverse noise matrix is based on the inverse noise spectrum
			iN    = 1/dset_ps
			#enmap.write_map(opre + "_iN_raw.fits", iN)
		else:
			print "Setting weight of dataset %s to zero" % dataset.name
			#print np.all(np.isfinite(dset_ps)), np.all(dset_ps>0)
			iN    = enmap.zeros(dset_ps.shape, dset_ps.wcs, dset_ps.dtype)

		# Add any fourier-space masks to this
		ly, lx   = enmap.laxes(tshape, twcs)
		lr       = (ly[:,None]**2 + lx[None,:]**2)**0.5
		if dataset.highpass:
			kxmask   = butter(lx, args.kxrad,   -3)
			kxmask   = 1-(1-kxmask[None,:])*(np.abs(ly)<bmin*args.kx_ymax_scale)[:,None]
			highpass = butter(lr, args.highpass,-10)
			filter   = highpass * kxmask
			#print "filter weighting", dataset.name
			del kxmask, highpass
		else:
			filter   = 1

		if not args.filter: iN *= filter

		# We should deconvolve the relative beam from the maps,
		# but that's numerically nasty. But it can be handled
		# inversely. We want (BiNB + ...)x = (BiNB iB m + ...)
		# where iB is the beam deconvolution operation in map space.
		# Instead of actually doing that operation, we can compute two
		# inverse noise matrixes: iN_A = BiNB for the left hand
		# side and iN_b = BiN for the right hand side. That way we
		# avoid dividing by any huge numbers.

		# Add the relative beam
		iN_A = iN.copy()
		iN_b = iN.copy()
		if ref_beam is not None:
			rel_beam  = beam_ratio(dataset.beam, ref_beam)
			bspec     = eval_beam(rel_beam, lr)
			iN_A     *= bspec**2
			iN_b     *= bspec
		#moo = iN*0+filter
		#enmap.write_map(opre + "_filter.fits", moo)
		# Add filter to noise model if we're downweighting
		# rather than filtering.
		dataset.iN_A  = iN_A
		dataset.iN_b  = iN_b
		dataset.filter = filter
		#print "A", opre
		#enmap.write_map(opre + "_iN_A.fits", iN_A)
		#enmap.write_map(opre + "_iN.fits", iN)

	# Cap to avoid single crazy pixels
	tot_div  = np.maximum(tot_div,np.median(tot_div[tot_div>0])*0.01)
	tot_idiv = tot_div*0
	tot_idiv[tot_div>div_div_tol] = 1/tot_div[tot_div>div_div_tol]

	# Build the right-hand side. The right-hand side is
	# sum(HNHm)
	if rhs is None: rhs = enmap.zeros(tshape, twcs, tdtype)
	for dataset in datasets:
		i=0
		for split in dataset.splits:
			if split.data.empty: continue
			#print "MOO", dataset.name, np.max(split.data.map), np.min(split.data.map), np.max(split.data.div), np.min(split.data.div)
			w   = split.data.H*split.data.map
			fw  = map_fft(w)
			fw *= dataset.iN_b
			if args.filter: fw *= dataset.filter
			w   = map_ifft(fw)*split.data.H
			#enmap.write_map(odir + "/%s_%02d_rhs.fits" % (dataset.name, i), w)
			rhs += w
			i += 1
	del w, iN, iN_A, iN_b, filter

	# Now solve the equation
	def A(x):
		global times
		m   = enmap.samewcs(x.reshape(rhs.shape), rhs)
		res = m*0
		times[:] = 0
		ntime = 0
		for dataset in datasets:
			for split in dataset.splits:
				if split.data.empty: continue
				t = [time.time()]
				w  = split.data.H*m; t.append(time.time())
				fw = map_fft(w);     t.append(time.time())
				fw*= dataset.iN_A;   t.append(time.time())
				w  = map_ifft(fw);   t.append(time.time())
				w *= split.data.H;   t.append(time.time())
				res += w
				for i in range(1,len(t)): times[i-1] += t[i]-t[i-1]
				ntime += 1
				#w  = enmap.harm2map(dataset.iN_A*enmap.map2harm(w))
				#w *= split.data.H
				#res += w
				del w
		times /= ntime
		return res.reshape(-1)
	def M(x):
		m   = enmap.samewcs(x.reshape(rhs.shape), rhs)
		res = m * tot_idiv
		return res.reshape(-1)
	solver = cg.CG(A, rhs.reshape(-1), M=M)
	for i in range(1000):
		t1 = time.time()
		solver.step()
		t2 = time.time()
		if verbose:
			print "%5d %15.7e %5.2f: %4.2f %4.2f %4.2f %4.2f %4.2f" % (solver.i, solver.err, t2-t1, times[0], times[1], times[2], times[3], times[4]), np.std(solver.x)
		if dump and solver.i in [1,2,5,10,20,50] + range(100,10000,100):
			m = enmap.samewcs(solver.x.reshape(rhs.shape), rhs)
			enmap.write_map(odir + "/step%04d.fits" % solver.i, m)
		if solver.err < cg_tol:
			if dump:
				m = enmap.samewcs(solver.x.reshape(rhs.shape), rhs)
				enmap.write_map(odir + "/step_final.fits", m)
			break
	tot_map = enmap.samewcs(solver.x.reshape(rhs.shape), rhs)
	# Get rid of the fourier padding
	ny,nx = tot_map.shape[-2:]
	tot_map = tot_map[...,:ny-ffpad[0],:nx-ffpad[1]]
	tot_div = tot_div[...,:ny-ffpad[0],:nx-ffpad[1]]
	return bunch.Bunch(map=tot_map, div=tot_div)

if args.template is None:
	# Process a single box
	box  = np.array([[float(w) for w in fromto.split(":")] for fromto in args.box.split(",")]).T*utils.degree
	utils.mkdir(args.odir)
	res  = coadd_tile_data(datasets, box, args.odir, pad=args.pad,
			ref_beam=ref_beam, dump=True, verbose=True, read_cache=args.cont,
			write_cache=True, cg_tol=args.cg_tol)
	enmap.write_map(args.odir + "/map.fits", res.map)
	enmap.write_map(args.odir + "/div.fits", res.div)
	#enmap.write_map(args.odir + "/ips.fits", res.ips)
else:
	# We will loop over tiles in the area defined by template
	shape, wcs = read_geometry(args.template)
	pre = read_geometry(datasets[0].splits[0].map)[0][:-2]
	shape = pre + shape
	tshape = np.array([args.tilesize,args.tilesize])
	ntile  = np.floor((shape[-2:]+tshape-1)/tshape).astype(int)
	tyx = [(y,x) for y in range(ntile[0]-1,-1,-1) for x in range(ntile[1])]
	for i in range(comm.rank, len(tyx), comm.size):
		y, x = tyx[i]
		if args.cont and os.path.isfile(args.odir + "/map_padtile%(y)03d_%(x)03d.fits" % {"y":y,"x":x}): continue
		print "%3d %3d %3d" % (comm.rank, y, x)
		tpos = np.array(tyx[i])
		pbox = np.array([tpos*tshape,np.minimum((tpos+1)*tshape,shape[-2:])])
		box  = enmap.pix2sky(shape, wcs, pbox.T).T
		res  = coadd_tile_data(datasets, box, args.odir, pad=args.pad,
				ref_beam=ref_beam, dump=False, verbose=args.verbose, cg_tol=args.cg_tol)
		if res is None:
			#print "skipping dummy %d %d" % (y,x)
			#continue
			print "make dummy %d %d" % (y,x), shape, wcs
			res = make_dummy_tile(shape, wcs, box, pad=args.pad)
		enmap.write_map(args.odir + "/map_padtile%(y)03d_%(x)03d.fits" % {"y":y,"x":x}, res.map)
		enmap.write_map(args.odir + "/div_padtile%(y)03d_%(x)03d.fits" % {"y":y,"x":x}, res.div)
		#enmap.write_map(args.odir + "/ips_padtile%(y)03d_%(x)03d.fits" % {"y":y,"x":x}, res.ips)
