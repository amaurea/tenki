# -*- coding: utf-8 -*-
# In uncorrelated mode, solves the model
#  HU'N"UH m = sum r
# In correlated mode solves
#  HU'N°WN°UH m = U'WU r
# Here H is the hitcount per pattern,
# r is the rhs per pattern, U is the unskew matrix
# and N is the per detector noise spectrum,
# with ' meaning tranpose, " meaning inverse, and
# ° meaning **-0.5.
#
# W is a 2d-fourier-diagonal per-pattern weighting matrix
# representing the detector correlations. In common-mode
# dominated regions, its job is to reduce the weight of
# the atmosphere from 1 mode per det to 1 mode per array.
# Hence, in its central region it will look like 1/ndet,
# and then gradually increase towards 1 as one moves outwards.
#
#  W = 1/(1+A*cov_common)
#
# where A = (ndet-1)/max(cov_common)

import numpy as np, argparse, h5py, enlib.cg, scipy.interpolate, time
import astropy.io.fits
from enlib import enmap, fft, coordinates, utils, bunch, interpol, bench, zipper, mpi, log
parser = argparse.ArgumentParser()
parser.add_argument("infos", nargs="+")
parser.add_argument("odir")
parser.add_argument("--nmax",            type=int, default=0)
parser.add_argument("-d", "--downgrade", type=int, default=1)
parser.add_argument("-O", "--order",     type=int, default=0)
parser.add_argument("-U", "--unskew",    type=str, default="shift")
parser.add_argument("-C", "--cmode",     type=int, default=0)
args = parser.parse_args()

fft.engine = "fftw"
comm = mpi.COMM_WORLD
log_level = log.verbosity2level(1)
L = log.init(level=log_level, rank=comm.rank, shared=False)
dtype = np.float32
ref_time = 55500
beam_sigma        = 1.4*utils.arcmin*utils.fwhm
corrfun_smoothing = 5*beam_sigma

def prepare(map, hitmap=False):
	"""Prepare a map for input by cutting off one pixel along each edge,
	as out-of-bounds data accumulates there, and downgrading to the
	target resolution."""
	# Get rid of polarization for now. Remove this later.
	if map.ndim == 3: map = map[:1]
	# Cut off edge pixels
	map[...,:1,:]  = 0
	map[...,-1:,:] = 0
	map[...,:,:1]  = 0
	map[...,:,-1:] = 0
	# Downsample
	map = enmap.downgrade(map, args.downgrade)
	if hitmap: map *= args.downgrade**2
	# Pad to fourier-friendly size. Because no cropping
	# is used, this will result in the same padding for
	# all maps.
	map = map.autocrop(method="fft", value="none")
	return map

def calc_scale(nbin, samprate, speed, pixsize):
	"""Get the scaling function that takes us from fourier-pixels in
	azimuth to fourier-pixels in the tod."""
	# Easier to consider the inverse problem.
	# Given a scale D in the map, this takes a time t = D/speed
	# to cross. The corresponding frequency is f = speed/D, which
	# is in bin b=nbin*f/(samprate/2). = 2*nbin*speed/(D*samprate).
	# Our spatial frequency bin i corresponds to a spatial mode k=i/pixsize,
	# which corresponds to the scale D = 1/k = pixsize/i. Hence the total
	# translation is b = 2*nbin*speed*i/(samprate*pixsize)
	scale = nbin*speed/(samprate/2.*pixsize)
	#print "nbin", nbin, "samprate", samprate, "speed", speed, "pixsize", pixsize, "scale", scale
	return scale

def calc_az_sweep(pattern, offset, site, pad=2.0, subsample=1.0):
	"""Helper function. Given a pattern and mean focalplane offset,
	computes the shape of an azimuth sweep on the sky."""
	el1 = pattern[0] + offset[0]
	az1 = pattern[1] + offset[1] - pad
	az2 = pattern[2] + offset[1] + pad
	daz = rhs.pixshape()[0]/np.cos(el1)/subsample
	naz  = int(np.ceil((az2-az1)/daz))
	naz  = fft.fft_len(naz, "above", [2,3,5,7])
	# Simulate a single sweep at arbitrary time
	sweep_az = np.arange(naz)*daz + az1
	sweep_el = np.full(naz,el1)
	sweep_cel = coordinates.transform("hor","cel", np.array([sweep_az,sweep_el]),time=ref_time,site=site)
	# Make ra safe
	sweep_cel = utils.unwind(sweep_cel)
	return bunch.Bunch(sweep_cel=sweep_cel, sweep_hor=np.array([sweep_az,sweep_el]),
			el=el1, az1=az1, az2=az2, naz=naz, daz=daz)

def calc_offset_upos(pattern, offset_array, offset_det, site, area, U):
	"""Compute how the detectors are offset from the array midpoint
	in unskewd pixel units. All angles are in radians."""
	ref_hor = np.array([pattern[0],np.mean(pattern[1:3])]) + offset_array
	det_hor = ref_hor[:,None] + offset_det.T
	# Transform to celestial coordinates
	ref_cel = coordinates.transform("hor","cel", ref_hor[::-1], time=ref_time, site=site)[::-1]
	det_cel = coordinates.transform("hor","cel", det_hor[::-1], time=ref_time, site=site)[::-1]
	# RA is ambiguous because we don't know the time properly
	ra0 = np.mean(area.box(),0)[1]
	det_cel[1] += ra0 - ref_cel[1]
	ref_cel[1]  = ra0
	# And convert into pixels
	ref_pix = area.sky2pix(ref_cel)
	det_pix = area.sky2pix(det_cel)
	# Unskew these to get pixels in the coordinate system
	# the noise model lives in
	ref_upix = U.apply_pix(ref_pix)
	det_upix = U.apply_pix(det_pix)
	# And express that as positions in this new coordinate system
	ref_upos = enmap.pix2sky(U.ushape, U.uwcs, ref_upix)
	det_upos = enmap.pix2sky(U.ushape, U.uwcs, det_upix)
	# And finally record the offset of each detector from the reference point
	offset_upos = det_upos - ref_upos[:,None]
	return offset_upos

def calc_cmode_corrfun(ushape, uwcs, offset_upos, sigma, nsigma=10):
	"""Compute the real-space correlation function for the atmospheric
	common mode in unskewed coordinates. The result has an arbitrary
	overall scaling."""
	res    = enmap.zeros(ushape, uwcs)
	# Generate corrfun around center of map
	upos   = offset_upos + np.mean(res.box(),0)[:,None]
	# We will work on a smaller cutout to speed things up
	pad    = sigma*nsigma
	box    = np.array([np.min(upos,1)-pad,np.max(upos,1)+pad])
	pixbox = res.sky2pix(box.T).T
	work   = res[pixbox[0,0]:pixbox[1,0],pixbox[0,1]:pixbox[1,1]]
	posmap = work.posmap()
	# Generate each part of the corrfun as a gaussian in real space.
	# Could do this in fourier space, but easier to get subpixel precision this way
	# (not that that is very important, though)
	for p in upos.T:
		r2    = np.sum((posmap-p[:,None,None])**2,0)
		work += np.exp(-0.5*r2/sigma**2)
	# Convolute with itself mirrored to get the actual correlation function
	fres  = fft.rfft(res, axes=[-2,-1])
	fres *= np.conj(fres)
	fft.ifft(fres, res, axes=[-2,-1])
	res /= np.max(res)
	return res

class UnskewCurved:
	def __init__(self, shape, wcs, pattern, offset, site, pad=2.0*utils.degree, order=0, subsample=2.0):
		"""Build an unskew operator that uses spline interpolation along an azimuth sweep
		to straighten out the scanning motion for one scanning pattern. Relatively slow, and
		leads to some smoothing due to the interpolation, but does not assume that dec changes
		with a constant speed during a sweep."""
		# Find the unskew transformation for this pattern.
		# We basically want dec->az and ra->ra0, with az spacing
		# similar to el spacing.
		ndec, nra = shape[-2:]
		info = calc_az_sweep(pattern, offset, site, pad=pad, subsample=subsample)
		sweep_ra, sweep_dec = info.sweep_cel
		#(sweep_ra, sweep_dec), naz, daz = calc_az_sweep(pattern, offset, site, pad=pad, subsample=subsample)
		# We want to be able to go from (y,x) to (ra,dec), with
		# dec = dec[y]
		# ra  = ra[y]-ra[0]+x
		# Precompute the pixel mapping. This will have the full witdh in ra,
		# but will be smaller in dec due to the limited az range.
		raw_dec, raw_ra = enmap.posmap(shape, wcs)
		skew_pos = np.zeros((2, info.naz, nra))
		skew_pos[0] = sweep_dec[:,None]
		skew_pos[1] = (sweep_ra-sweep_ra[0])[:,None] + raw_ra[None,ndec/2,:]
		skew_pix = enmap.sky2pix(shape, wcs, skew_pos).astype(dtype)
		# Build geometry for the unskewed system
		ushape, uwcs = enmap.geometry(pos=[0,0], shape=[info.naz, nra], res=[np.abs(info.daz), enmap.pixshape(shape,wcs)[1]], proj="car")
		# Save
		self.order    = order
		self.shape    = shape
		self.wcs      = wcs
		self.pattern  = pattern
		self.site     = site
		self.skew_pix = skew_pix
		# External interface
		self.ushape   = ushape
		self.uwcs     = uwcs
	def apply(self, map):
		"""Apply unskew operation to map, returning an array where the scanning motion
		goes along the vertical axis."""
		omap = interpol.map_coordinates(map, self.skew_pix, order=self.order)
		return omap
	def trans(self, imap, omap):
		"""Transpose of apply. omap = U.T(imap). Omap argument specifies the shape of the result,
		but its values may be destroyed during the operation."""
		interpol.map_coordinates(omap, self.skew_pix, imap, trans=True, order=self.order)
		return omap
	def apply_pix(self, ipix):
		"""Transform from normal sky pixels to unskewd pixels."""
		return self.skew_pix.at(ipix)

class UnskewShift:
	def __init__(self, shape, wcs, pattern, offset, site, pad=2.0*utils.degree):
		"""This unskew operation assumes that equal spacing in
		dec corresponds to equal spacing in time, and that shifts in
		RA can be done in units of whole pixels. This is an approximation
		relative to UnskewCurved, but it is several times faster, uses
		less memory, and causes less smoothing."""
		ndec, nra = shape[-2:]
		info = calc_az_sweep(pattern, offset, site, pad=pad)
		sweep_ra, sweep_dec = info.sweep_cel
		# For each pixel in dec (that we hit for this scanning pattern), we
		# want to know how far we have been displaced in ra.
		# First get the dec of each pixel center.
		ysweep, xsweep = enmap.sky2pix(shape, wcs, [sweep_dec,sweep_ra])
		y1  = max(int(np.min(ysweep)),0)
		y2  = min(int(np.max(ysweep))+1,shape[-2])
		# Make fft-friendly
		ny  = y2-y1
		ny2 = fft.fft_len(ny, "above", [2,3,5,7])
		y1  = max(y1-(ny2-ny)/2,0)
		y2  = min(y1+ny2,shape[-2])
		y   = np.arange(y1,y2)
		dec, _ = enmap.pix2sky(shape, wcs, [y,y*0])
		# Then interpolate the ra values corresponding to those decs.
		# InterpolatedUnivariateSpline broken. Returns nan even when
		# interpolating. So we will use UnivariateSpline
		spline  = scipy.interpolate.UnivariateSpline(sweep_dec, sweep_ra)
		ra      = spline(dec)
		dra     = ra - ra[len(ra)/2]
		y, x    = np.round(enmap.sky2pix(shape, wcs, [dec,ra]))
		dx      = x-x[len(x)/2]
		# It's also useful to be able to go from normal map index to
		# position in y and dx
		inv_y   = np.zeros(shape[-2],dtype=int)-1
		inv_y[y.astype(int)]= np.arange(len(y))
		# Compute the azimuth step size based on the total azimuth sweep.
		daz = (pattern[2]-pattern[1]+2*pad)/len(y)
		# Build the geometry of the unskewed system
		ushape, uwcs = enmap.geometry(pos=[0,0], shape=[len(y),shape[-1]], res=[daz,enmap.pixshape(shape,wcs)[1]], proj="car")
		# And store the result
		self.y  = y.astype(int)
		self.dx = np.round(dx).astype(int)
		self.dx_raw = dx
		self.inv_y  = inv_y
		self.ushape = ushape
		self.uwcs   = uwcs
	def apply(self, map):
		"""Apply the Unskew operation. This is a simple horizontal shift of the array,
		with the complication that the shift is different for each row. This manual loop
		in python is probably quite suboptimal."""
		omap = np.zeros(map.shape[:-2]+(self.ushape[-2],map.shape[-1]), map.dtype)
		for i, y in enumerate(self.y):
			omap[...,i,:] = np.roll(map[...,y,:],-self.dx[i],-1)
		return omap
	def trans(self, imap, omap):
		"""Transpose operation: omap = U.T(imap). The original omap is
		overwritten. For this version of unskew, the transpose operation is
		just shifting things back."""
		omap[:] = 0
		for i, y in enumerate(self.y):
			omap[...,y,:] = np.roll(imap[...,i,:],self.dx[i],-1)
		return omap
	def apply_pix(self, ipix):
		"""Transform from normal sky pixels to unskewd pixels. Result is
		given in units of whole pixels, in line with the nearest-neighbor
		approximation this calss uses."""
		ypix = np.round(ipix[0]).astype(int)
		i    = self.inv_y[ypix]
		return np.array([i,ipix[1]-self.dx[i]])

class NmatUncorr:
	"""Noise matrix representing noise that is independent between
	detectors, and hence only extends in the vertical direction after
	unskewing."""
	def __init__(self, shape, inspec, scale):
		freqs = fft.rfftfreq(shape[-2]) * scale
		# Should check units of fourier space here. Should
		# rescaling from samples to az change the value of the
		# spectrum? How much of that is handled by the hitcounts?
		self.spec_full = utils.interpol(inspec, freqs[None])
		self.scale = scale
	def apply(self, arr, inplace=False):
		# Because of our padding and multiplication by the hitcount
		# before this, we should be safely apodized, and can assume
		# periodic boundaries
		if not inplace: arr = np.array(arr)
		ft = fft.rfft(arr, axes=[-2])
		ft *= self.spec_full[:,None]
		return fft.ifft(ft, arr, axes=[-2], normalize=True)

class NmatCmode:
	"""Noise matrix representing both stripy noise and detector-correlated
	noise. Generates a 2d noise spectrum as the sum of these two noise
	components, and then inverts it to get the inverse noise."""
	# The stripy noise is given by horizontally stacking the
	# noise model from NmatUncorr, though we need to be careful
	# with the scaling so we get the right dimensions.
	#
	# The detector-correlated noise is given by computing a
	# map where the detector positions are set to 1 and the rest
	# is zero, probably with a gaussian around each detector to make
	# things smoother. The 2d fft of this should give the shape of the
	# correlated noise in 2d. Then scale the whole thing such that
	# it has the same dc amplitude as the stripy noise (since both
	# are caused by the atmosphere).
	def __init__(self, shape, inspec, scale, corrfun):
		freqs = np.abs(fft.fftfreq(shape[-2]) * scale)
		spec_full = utils.interpol(inspec, np.abs(freqs[None]))
		# Build our 2d noise spectrum. First get the stripy part.
		ps_stripe = np.tile(1/spec_full[:,None], [1,shape[-1]])
		# Then get our common mode
		ps_cmode  = fft.fft(corrfun*1j,axes=[-2,-1])
		# Scale common mode to have the same DC level as the striping
		ps_cmode *= ps_stripe[0,0]/ps_cmode[0,0]
		ps_tot    = ps_stripe + ps_cmode
		self.inv_ps = 1/ps_tot
	def apply(self, arr, inplace=False):
		# Because of our padding and multiplication by the hitcount
		# before this, we should be safely apodized, and can assume
		# periodic boundaries
		if not inplace: arr = np.array(arr)
		carr = arr.astype(complex)
		ft   = fft.fft(carr, axes=[-2,-1])
		ft  *= self.inv_ps
		carr = fft.ifft(ft, carr, axes=[-2,-1], normalize=True)
		arr  = carr.real
		return arr

class NmatUncorr2:
	"""Noise matrix representing noise that is independent between
	detectors, and hence only extends in the vertical direction after
	unskewing."""
	def __init__(self, shape, inspec, scale):
		freqs = np.abs(fft.fftfreq(shape[-2]) * scale)
		self.spec_full = utils.interpol(inspec, freqs[None])
		self.scale = scale
	def apply(self, ft, inplace=False, exp=1):
		if not inplace: ft = np.array(ft)
		ft *= self.spec_full[:,None]**exp
		return ft

class WeightMat:
	def __init__(self, shape, corrfun, ndet):
		ps  = fft.fft(corrfun+0j,axes=[-2,-1]).real
		ps *= (ndet-1)/np.max(ps)
		self.weight = 1/(1+ps)
	def apply(self, ft, inplace=False):
		if not inplace: ft = np.array(ft)
		ft *= self.weight
		return ft

class Amat:
	"""Matrix representing the operaion A = HU'N"UH"""
	def __init__(self, dof, infos, comm):
		self.dof   = dof
		self.infos = infos
		self.comm  = comm
	def __call__(self, x):
		xmap = self.dof.unzip(x)
		res  = xmap*0
		for info in self.infos:
			t  = [time.time()]
			work  = xmap*info.H
			t.append(time.time())
			umap  = info.U.apply(work)
			t.append(time.time())
			fmap  = fft.fft(umap+0j, axes=[-2,-1])
			t.append(time.time())
			fmap  = info.N.apply(fmap, exp=0.5)
			t.append(time.time())
			if info.W is not None:
				fmap = info.W.apply(fmap)
			t.append(time.time())
			fmap  = info.N.apply(fmap, exp=0.5)
			t.append(time.time())
			umap  = fft.ifft(fmap, umap+0j, axes=[-2,-1], normalize=True).real
			t.append(time.time())
			work = enmap.samewcs(info.U.trans(umap, work),work)
			t.append(time.time())
			work *= info.H
			t.append(time.time())
			t = np.array(t)
			print " %4.2f"*(len(t)-1) % tuple(t[1:]-t[:-1])
			res  += work
		res = utils.allreduce(res,comm)
		return self.dof.zip(res)

def normalize_hits(hits):
	"""Normalize the hitcounts by multiplying by a factor such that
	hits_scaled**2 approx hits for most of its area. This is not
	the same thing as taking the square root. 1d sims indicate that
	this is a better approximation, but I'm not sure."""
	return hits**0.5
	medval = np.median(hits[hits!=0])
	return hits/medval**0.5

tref  = 55500
infos = []
ifiles= args.infos
rhs_tot = None
if args.nmax: ifiles = ifiles[:args.nmax]
for infofile in ifiles[comm.rank::comm.size]:
	L.info("Reading %s" % (infofile))
	with h5py.File(infofile, "r") as hfile:
		rhs     = hfile["rhs"].value
		hits    = hfile["hits"].value
		srate   = hfile["srate"].value
		speed   = hfile["speed"].value   * utils.degree
		inspec  = hfile["inspec"].value
		offsets = hfile["offsets"].value * utils.degree
		site    = bunch.Bunch(**{k:hfile["site"][k].value for k in hfile["site"]})
		pattern = hfile["pattern"].value * utils.degree
		hwcs    = hfile["wcs"]
		header = astropy.io.fits.Header()
		for key in hwcs:
			header[key] = hwcs[key].value
		wcs = enlib.wcs.WCS(header).sub(2)
	# Set up our maps
	rhs  = enmap.ndmap(rhs,  wcs)
	hits = enmap.ndmap(hits, wcs)
	rhs  = prepare(rhs)
	hits = prepare(hits, hitmap=True)

	# Turn offsets into an average array offset and detector offsets relative to that,
	# and use the array offset to set up the Unskew matrix.
	offset_array = np.mean(offsets,0)
	offset_det   = offsets - offset_array
	ndet         = len(offsets)
	if args.unskew == "curved":
		U = UnskewCurved(rhs.shape, rhs.wcs, pattern, offset_array, site, order=args.order)
	elif args.unskew == "shift":
		U = UnskewShift(rhs.shape, rhs.wcs, pattern, offset_array, site)
	else: raise ValueError(args.unskew)
	scale = calc_scale(inspec.size, srate, speed, enmap.pixshape(U.ushape, U.uwcs)[0])

	# Set up the inv noise matrix N. This should take a 2d ft of the
	# map as input, though it actually only cares about the y direction.
	N = NmatUncorr2(U.ushape, inspec, scale)

	# Set up the weight matrix W
	if args.cmode > 0:
		offset_upos = calc_offset_upos(pattern, offset_array, offset_det, site, rhs, U)
		corrfun     = calc_cmode_corrfun(U.ushape, U.uwcs, offset_upos, corrfun_smoothing)
		W  = WeightMat(U.ushape, corrfun, 4)#ndet)
	else: W = None

	# The H in our equation is related to the hitcount, but isn't exactly it.
	# normalize_hits approximates it using the hitcounts.
	H = normalize_hits(hits)

	# Apply weight to rhs
	if W is not None:
		iH  = 1/np.maximum(H,np.max(H)*1e-2)
		urhs= U.apply(rhs*iH)
		ft  = fft.fft(urhs+0j, axes=[-2,-1])
		ft  = W.apply(ft)
		urhs= fft.ifft(ft, urhs+0j, axes=[-2,-1], normalize=True).real
		rhs = U.trans(urhs, rhs)*H
	
	if rhs_tot is None: rhs_tot = rhs
	else: rhs_tot += rhs

	infos.append(bunch.Bunch(U=U,N=N,H=H,W=W,pattern=pattern,site=site,srate=srate,scale=scale,speed=speed))

rhs = utils.allreduce(rhs_tot, comm)

#info = infos[0]
#foo  = rhs*info.H
#enmap.write_map("test1.fits", foo)
#bar  = enmap.samewcs(info.U.apply(foo),foo)
#enmap.write_map("test2.fits", bar)
#foo  = enmap.samewcs(info.U.trans(bar, foo),foo)
#enmap.write_map("test3.fits", foo)
#1/0
#
#

dof = zipper.ArrayZipper(rhs.copy())
A   = Amat(dof, infos, comm)
cg  = enlib.cg.CG(A, dof.zip(rhs))

utils.mkdir(args.odir)
for i in range(200):
	cg.step()
	if comm.rank == 0:
		#print np.std(cg.x[cg.x!=0])
		if cg.i % 10 == 0:
			map = dof.unzip(cg.x)
			enmap.write_map(args.odir + "/map%03d.fits" % cg.i, map)
		L.info("%4d %15.7e" % (cg.i, cg.err))
