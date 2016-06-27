import numpy as np, argparse, h5py, enlib.cg, scipy.interpolate, time
from enlib import enmap, fft, coordinates, utils, bunch, interpol, bench, zipper, mpi, log
parser = argparse.ArgumentParser()
parser.add_argument("rhs")
parser.add_argument("infos", nargs="+")
parser.add_argument("odir")
parser.add_argument("--nmax",            type=int, default=0)
parser.add_argument("-d", "--downgrade", type=int, default=1)
parser.add_argument("-O", "--order",     type=int, default=3)
args = parser.parse_args()

fft.engine = "fftw"
comm = mpi.COMM_WORLD
log_level = log.verbosity2level(1)
L = log.init(level=log_level, rank=comm.rank, shared=False)
dtype = np.float32

def prepare(map):
	if map.ndim == 3: map = map[:1]
	map[...,:1,:]  = 0
	map[...,-1:,:] = 0
	map[...,:,:1]  = 0
	map[...,:,-1:] = 0
	map = enmap.downgrade(map, args.downgrade)
	return map

def calc_az_sweep(pattern, offset, site, pad=2.0, subsample=1.0):
	el1 = pattern[0] + offset[0]
	az1 = pattern[1] + offset[1] - pad
	az2 = pattern[2] + offset[1] + pad
	daz = rhs.wcs.wcs.cdelt[1]/np.cos(el1*utils.degree)/subsample
	naz  = int(np.ceil((az2-az1)/daz))
	# Simulate a single sweep at arbitrary time
	sweep_az = np.arange(naz)*daz + az1
	sweep_el = np.full(naz,el1)
	sweep_cel = coordinates.transform("hor","cel",
			np.array([sweep_az,sweep_el])*utils.degree,time=55500,site=site)
	# Make ra safe
	sweep_cel = utils.unwind(sweep_cel)
	return sweep_cel, naz, daz

class UnskewCurved:
	def __init__(self, shape, wcs, pattern, offset, site, pad=2.0, order=3, subsample=2.0):
		# Find the unskew transformation for this pattern.
		# We basically want dec->az and ra->ra0, with az spacing
		# similar to el spacing.
		ndec, nra = shape[-2:]
		(sweep_ra, sweep_dec), naz, daz = calc_az_sweep(pattern, offset, site, pad=pad, subsample=subsample)
		# We want to be able to go from (y,x) to (ra,dec), with
		# dec = dec[y]
		# ra  = ra[y]-ra[0]+x
		# Precompute the pixel mapping. This will have the full witdh in ra,
		# but will be smaller in dec due to the limited az range.
		raw_dec, raw_ra = enmap.posmap(shape, wcs)
		skew_pos = np.zeros((2, naz, nra))
		skew_pos[0] = sweep_dec[:,None]
		skew_pos[1] = (sweep_ra-sweep_ra[0])[:,None] + raw_ra[None,ndec/2,:]
		skew_pix = enmap.sky2pix(shape, wcs, skew_pos).astype(dtype)
		# Save
		self.order    = order
		self.shape    = shape
		self.wcs      = wcs
		self.pattern  = pattern
		self.site     = site
		self.skew_pix = skew_pix
		self.daz      = daz
		self.naz      = naz
	def apply(self, map):
		work = np.ascontiguousarray(utils.moveaxis(map,0,-1))
		omap = interpol.map_coordinates(work, self.skew_pix, order=self.order)
		omap = np.ascontiguousarray(utils.moveaxis(omap,-1,0))
		return omap
	def trans(self, imap, omap):
		imap = np.ascontiguousarray(utils.moveaxis(imap,0,-1))
		omap = np.ascontiguousarray(utils.moveaxis(omap,0,-1))
		interpol.map_coordinates(omap, self.skew_pix, odata=imap, trans=True, order=self.order)
		return np.ascontiguousarray(utils.moveaxis(omap,-1,0))

#TODO: also implement UnskewFourier, which should be the fastest method.

class UnskewShift:
	def __init__(self, shape, wcs, pattern, offset, site, pad=2.0):
		"""This unskew operation assumes that equal spacing in
		dec corresponds to equal spacing in time, and that shifts in
		RA can be done in units of whole pixels. This is an approximation
		relative to UnskewCurved, but it should be much faster."""
		ndec, nra = shape[-2:]
		(sweep_ra, sweep_dec), naz, daz = calc_az_sweep(pattern, offset, site, pad=pad)
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
		spline  = scipy.interpolate.UnivariateSpline(sweep_dec, sweep_ra)
		ra      = spline(dec)
		dra     = ra - ra[len(ra)/2]
		y, x    = enmap.sky2pix(shape, wcs, [dec,ra])
		dx      = x-x[len(x)/2]
		# We also need to know how much az changes for each pixel in
		# dec, after unskewing. First get the change in az for each point
		# in our simulated azimuth sweep.
		step_ra  = sweep_ra[1:]-sweep_ra[:-1]
		step_dec = sweep_dec[1:]-sweep_dec[:-1]
		step_az  = ((step_ra*np.cos(sweep_dec[1:]))**2 + step_dec**2)**0.5
		# We now have the az change per step. But we want to be able to
		# translate from shifted pixels to time. For that we need the
		# speed (which we have) and the azimuth size per pixel.
		# Assuming a constant slope, the az change is proportional to the
		# dec change, so we can just rescale by the pixel/sample dec step.
		daz      = np.mean(step_az)*np.mean(np.abs(dec[1:]-dec[:-1]))/np.mean(np.abs(step_dec))/utils.degree
		# Sanity check
		daz2 = (pattern[2]-pattern[1]+2*pad)/len(y)
		print "daz", daz, daz2, y1, y2, shape[-2]
		daz = daz2



		# And store the result
		self.y  = y.astype(int)
		self.dx = np.round(dx).astype(int)
		np.savetxt("pat_"+"_".join([str(w) for w in pattern]) + ".txt", np.array([self.y, self.dx]).T, fmt="%8.3f")
		self.daz= daz
		self.naz= len(y)
		self.dx_raw = dx
	def apply(self, map):
		omap = np.zeros(map.shape[:-2]+(self.naz,map.shape[-1]), map.dtype)
		for i, y in enumerate(self.y):
			omap[...,i,:] = np.roll(map[...,y,:],-self.dx[i],-1)
		return omap
	def trans(self, imap, omap):
		omap[:] = 0
		for i, y in enumerate(self.y):
			omap[...,y,:] = np.roll(imap[...,i,:],self.dx[i],-1)
		return omap

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

class Nmat:
	def __init__(self, naz, inspec, scale):
		freqs = fft.rfftfreq(naz) * scale
		self.spec_full = utils.interpol(inspec, freqs[None])
		#np.savetxt("foo.txt", self.spec_full)
		#np.savetxt("bar.txt", inspec)
		#1/0
		self.scale = scale
	def apply(self, arr, inplace=False):
		# Because of our padding and multiplication by the hitcount
		# before this, we should be safely apodized, and can assume
		# periodic boundaries
		if not inplace: arr = np.array(arr)
		ft = fft.rfft(arr, axes=[-2])
		ft *= self.spec_full[:,None]
		return fft.ifft(ft, arr, axes=[-2], normalize=True)

class Amat:
	def __init__(self, dof, infos, comm):
		self.dof   = dof
		self.infos = infos
		self.comm  = comm
	def __call__(self, x):
		xmap = self.dof.unzip(x)
		res  = xmap*0
		for info in self.infos:
			t0 = time.time()
			work  = xmap*info.H
			t1 = time.time()
			flat  = info.U.apply(work)
			t2 = time.time()
			flat  = info.N.apply(flat,inplace=True)
			t3 = time.time()
			work = enmap.samewcs(info.U.trans(flat, work),work)
			t4 = time.time()
			work *= info.H
			t5 = time.time()
			#print "t %4.2f %4.2f %4.2f %4.2f %4.2f" % (t1-t0, t2-t1, t3-t2, t4-t3, t5-t4), info.U.naz, info.U.daz
			res  += work
		res = utils.allreduce(res,comm)
		return self.dof.zip(res)

def normalize_hits(hits):
	"""Normalize the hitcounts by multiplying by a factor such that
	hits_scaled**2 approx hits for most of its area. This is not
	the same thing as taking the square root. 1d sims indicate that
	this is a better approximation, but I'm not sure."""
	medval = np.median(hits[hits!=0])
	return hits/medval**0.5

tref  = 55500
rhs   = enmap.read_map(args.rhs)
rhs   = prepare(rhs)
ipos  = rhs.posmap()
infos = []
ifiles= args.infos
if args.nmax: ifiles = ifiles[:args.nmax]
for infofile in ifiles[comm.rank::comm.size]:
	L.info("Reading %s" % (infofile))
	with h5py.File(infofile, "r") as hfile:
		hits   = enmap.samewcs(hfile["hits"].value, rhs)
		srate  = hfile["srate"].value
		speed  = hfile["speed"].value
		inspec = hfile["inspec"].value
		offset = hfile["offset"].value
		site   = bunch.Bunch(**{k:hfile["site"][k].value for k in hfile["site"]})
		pattern= hfile["pattern"].value
	hits = prepare(hits)
	#U = UnskewCurved(rhs.shape, rhs.wcs, pattern, offset, site, order=args.order)
	U = UnskewShift(rhs.shape, rhs.wcs, pattern, offset, site)
	scale = calc_scale(inspec.size, srate, speed, U.daz)
	#foo = enmap.samewcs(U.apply(rhs), rhs)
	#enmap.write_map("foo.fits", foo)
	#U.trans(foo,rhs)
	#enmap.write_map("foo2.fits", rhs)
	#1/0



	N = Nmat(U.naz, inspec, scale)
	H = normalize_hits(hits)
	infos.append(bunch.Bunch(U=U,N=N,H=H,pattern=pattern,site=site,srate=srate,scale=scale,speed=speed))

dof = zipper.ArrayZipper(rhs.copy())
A   = Amat(dof, infos, comm)
cg  = enlib.cg.CG(A, dof.zip(rhs))

utils.mkdir(args.odir)
for i in range(1000):
	cg.step()
	if comm.rank == 0:
		print np.std(cg.x[cg.x!=0])
		if cg.i % 10 == 0:
			map = dof.unzip(cg.x)
			enmap.write_map(args.odir + "/map%03d.fits" % cg.i, map)
		L.info("%4d %15.7e" % (cg.i, cg.err))
