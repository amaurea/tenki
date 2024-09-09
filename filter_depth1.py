import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("imaps", nargs="+", help="Input depth1 maps files. File names of associated files like ivar will be inferred from these.")
parser.add_argument("-b", "--beam", type=str,   required=True, help="Beam transform")
parser.add_argument("-f", "--freq", type=float, required=True, help="Frequency in GHz")
parser.add_argument("-L", "--lres", type=str, default="70,100", help="y,x block size to use when smoothing the noise spectrum")
parser.add_argument("-m", "--mask", type=str, default=None, help="Mask to use when building noise model, making e.g. point sources and bright galactic regions")
parser.add_argument(      "--apod-edge",    type=float, default=10)
parser.add_argument(      "--apod-holes",   type=float, default=10)
parser.add_argument(      "--shrink-holes", type=float, default=1.0, help="Don't apodize holes smaller than this number of beam fwhms")
parser.add_argument("-B", "--band-height",  type=float, default=2)
parser.add_argument(      "--highpass",     type=float, default=0)
parser.add_argument(      "--shift",        type=int,   default=0)
parser.add_argument("-c", "--cont",    action="store_true")
parser.add_argument(      "--simple",  action="store_true")
parser.add_argument(      "--simple-lknee", type=float, default=1000)
parser.add_argument(      "--simple-alpha", type=float, default=-3.5)
parser.add_argument(      "--noisemask-lim",type=float, default=None)
parser.add_argument("-p", "--pixwin",       type=str,   default="nn")
parser.add_argument(      "--suffix",       type=str,   default="")
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils, bunch, mpi, uharm, array_ops, analysis

# TODO: Find a home for these function in enlib. They're currently duplicated
# from coadd_depth1.
def fix_profile(profile):
	"""profile has a bug where it was generated over too big an
	az range. This is probably harmless, but could in theory lead to it moving
	past north, which makes the profile non-monotonous in dec. This function
	chops off those parts."""
	# First fix any wrapping issues
	profile[1] = utils.unwind(profile[1])
	# Then handle any north/south-crossing
	dec   = profile[0]
	i0    = len(dec)//2
	# Starting from the reference point i0, run left and right until
	# the slope switches sign. That's equivalent to the code below.
	# The point i0 is double-counted here, but that compensates for
	# the ddec calculation shortening the array.
	ddec  = dec[1:]-dec[:-1]
	good1 = ddec[:i0+1]*ddec[i0] > 0
	good2 = ddec[i0:]*ddec[i0] > 0
	good  = np.concatenate([good1,good2])
	return profile[:,good]

class ShiftMatrix:
	def __init__(self, shape, wcs, profile):
		"""Given a map geometry and a scanning profile [{dec,ra},:]
		create an operator that can be used to transform a map to/from
		a coordinate system where the scans are straight vertically."""
		map_decs, map_ras = enmap.posaxes(shape, wcs)
		# make sure it's sorted, otherwise interp won't work
		profile = fix_profile(profile)
		profile = profile[:,np.argsort(profile[0])]
		# get the profile ra for each dec in the map,
		# and since its position is arbitrary, put it in the middle ofthe
		# map to be safe
		ras     = np.interp(map_decs, profile[0], profile[1])
		ras    += np.mean(map_ras)-np.mean(ras)
		# Transform to x pixel positions
		xs      = enmap.sky2pix(shape, wcs, [map_decs,ras])[1]
		# We want to turn this into pixel *offsets* rather than
		# direct pixel values.
		dxs     = xs - shape[-1]/2
		# This operator just shifts whole pixels, it doesn't interpolate
		dxs     = utils.nint(dxs)
		self.dxs = dxs
	def forward(self, map):
		# Numpy can't do this efficiently
		return array_ops.roll_rows(map, -self.dxs)
	def backward(self, map):
		return array_ops.roll_rows(map,  self.dxs)

class ShiftDummy:
	def forward (self, map): return map.copy()
	def backward(self, map): return map.copy()

class NmatShiftConstCorr:
	def __init__(self, S, ivar, iC):
		self.S  = S
		self.H  = ivar**0.5
		self.iC = iC
	def apply(self, map, omap=None):
		sub = map.extract(self.H.shape, self.H.wcs)
		sub = self.H*self.S.backward(enmap.ifft(self.iC*enmap.fft(self.S.forward(self.H*sub))).real)
		if omap is None: return sub.extract(map.shape, map.wcs)
		else: return omap.insert(sub, op=np.add)

def build_ips2d_udgrade(map, ivar, lres=(70,100), apod_corr=1):
	# Apodize
	white = map*ivar**0.5
	# Compute the 2d spectrun
	ps2d = np.abs(enmap.fft(white))**2 / apod_corr
	del white
	# Smooth it. Using downgrade/upgrade is crude, but
	# has the advantage of being local, so that stong values
	# don't leak far
	down = np.maximum(1,utils.nint(lres/ps2d.lpixshape()))
	down[1] = max(down[1], 4)
	ps2d = enmap.upgrade(enmap.downgrade(ps2d, down, inclusive=True), down, inclusive=True, oshape=ps2d.shape)
	return 1/ps2d

def overlapping_range_iterator(n, nblock, overlap, padding=0):
	if nblock == 0: return
	# We don't handle overlap > half our block size
	min_bsize = n//nblock
	if min_bsize == 0: raise ValueError("Empty blocks in overlapping_range_iterator. Too low n or too high nblock? n = %d nblock = %d" % (n, nblock))
	overlap   = min(overlap, min_bsize//2)
	for bi in range(nblock):
		# Range we would have had without any overlap etc
		i1 = bi*n//nblock
		i2 = (bi+1)*n//nblock
		# Full range including overlap and padding
		ifull1 = max(i1-overlap-padding,0)
		ifull2 = min(i2+overlap+padding,n)
		# Range that crops away padding
		iuse1  = max(i1-overlap,0)
		iuse2  = min(i2+overlap,n)
		ionly1 = i1+overlap if bi > 0        else 0
		ionly2 = i2-overlap if bi < nblock-1 else n
		nover1 = ionly1-iuse1
		nover2 = iuse2-ionly2
		left   = (np.arange(nover1)+1)/(nover1+2)
		right  = (np.arange(nover2)+2)[::-1]/(nover2+2)
		middle = np.full(ionly2-ionly1,1.0)
		weight = np.concatenate([left,middle,right])
		yield bunch.Bunch(i1=ifull1, i2=ifull2, p1=iuse1-ifull1, p2=ifull2-iuse2, weight=weight)

def highpass_ips2d(ips2d, lknee, alpha=-20):
	l = np.maximum(ips2d.modlmap(), 0.5)
	return ips2d * (1 + (l/lknee)**alpha)**-1

def expand_files(fname):
	if fname.endswith(".txt"):
		with open(fname, "r") as ifile:
			return [line.strip() for line in ifile]
	else:
		return utils.glob(fname)

# This is a bit complicated because I have long lists of files, and I
# need to distinguish between before and after the beam change. Therefore
# a simple glob won't do.
imaps  = sum([sorted(expand_files(fname)) for fname in args.imaps],[])

comm   = mpi.COMM_WORLD
nfile  = len(imaps)
freq   = args.freq*1e9
beam1d = np.loadtxt(args.beam).T[1]
beam1d /= np.max(beam1d)
# Estimate the fwhm, which we use to determine which holes
# are too small to worry about
lfwhm  = np.where(beam1d<0.5)[0][0]
fwhm   = 1/(lfwhm*utils.fwhm)/utils.fwhm
# Apodization settings
apod_edge   = args.apod_edge *utils.arcmin
apod_holes  = args.apod_holes*utils.arcmin
shrink_holes= args.shrink_holes*fwhm
# Noise model resolution
lres = utils.parse_ints(args.lres)
# bands
band_height = args.band_height*utils.degree
# µK -> mJy/sr
fconv = utils.dplanck(freq)/1e3

for fi in range(comm.rank, nfile, comm.size):
	# Input file names
	mapfile  = imaps[fi]
	ivarfile = utils.replace(mapfile, "map", "ivar")
	infofile = utils.replace(utils.replace(mapfile, "map", "info"), ".fits", ".hdf")
	# Output file names
	rhofile  = utils.replace(mapfile, "map", "rho"+args.suffix)
	kappafile= utils.replace(mapfile, "map", "kappa"+args.suffix)
	# Optionally skip existing files
	if args.cont and os.path.isfile(rhofile) and os.path.isfile(kappafile):
		continue
	print("%4d %5d/%d Processing %s" % (comm.rank, fi+1, nfile, os.path.basename(mapfile)))
	# Read in our data
	map  = enmap.read_map(mapfile)  * fconv
	ivar = enmap.read_map(ivarfile) / fconv**2
	if args.shift > 0:
		info = bunch.read(infofile)
	dtype  = map.dtype
	ny, nx = map.shape[-2:]
	# Build our shift matrix
	if args.shift > 0: S = ShiftMatrix(map.shape, map.wcs, info.profile)
	else:              S = ShiftDummy()
	# Set up apodization. A bit messy due to handling two types of apodizion
	# depending on whether it's based on the extrnal mask or not
	hit  = ivar > 0
	if shrink_holes > 0:
		hit = enmap.shrink_mask(enmap.grow_mask(hit, shrink_holes), shrink_holes)
	# Apodize the edge by decreasing the significance in ivar
	noise_apod = enmap.apod_mask(hit, apod_edge)
	# Check if we have a noise model mask too
	mask = 0
	if args.mask:
		mask |= enmap.read_map(args.mask, geometry=map.geometry)
	# Optionally mask very bright regions
	if args.noisemask_lim:
		bright= np.abs(map.preflat[0] < args.noisemask_lim * fconv)
		rmask = 5*utils.arcmin
		mask |= bright.distance_transform(rmax=rmask) < rmask
		del bright
	mask = np.asanyarray(mask)
	if mask.size > 0 and mask.ndim > 0:
		noise_apod *= enmap.apod_mask(1-mask, apod_holes)
	del mask
	# Build the noise model
	iC  = build_ips2d_udgrade(S.forward(map), S.forward(ivar*noise_apod), apod_corr=np.mean(noise_apod**2), lres=lres)
	del noise_apod
	if args.highpass > 0:
		iC = highpass_ips2d(iC, args.highpass)

	# Set up output map buffers
	rho   = map*0
	kappa = map*0
	tot_weight = np.zeros(ny, map.dtype)
	# Bands. At least band_height in height and with at least
	# 2*apod_edge of overlapping padding at top and bottom. Using narrow bands
	# make the flat sky approximation a good approximation
	nband    = utils.ceil(map.extent()[0]/band_height) if band_height > 0 else 1
	bedge    = utils.ceil(apod_edge/map.pixshape()[0]) * 2
	boverlap = bedge
	for bi, r in enumerate(overlapping_range_iterator(ny, nband, boverlap, padding=bedge)):
		# Get band-local versions of the map etc.
		bmap, bivar, bhit = [a[...,r.i1:r.i2,:] for a in [map, ivar, hit]]
		bny, bnx = bmap.shape[-2:]
		if args.shift > 0: bS = ShiftMatrix(bmap.shape, bmap.wcs, info.profile)
		else:              bS = ShiftDummy()
		# Translate iC to smaller fourier space
		if args.simple:
			bl  = bmap.modlmap()
			biC = (1 + (np.maximum(bl,0.5)/args.simple_lknee)**args.simple_alpha)**-1
		else:
			biC     = enmap.ifftshift(enmap.resample(enmap.fftshift(iC), bmap.shape, method="spline", order=1))
			biC.wcs = bmap.wcs.deepcopy()
		# 2d beam
		beam2d  = enmap.samewcs(utils.interp(bmap.modlmap(), np.arange(len(beam1d)), beam1d), bmap)
		# Pixel window. We include it as part of the 2d beam, which is valid in the flat sky
		# approximation we use here.
		if args.pixwin in ["nn","0"]:
			beam2d = enmap.apply_window(beam2d, order=0, nofft=True)
		elif args.pixwin in ["lin","bilin","1"]:
			beam2d = enmap.apply_window(beam2d, order=1, nofft=True)
		elif args.pixwin in ["none"]:
			pass
		else:
			raise ValueError("Invalid pixel window '%s'" % str(args.pixwin))
		# Set up apodization
		filter_apod = enmap.apod_mask(bhit, apod_edge)
		bivar       = bivar*filter_apod
		del filter_apod
		# Phew! Actually perform the filtering
		uht  = uharm.UHT(bmap.shape, bmap.wcs, mode="flat")
		brho, bkappa = analysis.matched_filter_constcorr_dual(bmap, beam2d, bivar, biC, uht, S=bS.forward, iS=bS.backward)
		del uht
		# Restrict to exposed area
		brho   *= bhit
		bkappa *= bhit
		# Merge into full output
		weight = r.weight.astype(map.dtype)
		rho  [...,r.i1+r.p1:r.i2-r.p2,:] += brho  [...,r.p1:bny-r.p2,:]*weight[:,None]
		kappa[...,r.i1+r.p1:r.i2-r.p2,:] += bkappa[...,r.p1:bny-r.p2,:]*weight[:,None]
		tot_weight[r.i1+r.p1:r.i2-r.p2]  += weight
		del bmap, bivar, bhit, bS, biC, beam2d, brho, bkappa

	del map, ivar, iC
	# Write the output maps
	enmap.write_map(rhofile,   rho)
	enmap.write_map(kappafile, kappa)
	del rho, kappa
