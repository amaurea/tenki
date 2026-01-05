import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("imaps", nargs="+", help="Input depth1 maps files. File names of associated files like ivar will be inferred from these.")
parser.add_argument("-b", "--beam", type=str,   required=True, help="Beam transform")
parser.add_argument("-f", "--freq", type=float, required=True, help="Frequency in GHz")
parser.add_argument("-L", "--lres", type=str, default="70,100", help="y,x block size to use when smoothing the noise spectrum")
parser.add_argument("-m", "--mask", type=str, default=None, help="Mask to use when building noise model, making e.g. point sources and bright galactic regions")
parser.add_argument(      "--crop-edge",    type=float, default=2)
parser.add_argument(      "--apod-edge",    type=float, default=10)
parser.add_argument(      "--apod-holes",   type=float, default=10)
parser.add_argument(      "--shrink-holes", type=float, default=1.0, help="Don't apodize holes smaller than this number of beam fwhms")
parser.add_argument("-F", "--bsize-flat",  type=float, default=2)
parser.add_argument("-B", "--bsize-noise", type=float, default=10)
parser.add_argument(      "--highpass",     type=float, default=0)
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

class DataMissing(Exception): pass

def get_beam(fname):
	try:
		bsize = float(fname)*utils.fwhm*utils.arcmin
		l     = np.arange(20000)
		return np.exp(-0.5*(l*bsize)**2)
	except ValueError:
		beam  = np.loadtxt(fname).T[1]
		beam /= np.max(beam)
		return beam

def build_ips2d_udgrade(map, ivar, apod_map, lres=(70,100)):
	# Apodize
	white = map*(ivar*apod_map)**0.5
	# Compute the 2d spectrun
	ps2d = np.abs(enmap.fft(white))**2 / np.mean(apod_map)
	del white
	# Smooth it. Using downgrade/upgrade is crude, but
	# has the advantage of being local, so that stong values
	# don't leak far
	down = np.maximum(1,utils.nint(lres/ps2d.lpixshape()))
	down[1] = max(down[1], 4)
	ps2d = enmap.upgrade(enmap.downgrade(ps2d, down, inclusive=True), down, inclusive=True, oshape=ps2d.shape)
	return 1/ps2d

def build_ips2d_hblock(map, ivar, apod_map, apod_edge=10*utils.arcmin, bsize=500, lres=(70,100), hit_tol=0.25):
	apod_pix = utils.ceil(apod_edge/map.pixshape()[1])
	# outputs
	ispecs  = []
	weights = []
	# Loop over blocks
	for x1 in range(0, map.shape[-1], bsize):
		x2     = x1+bsize
		pixbox = np.array([[0,x1],[map.shape[-2],x2]])
		bmap, bivar, bapod = [a.extract_pixbox(pixbox) for a in [map, ivar, apod_map]]
		bapod = enmap.apod(bapod, apod_pix).astype(map.dtype)
		if np.mean(bapod) == 0: continue
		ips2d = build_ips2d_udgrade(bmap, bivar, apod_map=bapod, lres=lres)
		weight= np.mean(bapod**2)
		ispecs.append(ips2d)
		weights.append(weight)
	if len(ispecs) == 0: raise DataMissing("All blocks empty")
	ispecs = enmap.enmap(ispecs)
	weights= np.array(weights)
	assert ispecs.dtype == map.dtype, "ispecs.dtype %s != map.dtype %s" % (str(ispecs.dtype), str(map.dtype))
	# Ignore ones with too low weight
	good = weights > np.max(weight)*hit_tol
	ispecs = ispecs[good]
	# Median of the rest
	return np.median(ispecs,0)

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
minsize= 10
nfile  = len(imaps)
freq   = args.freq*1e9
beam1d = get_beam(args.beam)
# Estimate the fwhm, which we use to determine which holes
# are too small to worry about
lfwhm  = 2*np.where(beam1d<0.5)[0][0]
fwhm   = 1/(lfwhm*utils.fwhm)/utils.fwhm # radians
# Apodization settings
apod_edge   = args.apod_edge *utils.arcmin
apod_holes  = args.apod_holes*utils.arcmin
shrink_holes= args.shrink_holes*fwhm
crop_edge   = args.crop_edge*fwhm
# Noise model resolution
lres = utils.parse_ints(args.lres)
# bands
bsize_noise = args.bsize_noise*utils.degree
bsize_flat  = args.bsize_flat *utils.degree
# µK -> mJy/sr
fconv = utils.dplanck(freq)/1e3

for fi in range(comm.rank, nfile, comm.size):
	# Input file names
	mapfile  = imaps[fi]
	ivarfile = utils.replace(mapfile, "map", "ivar")
	#infofile = utils.replace(utils.replace(mapfile, "map", "info"), ".fits", ".hdf")
	# Output file names
	rhofile  = utils.replace(mapfile, "map", "rho"+args.suffix)
	kappafile= utils.replace(mapfile, "map", "kappa"+args.suffix)
	# Optionally skip existing files
	if args.cont and os.path.isfile(rhofile) and os.path.isfile(kappafile):
		continue
	print("%4d %5d/%d Processing %s" % (comm.rank, fi+1, nfile, os.path.basename(mapfile)))
	# Read in our data
	map  = enmap.read_map(mapfile)
	if map.shape[-2] < minsize or map.shape[-1] < minsize:
		print("%4d Skipping %s: too small" % (comm.rank, os.path.basename(mapfile)))
		continue
	ivar = enmap.read_map(ivarfile)
	dtype  = map.dtype
	ny, nx = map.shape[-2:]
	# Convet to mJy/sr
	map  *= fconv
	ivar /= fconv**2
	# Set up apodization. A bit messy due to handling two types of apodizion
	# depending on whether it's based on the extrnal mask or not
	hit  = ivar > 0
	if shrink_holes > 0:
		hit = enmap.shrink_mask(enmap.grow_mask(hit, shrink_holes), shrink_holes+crop_edge)
	# Apodize the edge by decreasing the significance in ivar
	noise_apod = enmap.apod_mask(hit, apod_edge).astype(map.dtype)
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

	# Set up output map buffers
	rho   = map*0
	kappa = map*0

	# Bands. Loop over broad dec-bands for noise due to scanning curvature,
	# and smaller sub-bands for the filtering itself due to the flat sky
	# approximation.
	map_height = map.extent()[0]
	pix_height = map.pixshape()[0]
	nband = utils.ceil(map_height/bsize_noise) if bsize_noise > 0 else 1
	nsub  = utils.ceil(bsize_noise/bsize_flat) if bsize_flat  > 0 else 1
	# 2*apod_edge overlap
	pad   = utils.ceil(2*apod_edge/pix_height)

	sub_iterator = overlapping_range_iterator(ny, nband*nsub, overlap=pad, padding=pad)

	# Ok, ready to loop over noise bands
	for nbi, nr in enumerate(overlapping_range_iterator(ny, nband, overlap=pad, padding=pad)):
		# Get band-local versions of the map etc.
		nbmap, nbivar, nbhit, nbapod = [a[...,nr.i1:nr.i2,:] for a in [map, ivar, hit, noise_apod]]

		# Build the noise model
		try:
			#iC  = build_ips2d_udgrade(nbmap, nbivar, apod_map=nbapod, lres=lres)
			iC  = build_ips2d_hblock (nbmap, nbivar, apod_map=nbapod, apod_edge=apod_edge, lres=lres)
			if args.highpass > 0:
				iC = highpass_ips2d(iC, args.highpass)
		except DataMissing as e:
			print("Warning: Failed to build noise model for %s noise band %d: %s" % (mapfile, nbi, str(e)))
			continue

		# Loop over our sub-bands
		for bi in range(nsub):
			r = next(sub_iterator)
			# Get band-local versions of the map etc.
			bmap, bivar, bhit = [a[...,r.i1:r.i2,:] for a in [map, ivar, hit]]
			bny, bnx = bmap.shape[-2:]
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
			brho, bkappa = analysis.matched_filter_constcorr_dual(bmap, beam2d, bivar, biC, uht)
			del uht
			# Restrict to exposed area
			brho   *= bhit
			bkappa *= bhit
			# Merge into full output
			weight = r.weight.astype(map.dtype)
			rho  [...,r.i1+r.p1:r.i2-r.p2,:] += brho  [...,r.p1:bny-r.p2,:]*weight[:,None]
			kappa[...,r.i1+r.p1:r.i2-r.p2,:] += bkappa[...,r.p1:bny-r.p2,:]*weight[:,None]
			del bmap, bivar, bhit, beam2d, brho, bkappa
		del nbmap, nbivar, nbhit, nbapod, iC

	del map, ivar
	# Write the output maps
	enmap.write_map(rhofile,   rho)
	enmap.write_map(kappafile, kappa)
	del rho, kappa
