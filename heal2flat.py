import numpy as np, argparse, os, healpy
from enlib import sharp, utils, enmap, curvedsky, log, coordinates
parser = argparse.ArgumentParser()
parser.add_argument("ihealmap")
parser.add_argument("templates", nargs="+")
parser.add_argument("ofile")
parser.add_argument("-n", "--ncomp", type=int, default=1)
parser.add_argument("-i", "--first", type=int, default=0)
parser.add_argument("-v", "--verbosity", type=int, default=2)
parser.add_argument("-r", "--rot", type=str, default=None)
parser.add_argument("-l", "--lmax",type=int, default=0)
parser.add_argument("--rot-method",type=str, default="alm")
parser.add_argument("-u", "--unit", type=float, default=1)
parser.add_argument("--oslice",    type=str, default=None)
args = parser.parse_args()

log_level = log.verbosity2level(args.verbosity)
L = log.init(level=log_level)

# equatorial to galactic euler zyz angles
euler = np.array([57.06793215,  62.87115487, -167.14056929])*utils.degree

# If multiple templates are specified, the output file is
# interpreted as an output directory.
if len(args.templates) > 1:
	utils.mkdir(args.ofile)
ncomp = args.ncomp

assert ncomp == 1 or ncomp == 3, "Only 1 or 3 components supported"
dtype = np.float32
ctype = np.result_type(dtype,0j)

# Read the input maps
L.info("Reading " + args.ihealmap)
m = np.atleast_2d(healpy.read_map(args.ihealmap, field=tuple(range(args.first,args.first+ncomp)))).astype(dtype,copy=False)
mask    = m < -1e20
m[mask] = np.mean(m[~mask])

if args.unit != 1: m /= args.unit
# Prepare the transformation
L.debug("Preparing SHT")
nside = healpy.npix2nside(m.shape[1])
lmax  = args.lmax or 3*nside
minfo = sharp.map_info_healpix(nside)
ainfo = sharp.alm_info(lmax)
sht   = sharp.sht(minfo, ainfo)
alm   = np.zeros((ncomp,ainfo.nelem), dtype=ctype)
# Perform the actual transform
L.debug("T -> alm")
print m.dtype, alm.dtype
sht.map2alm(m[0], alm[0])
if ncomp == 3:
	L.debug("P -> alm")
	sht.map2alm(m[1:3],alm[1:3], spin=2)
del m

# Project down on each template
for tfile in args.templates:
	L.info("Reading " + tfile)
	shape, wcs = enmap.read_map(tfile).geometry

	if args.rot and args.rot_method != "alm":
		# Rotate by displacing coordinates and then fixing the polarization
		L.debug("Computing pixel positions")
		pmap = enmap.posmap(shape, wcs)
		if args.rot:
			L.debug("Computing rotated positions")
			s1,s2 = args.rot.split(",")
			opos = coordinates.transform(s2, s1, pmap[::-1], pol=ncomp==3)
			pmap[...] = opos[1::-1]
			if len(opos) == 3: psi = -opos[2].copy()
			del opos
		L.debug("Projecting")
		res  = curvedsky.alm2map_pos(alm, pmap)
		if args.rot and ncomp==3:
			L.debug("Rotating polarization vectors")
			res[1:3] = enmap.rotate_pol(res[1:3], psi)
	else:
		# We will project directly onto target map if possible
		if args.rot:
			L.debug("Rotating alms")
			s1,s2 = args.rot.split(",")
			if s1 != s2:
				# Note: rotate_alm does not actually modify alm
				# if it is single precision
				alm = alm.astype(np.complex128,copy=False)
				if s1 == "gal" and (s2 == "equ" or s2 == "cel"):
					healpy.rotate_alm(alm, euler[0], euler[1], euler[2])
				elif s2 == "gal" and (s1 == "equ" or s1 == "cel"):
					healpy.rotate_alm(alm,-euler[2],-euler[1],-euler[0])
				else:
					raise NotImplementedError
				alm = alm.astype(ctype,copy=False)
		L.debug("Projecting")
		res = enmap.zeros((len(alm),)+shape[-2:], wcs, dtype)
		res = curvedsky.alm2map(alm, res)
	if len(args.templates) > 1:
		oname = args.odir + "/" + os.path.basename(tfile)
	else:
		oname = args.ofile
	if args.oslice:
		res = eval("res"+args.oslice)
	L.info("Writing " + oname)
	enmap.write_map(oname, res)
