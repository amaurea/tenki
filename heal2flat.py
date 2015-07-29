import numpy as np, argparse, os, healpy
from enlib import sharp, utils, enmap, curvedsky, log, coordinates
parser = argparse.ArgumentParser()
parser.add_argument("ihealmap")
parser.add_argument("templates", nargs="+")
parser.add_argument("odir")
parser.add_argument("-n", "--ncomp", type=int, default=1)
parser.add_argument("-i", "--first", type=int, default=0)
parser.add_argument("-v", "--verbosity", type=int, default=2)
parser.add_argument("-r", "--rot", type=str, default=None)
parser.add_argument("-u", "--unit", type=float, default=1)
args = parser.parse_args()

log_level = log.verbosity2level(args.verbosity)
L = log.init(level=log_level)

utils.mkdir(args.odir)
ncomp = args.ncomp

assert ncomp == 1 or ncomp == 3, "Only 1 or 3 components supported"
# Read the input maps
L.info("Reading " + args.ihealmap)
m = np.atleast_2d(healpy.read_map(args.ihealmap, field=tuple(range(args.first,args.first+ncomp))))
if args.unit != 1: m /= args.unit
# Prepare the transformation
L.debug("Preparing SHT")
nside = healpy.npix2nside(m.shape[1])
lmax  = 3*nside
minfo = sharp.map_info_healpix(nside)
ainfo = sharp.alm_info(lmax)
sht   = sharp.sht(minfo, ainfo)
alm   = np.zeros((ncomp,ainfo.nelem), dtype=np.complex)
# Perform the actual transform
L.debug("T -> alm")
sht.map2alm(m[0], alm[0])
if ncomp == 3:
	L.debug("P -> alm")
	sht.map2alm(m[1:3],alm[1:3], spin=2)
del m

# Project down on each template
for tfile in args.templates:
	L.info("Reading " + tfile)
	tmap = enmap.read_map(tfile)
	L.debug("Computing pixel positions")
	pmap = tmap.posmap()
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
	L.info("Writing " + args.odir + "/" + os.path.basename(tfile))
	enmap.write_map(args.odir + "/" + os.path.basename(tfile), res)
