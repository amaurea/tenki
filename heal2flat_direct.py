import numpy as np, argparse, os, healpy
from enlib import utils, enmap, curvedsky, log, coordinates
parser = argparse.ArgumentParser()
parser.add_argument("ihealmap")
parser.add_argument("template")
parser.add_argument("ofile")
parser.add_argument("-n", "--ncomp", type=int, default=1)
parser.add_argument("-i", "--first", type=int, default=0)
parser.add_argument("-v", "--verbosity", type=int, default=2)
parser.add_argument("-r", "--rot",   type=str, default=None)
parser.add_argument("-u", "--unit",  type=float, default=1)
parser.add_argument("-O", "--order", type=int, default=0)
parser.add_argument("-s", "--scalar", action="store_true")
args = parser.parse_args()

log_level = log.verbosity2level(args.verbosity)
L = log.init(level=log_level)
ncomp = args.ncomp
assert ncomp == 1 or ncomp == 3, "Only 1 or 3 components supported"

# Read the input maps
L.info("Reading " + args.ihealmap)
imap    = np.atleast_2d(healpy.read_map(args.ihealmap, field=tuple(range(args.first,args.first+ncomp))))
nside   = healpy.npix2nside(imap.shape[-1])
mask    = imap < -1e20
dtype   = imap.dtype
bsize   = 100

if args.unit != 1: imap[~mask]/= args.unit

# Read the template
shape, wcs = enmap.read_map_geometry(args.template)
shape = (args.ncomp,)+shape[-2:]

# Allocate our output map
omap   = enmap.zeros(shape, wcs, dtype)
nblock = (omap.shape[-2]+bsize-1)//bsize

for bi in range(nblock):
	r1 = bi*bsize
	r2 = (bi+1)*bsize
	print "Processing row %5d/%d" % (r1, omap.shape[-2])
	
	# Output map coordinates
	osub = omap[...,r1:r2,:]
	pmap = osub.posmap()
	# Coordinate transformation
	if args.rot:
			s1,s2 = args.rot.split(",")
			opos  = coordinates.transform(s2, s1, pmap[::-1], pol=ncomp==3)
			pmap[...] = opos[1::-1]
			if len(opos) == 3: psi = -opos[2].copy()
			del opos
	# Switch to healpix convention
	theta = np.pi/2-pmap[0]
	phi   = pmap[1]
	# Evaluate map at these locations
	if args.order == 0:
		pix  = healpy.ang2pix(nside, theta, phi)
		osub[:] = imap[:,pix]
	elif args.order == 1:
		for i in range(ncomp):
			osub[i] = healpy.get_interp_val(imap[i], theta, phi)
	# Rotate polarization if necessary
	if args.rot and ncomp==3:
		osub[1:3] = enmap.rotate_pol(osub[1:3], psi)

print "Writing"
if args.scalar: omap = omap.preflat[0]
enmap.write_map(args.ofile, omap)
print "Done"
