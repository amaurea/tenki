import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ifile")
parser.add_argument("ofile")
parser.add_argument("-r", "--res",  type=int,   default=16)
parser.add_argument("-l", "--lmax", type=int,   default=21000)
parser.add_argument(      "--vmax", type=float, default=1e5)
parser.add_argument(      "--off",  type=str,   default="-0.5,-0.5")
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils, wcsutils, curvedsky
from enlib import bench

# Build mercator geometry. Res is log2 of the number of pixels around the sky.
# Web maps have 256 pixels around the sky at z=0, so z = res+8. Normal ACT maps
# have a resolution of 0.5 arcmin, corresponding to res = 15.4. So to preserve our
# full resolution we would need res = 16, but res = 15 = 0.66', which is probably good enough.
npix  = 2**args.res
res   = 360/npix
shape = (npix, npix)
off   = utils.parse_floats(args.off)
wcs   = wcsutils.explicit(ctype=["RA---MER", "DEC--MER"], cdelt=[-res,res], crval=[0,0], crpix=[npix//2+1+off[1],npix//2+1+off[0]])

def logify(x, x0): return np.arcsinh(x/x0)
def unlogify(x, x0): return np.sinh(x)*x0

# Read in map
with bench.show("read"):
	imap  = enmap.read_map(args.ifile)
shape = imap.shape[:-2]+shape
dtype = imap.dtype

# Get rid of too high values, to avoid ringing
with bench.show("logify"):
	imap = logify(imap, args.vmax)

# Build a mask of exposed areas
with bench.show("build mask"):
	mask = imap != 0
	oy   = np.arange(npix)
	ox   = np.arange(npix)
	iy   = np.maximum(0, np.minimum(imap.shape[-2]-1, utils.nint(mask.sky2pix(enmap.pix2sky(shape, wcs, [oy,oy*0])))[0]))
	ix   = np.maximum(0, np.minimum(imap.shape[-1]-1, utils.nint(mask.sky2pix(enmap.pix2sky(shape, wcs, [ox*0,ox])))[1]))
	mask = mask[...,iy,:][...,:,ix]

# Transform
with bench.show("map2alm"):
	alms  = curvedsky.map2alm(imap, lmax=args.lmax, tweak=True)
del imap
omap  = enmap.zeros(shape, wcs, dtype)
with bench.show("alm2map"):
	omap  = curvedsky.alm2map_cyl(alms, omap)

# Restore the map to linear scale
with bench.show("unlogify"):
	omap = unlogify(omap, args.vmax)

# Apply the mask
with bench.show("apply mask"):
	omap *= mask
	del mask

# And write
with bench.show("write"):
	enmap.write_map(args.ofile, omap)
