import numpy as np, argparse, time, sys
from enlib import enmap, powspec, curvedsky, sharp, wcs as wcslib, memory
parser = argparse.ArgumentParser()
parser.add_argument("powspec")
parser.add_argument("omap")
parser.add_argument("-R", "--radius", type=float, default=4)
parser.add_argument("-r", "--res",    type=float, default=0.5)
parser.add_argument("-s", "--supersample", type=float, default=2.0)
parser.add_argument("-m", "--mmax", type=int, default=None)
parser.add_argument("--ncomp", type=int, default=3)
args = parser.parse_args()

deg2rad = np.pi/180
min2rad = np.pi/180/60
ncomp   = args.ncomp

class dprint:
	def __init__(self, desc):
		self.desc = desc
	def __enter__(self):
		self.t1 = time.time()
	def __exit__(self, type, value, traceback):
		sys.stderr.write("%6.2f %6.3f %6.3f %s\n" % (time.time()-self.t1,memory.current()/1024.**3, memory.max()/1024.**3, self.desc))

with dprint("spec"):
	ps = powspec.read_spectrum(args.powspec)[:ncomp,:ncomp]

# Construct our output coordinates, a zea system. My standard
# constructor doesn't handle pole crossing, so do it manually.
with dprint("construct omap"):
	R   = args.radius*deg2rad
	res = args.res*min2rad
	wo  = wcslib.WCS(naxis=2)
	wo.wcs.ctype = ["RA---ZEA","DEC--ZEA"]
	wo.wcs.crval = [0,90]
	wo.wcs.cdelt = [res/deg2rad, res/deg2rad]
	wo.wcs.crpix = [1,1]
	x, y = wo.wcs_world2pix(0,90-R/deg2rad,1)
	y = int(np.ceil(y))
	n = 2*y-1
	wo.wcs.crpix = [y,y]
	omap = enmap.zeros((n,n),wo)

# Construct our projection coordinates this is a CAR system in order
# to make interpolation easy.
with dprint("construct imap"):
	ires = np.array([1,1./np.sin(R)])*res/args.supersample
	shape, wi = enmap.geometry(pos=[[np.pi/2-R,-np.pi],[np.pi/2,np.pi]], res=ires, proj="car")
	imap = enmap.zeros((ncomp,)+shape, wi)

# Define SHT for interpolation pixels
with dprint("construct sht"):
	minfo = curvedsky.map2minfo(imap)
	lmax_ideal = np.pi/res
	ps = ps[:,:,:lmax_ideal]
	lmax = ps.shape[-1]
	# We do not need all ms when centered on the pole. To reach 1e-10 relative
	# error up to R, we need mmax approx 9560*R in radians
	mmax = args.mmax or int(R*9560)
	ainfo = sharp.alm_info(lmax, mmax)
	sht = sharp.sht(minfo, ainfo)

with dprint("curvedsky tot"):
	with dprint("rand alm"):
		alm = curvedsky.rand_alm(ps, ainfo=ainfo, seed=1, m_major=False)
	with dprint("alm2map"):
		sht.alm2map(alm[:1], imap[:1].reshape(1,-1))
		if ncomp == 3:
			sht.alm2map(alm[1:3], imap[1:3,:].reshape(2,-1), spin=2)
		del alm
	# Make a test map to see if we can project between these
	with dprint("project"):
		omap = enmap.project(imap, omap.shape, omap.wcs, mode="constant", cval=np.nan)
		del imap

enmap.write_map(args.omap, omap)
