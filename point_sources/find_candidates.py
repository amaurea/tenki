import numpy as np, argparse, sys, itertools, os, errno
from enlib import enmap, powspec, utils, fft, array_ops, gibbs, log
from enlib.degrees_of_freedom import DOF
from enlib.cg import CG
from scipy import ndimage
parser = argparse.ArgumentParser()
parser.add_argument("map")
parser.add_argument("noise")
parser.add_argument("powspec")
parser.add_argument("odir")
parser.add_argument("--srcrms", type=float, default=5)
parser.add_argument("-b", "--beam", type=float, default=1.5)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--rmask", type=int, default=5)
parser.add_argument("--siglim", type=float, default=3.5)
parser.add_argument("--masklim", type=float, default=6)
parser.add_argument("--mindist", type=int, default=2)
parser.add_argument("--nmax", type=int, default=350)
parser.add_argument("--noise-max", type=float, default=1e5)
parser.add_argument("--nocrop", action="store_true")
args = parser.parse_args()

log_level = log.verbosity2level(args.verbose+1)
L = log.init(level=log_level)
L.info("Init")

utils.mkdir(args.odir)

L.info("Reading %s" % args.map)
map    = enmap.read_map(args.map)
map    = map.reshape((-1,)+map.shape[-2:])[0][None]
L.info("Reading %s" % args.noise)
inoise  = enmap.read_map(args.noise)
inoise  = inoise.reshape((-1,)+inoise.shape[-2:])[0][None,None]
print map.shape, inoise.shape
shape_orig = map.shape[-2:]
# Mask out too noisy areas
L.info("Masking")
mask = inoise[0,0] < args.noise_max**-2
map[:,mask] = np.nan
if not args.nocrop:
	L.info("Cropping")
	map, info = enmap.autocrop(map, method="fft", return_info=True)
	inoise = enmap.padcrop(inoise, info)
	shape_crop = map.shape[-2:]
	L.info("Cropped from %s to %s" % (str(shape_orig),str(shape_crop)))
map[np.isnan(map)] = 0

## Truncate to fft-friendly areas
#h = fft.fft_len(map.shape[-2],[2,3,5])
#w = fft.fft_len(map.shape[-1],[2,3,5])
#map    = map[:,:h,:w]
#inoise = inoise[:,:,:h,:w]
#map[~np.isfinite(map)] = 0

# Kill extreme values
L.info("Removing extreme outliers above 1e6")
map = np.minimum(1e6,np.maximum(-1e6,map))

nmin = 50
#cut = 0x100
#map = map[:,h/2-cut:h/2+cut,w/2-cut:w/2+cut]
#inoise = inoise[:,:,h/2-cut:h/2+cut,w/2-cut:w/2+cut]
#map = map[:,250:750,700:1700]
#inoise = inoise[:,:,250:750,700:1700]

L.info("Writing preprocessed map")
enmap.write_map(args.odir + "/map.fits", map[0])

L.info("Reading spectrum " + args.powspec)
ps_cmb_tmp = powspec.read_spectrum(args.powspec)[:1,:1,:]
# Extend to max l
lmax     = np.max(np.sum(map.lmap()**2,0)**0.5).astype(int)+1
lmax_tmp = min(ps_cmb_tmp.shape[-1],lmax)
ps_cmb = np.zeros([1,1,lmax])
ps_cmb[:,:,:lmax_tmp] = ps_cmb_tmp[:,:,:lmax_tmp]
ps_cmb[:,:,lmax_tmp:] = ps_cmb_tmp[:,:,-1]

sigma  = args.beam * np.pi/180/60/(8*np.log(2))
l      = np.arange(ps_cmb.shape[-1])
ps_src = np.exp(-l*(l+1)*sigma**2)[None,None,:]*(args.srcrms*np.pi/180/60)**2

L.info("Setting up signal and noise matrices")
S  = enmap.spec2flat(map.shape, map.wcs, ps_cmb, 1.0)
iP = enmap.spec2flat(map.shape, map.wcs, ps_src,-1.0)
N  = (inoise + np.max(inoise)*1e-3)**-1

# apodize map based on a smooth noise map
L.info("Apodizing")
print inoise.shape, inoise.dtype
inoise_smooth = enmap.smooth_gauss(inoise[0,0],10*np.pi/180/60)[None,None]
apod = (np.minimum(1,inoise_smooth/(np.max(inoise_smooth)*0.05))**4)[0,0]
map *= apod[None]

enmap.write_map(args.odir + "/inoise.fits", inoise)
enmap.write_map(args.odir + "/inoise_smooth.fits", inoise_smooth)
enmap.write_map(args.odir + "/apod.fits", apod)
enmap.write_map(args.odir + "/map_apod.fits", map)

def mul(mat, vec, axes=[0,1]):
	return enmap.samewcs(array_ops.matmul(mat.astype(vec.dtype),vec,axes=axes),mat,vec)
def pow(mat, exp, axes=[0,1]): return enmap.samewcs(array_ops.eigpow(mat,exp,axes=axes),mat,exp)

class Trifilter:
	"""Solves the equation (S+N+P)P"x = map, where S is the CMB noise covariance,
	N is the noise covariance and P is the beam-point-source covariance.
	This looks like a wiener filter. But is it?"""
	def __init__(self, map, iP, S, noise):
		self.map, self.iP, self.S, self.noise = map, iP, S, noise
		# Construct preconditioner. This assumes constant noise
		pixarea = map.area()/np.prod(map.shape[-2:])
		ntyp = (np.median(inoise[inoise>np.max(inoise)*0.1])/pixarea)**-0.5
		lmax = np.max(np.sum(map.lmap()**2,0)**0.5)+1
		ps_ntyp = np.zeros(lmax)[None,None]+ntyp**2
		Ntyp = enmap.spec2flat(map.shape, map.wcs, ps_ntyp, 1.0)
		self.prec = pow(mul(S+Ntyp,iP)+1,-1)
	def A(self, xmap):
		#enmap.write_map(args.odir + "/A1.fits", xmap)
		iPx   = mul(self.iP,enmap.map2harm(xmap))
		#enmap.write_map(args.odir + "/A2.fits", iPx.real)
		SiPx  = enmap.harm2map(mul(self.S, iPx))
		#enmap.write_map(args.odir + "/A3.fits", SiPx)
		NiPx  = mul(self.noise,enmap.harm2map(iPx))
		#enmap.write_map(args.odir + "/A3.fits", NiPx)
		res = xmap + SiPx + NiPx
		#enmap.write_map(args.odir + "/A4.fits", res)
		return res
	def M(self, xmap):
		#enmap.write_map(args.odir + "/M1.fits", xmap)
		res = enmap.harm2map(mul(self.prec,enmap.map2harm(xmap)))
		#enmap.write_map(args.odir + "/M2.fits", res)
		return res
	def solve(self, b, x0=None, verbose=False, nmin=0):
		if x0 is None: x0 = b*0
		def wrap(fun):
			def foo(x):
				xmap = enmap.samewcs(x.reshape(b.shape),b)
				return fun(xmap).reshape(-1)
			return foo
		solver = CG(wrap(self.A), b.reshape(-1), x0=x0.reshape(-1), M=wrap(self.M))
		#for i in range(50):
		#while solver.err > 1e-6:
		while solver.err > 1e-2 or solver.i < nmin:
			solver.step()
			if verbose:
				print "%5d %15.7e %15.7e" % (solver.i, solver.err, solver.err_true)
		return enmap.samewcs(solver.x.reshape(b.shape),b)

# Try solving
trifilter = Trifilter(map, iP, S, N)
L.info("Filtering map")
pmap = trifilter.solve(map, verbose=args.verbose, nmin=nmin)
L.info("Writing pmap")
enmap.write_map(args.odir+"/pmap.fits",pmap[0])

# Estimate uncertainty with a couple of other realizations
varmap = inoise*0
nsim = 2
for i in range(nsim):
	L.info("Filtering sim %d" % i)
	r = enmap.rand_gauss(map.shape, map.wcs)*(inoise+np.max(inoise)*1e-4)[0]**-0.5
	c = enmap.rand_map(map.shape, map.wcs, ps_cmb)
	sim = r+c
	sim_flt = trifilter.solve(sim, verbose=args.verbose, nmin=nmin)
	enmap.write_map(args.odir+"/sim%d.fits" % i, sim[0])
	enmap.write_map(args.odir+"/sim_flt%d.fits" % i, sim_flt[0])
	varmap += sim_flt[None,:]*sim_flt[:,None]
varmap /= nsim

L.info("Writing pstd_raw")
enmap.write_map(args.odir+"/pstd_raw.fits", varmap[0,0]**0.5)

# Smooth var map to reduce noise
L.info("Computing significance")
varmap = enmap.smooth_gauss(varmap[0,0], 15*np.pi/180/60)[None,None]
# Create signal-to-noise map
significance = pmap[0]/varmap[0,0]**0.5

L.info("Writing pstd, sigma")
enmap.write_map(args.odir+"/pstd.fits",varmap[0,0]**0.5)
enmap.write_map(args.odir+"/sigma.fits",significance)

# Mask out strong sources
# For some reason the significance calculation is way off, and what works
# for one patch does not work for another.
lim = args.masklim
while True:
	mask = ndimage.distance_transform_edt(np.abs(significance)<lim) > args.rmask
	labels, nlabel = ndimage.label(1-mask)
	print "nlabel: %4d lim: %5.1f" % (nlabel, lim)
	if True: break
	if nlabel < args.nmax: break
	lim *= 1.01

# Create a constrained realization of CMB in holes
L.info("Subtracting cmb")
inoise_masked = inoise*mask
iS = enmap.spec2flat(map.shape, map.wcs, ps_cmb, -1.0)
cmb_sampler = gibbs.FieldSampler(iS, inoise_masked, data=map)
cmb = cmb_sampler.wiener(verbose=args.verbose)

L.info("Writing cmb, nocmb")
enmap.write_map(args.odir+"/cmb.fits", cmb[0])
enmap.write_map(args.odir+"/nocmb.fits", (map-cmb)[0])

# Then detect point sources in the new CMB-free map
L.info("Filtering cmb-free map")
trifilter = Trifilter(map, iP, S*1e-6, N)
pmap2 = trifilter.solve(map-cmb, verbose=args.verbose, nmin=nmin)
significance2 = pmap2[0]/varmap[0,0]**0.5
L.info("Writing pmap2, sigmap2")
enmap.write_map(args.odir+"/pmap2.fits", pmap2[0])
enmap.write_map(args.odir+"/sigma2.fits", significance2)

# Mask significant areas and collect. Do it separately for
# positive and negative excursions, to catch clusters very
# close to point sources.
def amax(a,b): return a if np.abs(a) > np.abs(b) else b
def find_objs(pmap2, sigmap, lim):
	mask = ndimage.distance_transform_edt(sigmap/lim < 1) < args.mindist
	labels, nlabel = ndimage.label(mask)
	objs = ndimage.find_objects(labels)
	amps = np.array([amax(np.max(pmap2[0][o]),np.min(pmap2[0][o])) for o in objs])
	sigs = np.array([amax(np.max(sigmap[o]),np.min(sigmap[o])) for o in objs])
	poss_pix = np.array(ndimage.center_of_mass(sigmap**2, labels, range(1,nlabel+1)))
	poss = pmap2.pix2sky(poss_pix.T).T
	return amps, sigs, poss, poss_pix

L.info("Finding objects")
amps_pos, sigs_pos, poss_pos, poss_pix_pos = find_objs(pmap2, significance2, args.siglim)
amps_neg, sigs_neg, poss_neg, poss_pix_neg = find_objs(pmap2, significance2,-args.siglim)
# Combine
amps = np.concatenate([amps_pos,amps_neg])
sigs = np.concatenate([sigs_pos,sigs_neg])
poss = np.concatenate([poss_pos,poss_neg])
poss_pix = np.concatenate([poss_pix_pos,poss_pix_neg])

# Order by amplitude
order = np.argsort(np.abs(amps))[::-1]
amps, sigs, poss, poss_pix = amps[order], sigs[order], poss[order], poss_pix[order]

# And output our candidate information
L.info("Writing objects")
with open(args.odir+"/pos.txt","w") as ofile:
	for amp, sig, pos, pos_pix in zip(amps, sigs, poss, poss_pix):
		ofile.write((" %12.4f"*6+"\n") % (pos[0]*180/np.pi,pos[1]*180/np.pi,sig,pos_pix[0],pos_pix[1],amp))
