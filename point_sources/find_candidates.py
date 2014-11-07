import numpy as np, argparse, sys, itertools, os, errno
from mpi4py import MPI
from enlib import enmap, powspec, utils, fft, array_ops, gibbs
from enlib.degrees_of_freedom import DOF
from enlib.cg import CG
from scipy import ndimage
parser = argparse.ArgumentParser()
parser.add_argument("map")
parser.add_argument("noise")
parser.add_argument("powspec")
parser.add_argument("odir")
parser.add_argument("--srcrms", type=float, default=5)
parser.add_argument("-b", "--beam", type=float, default=1.2)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--rmask", type=int, default=5)
parser.add_argument("--siglim", type=float, default=3.5)
parser.add_argument("--masklim", type=float, default=6)
parser.add_argument("--mindist", type=int, default=2)
parser.add_argument("--nmax", type=int, default=350)
parser.add_argument("--noise-max", type=float, default=1e5)
args = parser.parse_args()

utils.mkdir(args.odir)

map    = enmap.read_map(args.map)
map    = map.reshape((-1,)+map.shape[-2:])[0][None]
inoise  = enmap.read_map(args.noise)
inoise  = inoise.reshape((-1,)+inoise.shape[-2:])[0][None,None]
shape_orig = map.shape[-2:]
# Mask out too noisy areas
mask = inoise[0,0] < args.noise_max**-2
map[:,mask] = np.nan
map, info = enmap.autocrop(map, method="fft", return_info=True)
inoise = enmap.padcrop(inoise, info)
map[np.isnan(map)] = 0
shape_crop = map.shape[-2:]
print "Cropped from %s to %s" % (str(shape_orig),str(shape_crop))

## Truncate to fft-friendly areas
#h = fft.fft_len(map.shape[-2],[2,3,5])
#w = fft.fft_len(map.shape[-1],[2,3,5])
#map    = map[:,:h,:w]
#inoise = inoise[:,:,:h,:w]
#map[~np.isfinite(map)] = 0

# Kill extreme values
map = np.minimum(1e6,np.maximum(-1e6,map))

nmin = 50
#cut = 0x100
#map = map[:,h/2-cut:h/2+cut,w/2-cut:w/2+cut]
#inoise = inoise[:,:,h/2-cut:h/2+cut,w/2-cut:w/2+cut]
#map = map[:,250:750,700:1700]
#inoise = inoise[:,:,250:750,700:1700]

enmap.write_map(args.odir + "/map.fits", map[0])

ps_cmb_tmp = powspec.read_spectrum(args.powspec, expand="diag")[:1,:1,:]
# Extend to max l
lmax     = np.max(np.sum(map.lmap()**2,0)**0.5).astype(int)+1
lmax_tmp = min(ps_cmb_tmp.shape[-1],lmax)
ps_cmb = np.zeros([1,1,lmax])
ps_cmb[:,:,:lmax_tmp] = ps_cmb_tmp[:,:,:lmax_tmp]
ps_cmb[:,:,lmax_tmp:] = ps_cmb_tmp[:,:,-1]

sigma  = args.beam * np.pi/180/60/(8*np.log(2))
l      = np.arange(ps_cmb.shape[-1])
ps_src = np.exp(-l*(l+1)*sigma**2)[None,None,:]*(args.srcrms*np.pi/180/60)**2

S  = enmap.spec2flat(map.shape, map.wcs, ps_cmb, 1.0)
iP = enmap.spec2flat(map.shape, map.wcs, ps_src,-1.0)
N  = (inoise + np.max(inoise)*1e-3)**-1

# apodize map based on a smooth noise map
inoise_smooth = enmap.smooth_gauss(inoise,10*np.pi/180/60)
apod = (np.minimum(1,inoise_smooth/(np.max(inoise_smooth)*0.1))**4)[0,0]
map *= apod[None]

def mul(mat, vec, axes=[0,1]):
	return array_ops.matmul(mat.astype(vec.dtype),vec,axes=axes)
def pow(mat, exp, axes=[0,1]): return array_ops.eigpow(mat,exp,axes=axes)

class Trifilter:
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
		iPx   = mul(self.iP,enmap.map2harm(xmap))
		SiPx  = enmap.harm2map(mul(self.S, iPx))
		NiPx  = mul(self.noise,enmap.harm2map(iPx))
		return xmap + SiPx + NiPx
	def M(self, xmap):
		return enmap.harm2map(mul(self.prec,enmap.map2harm(xmap)))
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
print "Filtering map"
pmap = trifilter.solve(map, verbose=args.verbose, nmin=nmin)

# Estimate uncertainty with a couple of other realizations
varmap = inoise*0
nsim = 2
for i in range(nsim):
	print "Filtering sim %d" % i
	r = enmap.rand_gauss(map.shape, map.wcs)*(inoise+np.max(inoise)*1e-4)[0]**-0.5
	c = enmap.rand_map(map.shape, map.wcs, ps_cmb)
	sim = r+c
	sim_flt = trifilter.solve(sim, verbose=args.verbose, nmin=nmin)
	enmap.write_map(args.odir+"/sim%d.fits" % i, sim[0])
	enmap.write_map(args.odir+"/sim_flt%d.fits" % i, sim_flt[0])
	varmap += sim_flt[None,:]*sim_flt[:,None]
varmap /= nsim

enmap.write_map(args.odir+"/pstd_raw.fits", varmap[0,0]**0.5)

# Smooth var map to reduce noise
varmap = enmap.smooth_gauss(varmap, 15*np.pi/180/60)
# Create signal-to-noise map
significance = pmap[0]/varmap[0,0]**0.5

enmap.write_map(args.odir+"/pmap.fits",pmap[0])
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
print "Subtracting cmb"
inoise_masked = inoise*mask
iS = enmap.spec2flat(map.shape, map.wcs, ps_cmb, -1.0)
cmb_sampler = gibbs.FieldSampler(iS, inoise_masked, data=map)
cmb = cmb_sampler.wiener(verbose=args.verbose)

enmap.write_map(args.odir+"/cmb.fits", cmb[0])
enmap.write_map(args.odir+"/nocmb.fits", (map-cmb)[0])

# Then detect point sources in the new CMB-free map
print "Filtering cmb-free map"
trifilter = Trifilter(map, iP, S*1e-6, N)
pmap2 = trifilter.solve(map-cmb, verbose=args.verbose, nmin=nmin)
significance2 = pmap2[0]/varmap[0,0]**0.5
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
with open(args.odir+"/pos.txt","w") as ofile:
	for amp, sig, pos, pos_pix in zip(amps, sigs, poss, poss_pix):
		ofile.write((" %12.4f"*6+"\n") % (pos[0]*180/np.pi,pos[1]*180/np.pi,sig,pos_pix[0],pos_pix[1],amp))
