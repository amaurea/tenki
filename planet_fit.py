import numpy as np, argparse, sys, os
from enlib import enmap, utils
from scipy import optimize
#from matplotlib.pylab import *
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+", help="map map ... div div ...")
parser.add_argument("odir")
parser.add_argument("--slice",           type=str,   default="")
parser.add_argument("-d", "--downgrade", type=int,   default=1)
parser.add_argument("-T", "--transpose", action="store_true")
parser.add_argument("-L", "--ref-nloop", type=int,   default=2)
args = parser.parse_args()

ref_nloop = args.ref_nloop
utils.mkdir(args.odir)
nfile = len(args.ifiles)/2
if not args.transpose:
	mapfiles = args.ifiles[:nfile]
	divfiles = args.ifiles[nfile:]
else:
	mapfiles = args.ifiles[0::2]
	divfiles = args.ifiles[1::2]

def read_map(fname):
	m = enmap.read_map(fname)
	m = m[...,1:-1,1:-1]
	#m = eval("m" + args.slice)
	m = enmap.downgrade(m, args.downgrade)
	return m

def apply_params(maps, params, inverse=False, nooff=False):
	"""Given maps[nmap,...,ny,nx] and params[nmap,{dy,dx,A,o}],
	return maps shifted, offset and scaled by those parameters."""
	omaps = maps.copy()
	shape = omaps.shape
	omaps = omaps.reshape((-1,)+shape[-3:])
	params= np.array(params)
	params= params.reshape(-1,params.shape[-1])
	for i in range(omaps.shape[0]):
		dy,dx,A = params[i,:3]
		dy = int(np.round(dy))
		dx = int(np.round(dx))
		o = params[i,3:]
		if inverse:
			omaps[i]  = np.roll(omaps[i], dy, -2)
			omaps[i]  = np.roll(omaps[i], dx, -1)
			omaps[i] *= A
			if not nooff:
				omaps[i] += o[:,None,None]
		else:
			if not nooff:
				omaps[i] -= o[:,None,None]
			omaps[i] /= A
			omaps[i]  = np.roll(omaps[i],-dx, -1)
			omaps[i]  = np.roll(omaps[i],-dy, -2)
	omaps = omaps.reshape(shape)
	return omaps

def solve(div, rhs):
	if rhs.ndim == 3: return rhs/div
	elif rhs.ndim == 4: return rhs/div[:,None]
	raise NotImplementedError

class SingleFitter:
	def __init__(self, ref, map, div):
		self.ref = ref
		self.map = map
		self.div = div
		self.ncomp = ref.shape[0]
		self.verbose = False
		self.i = 0
	def calc_chisq(self, params):
		model    = apply_params(self.ref, params, inverse=True)
		residual = self.map-model
		chisq    = np.sum(residual**2*self.div)
		if self.verbose:
			print ("%4d" + " %9.4f"*params.size + " %15.7e") % (
				(self.i,)+tuple(params)+(chisq,))
		self.i += 1
		return chisq
	def fit(self, verbose=False):
		self.verbose = verbose
		self.i = 0
		params = np.zeros([3+self.ncomp])
		params[2] = 1
		params = optimize.fmin_powell(self.calc_chisq, params, disp=False)
		return params

# Ok, read in the maps
maps, divs, ids = [], [], []
for	i, (rfile,dfile) in enumerate(zip(mapfiles, divfiles)):
	print "Reading %s" % rfile
	map = read_map(rfile)
	print "Reading %s" % dfile
	div = read_map(dfile)[0,0]
	maps.append(map)
	divs.append(div)
	ids.append(os.path.basename(rfile)[:-9])
maps = enmap.samewcs(np.asarray(maps),maps[0])
divs = enmap.samewcs(np.asarray(divs),divs[0])
rhss = divs[:,None]*maps
nmap = maps.shape[0]
ncomp= maps.shape[1]

ref = maps[0]
# Choose a reference map. Fit all maps to it. Coadd to find
# new reference map. Repeat.
for ri in range(ref_nloop):
	print "Loop %d" % ri
	params = []
	for mi in range(nmap):
		print "Map %2d" % mi
		ref_small = eval("ref"+args.slice)
		map_small = eval("maps[mi]"+args.slice)
		div_small = eval("divs[mi]"+args.slice)
		#fitter = SingleFitter(ref, maps[mi], divs[mi,:1,:1])
		fitter = SingleFitter(ref_small, map_small, div_small)
		p = fitter.fit(verbose=True)
		params.append(p)
	params = np.array(params)
	mrhss = apply_params(rhss, params, nooff=True)
	mdivs = apply_params(np.tile(divs[:,None],[1,rhss.shape[-3],1,1]), params, nooff=True)[:,0]
	mrhs = np.sum(mrhss,0)
	mdiv = np.sum(mdivs,0)
	ref  = solve(mdiv, mrhs)

	with open(args.odir + "/fit_%03d.txt"%ri, "w") as f:
		for id, p in zip(ids,params):
			f.write(("%7.4f %7.4f %7.4f" + " %12.4f"*(params.shape[-1]-3) + " %s\n") %
					(tuple(p)+(id,)))
	enmap.write_map(args.odir + "/model_%03d.fits" % ri, ref)

	# Output the individual best fits too
	smaps = apply_params(maps, params)
	for id, m in zip(ids, smaps):
		enmap.write_map(args.odir + "/%s_map_%03d.fits" % (id,ri), m)
		enmap.write_map(args.odir + "/%s_resid_%03d.fits" % (id,ri), m-ref)
