import numpy as np, argparse
from enlib import enmap, fft, utils, cg, zipper, mpi, bunch
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("template")
parser.add_argument("odir")
parser.add_argument("-n", "--nmax", type=int, default=0)
parser.add_argument("-F", "--fiducial", action="store_true")
args  = parser.parse_args()

bsize  = 10
nrow, ncol = 33, 32
comm  = mpi.COMM_WORLD
ifiles= args.ifiles
if args.nmax: ifiles = ifiles[:args.nmax]

def window(m, w=50):
	m = m.copy()
	a = (1-np.cos(np.linspace(0,np.pi,w)))/2
	m[...,:w] *= a
	m[...,-w:] *= a[::-1]
	return m

def estimate_nmat1(diff):
	diff   = window(diff)
	fdiff  = fft.rfft(diff)
	power  = np.abs(fdiff)**2
	Nmat   = power.copy()
	for i, p in enumerate(power):
		Nmat[i] = 1/utils.block_mean_filter(p, bsize)
	Nmat[~np.isfinite(Nmat)] = 0
	return Nmat

def V(x):
	res = np.zeros((ncol,nrow,x.shape[-1]),x.dtype)
	res[:] = x[:,None,:]
	return res.reshape(nrow*ncol,-1)
def VT(x):
	return np.sum(x.reshape(ncol,nrow,-1),1)

def estimate_nmat2(diff):
	fdiff  = fft.rfft(diff)
	# Estimate common mode per detector block
	cmode  = np.mean(fdiff.reshape(ncol, nrow, -1),1)
	# Subtract it. Though I worry this might
	# cause trouble when single detectors are weird
	fdiff -= V(cmode)
	# Estimate frequency noise power for the rest
	fpower = np.abs(fdiff)**2
	cpower = np.abs(cmode)**2
	# Block mean to incrase S/N
	for i in range(len(fpower)):
		fpower[i] = 1/utils.block_mean_filter(fpower[i], bsize)
	for i in range(len(cpower)):
		cpower[i] = 1/utils.block_mean_filter(cpower[i], bsize)
	fpower[~np.isfinite(fpower)] = 0
	cpower[~np.isfinite(cpower)] = 0
	# Will need woodbury to apply this
	# Model is N" = (F+VCV')" = F" - F"V(C" + V'F"V)V'F"
	# where F" is fpower and C" is cpower
	return (fpower, cpower)

def estimate_nmat3(diff):
	fdiff  = fft.rfft(diff, axes=(0,1))
	power  = np.abs(fdiff)**2
	Nmat   = power.copy()
	for i, p in enumerate(power):
		Nmat[i] = 1/utils.block_mean_filter(p, bsize)
	Nmat[~np.isfinite(Nmat)] = 0
	return Nmat

def apply_nmat1(Nmat, vec):
	res = vec*0
	fft.irfft(Nmat*fft.rfft(window(vec)), res, normalize=True)
	return window(res)

def apply_nmat2(Nmat, vec):
	F, C = Nmat
	f     = fft.rfft(vec)
	Ff    = F*f
	b     = VT(Ff)
	fres  = Ff - F*V(C*b + VT(F*V(b)))
	res   = vec.copy()
	fft.irfft(fres, resm, normalize=True)
	return res

def apply_nmat3(Nmat, vec):
	res = vec*0
	fft.irfft(Nmat*fft.rfft(vec, axes=(0,1)), res, axes=(0,1), normalize=True)
	return res

def get_slice(m, template):
	corner = np.floor(template.sky2pix(m.pix2sky([0,0]))).astype(int)
	rout = [max(0,corner[1]),min(template.shape[-1],corner[1]+m.shape[-1])]
	rin  = [0,rout[1]-rout[0]]
	pslice = (Ellipsis,slice(None),slice(rout[0],rout[1]))
	m = m[...,rin[0]:rin[1]]
	return m, pslice

if True:
	estimate_nmat = estimate_nmat1
	apply_nmat    = apply_nmat1
else:
	estimate_nmat = estimate_nmat3
	apply_nmat    = apply_nmat3

template = enmap.read_map(args.template)
while template.ndim > 2: template = template[0]

rhs = template*0
data = []
for ind in range(comm.rank, len(ifiles), comm.size):
	ifile = ifiles[ind]
	print ifile
	m = enmap.read_map(ifile)
	m, pslice = get_slice(m, template)
	unhit = ~np.isfinite(np.sum(m,0)) | (np.sum(m,0) == 0)
	m[:,unhit] = 0
	if args.fiducial:
		diff = np.mean(m,0) - template[pslice]
		diff[unhit] = 0
	else:
		diff = m[1]-m[0]
	# Find slice that copies between us and template, assuming
	# they have the same resolution
	Nmat = estimate_nmat(diff)
	# Compute our contribution to the RHS. Since
	# we assume that the two sub-maps differ only
	# by noise, we operate on their mean from now on.
	mavg = np.mean(m,0)
	rhs[pslice] += apply_nmat(Nmat, mavg)
	data.append(bunch.Bunch(Nmat=Nmat, pslice=pslice))

# Get the total rhs
rhs = utils.allreduce(rhs, comm)

# Set up our degrees of freedom
dof = zipper.ArrayZipper(rhs.copy())

def A(x):
	map = dof.unzip(x)
	res = map*0
	for d in data:
		locmap = map[d.pslice]
		locmap = apply_nmat(d.Nmat, locmap)
		res[d.pslice] += locmap
	return dof.zip(utils.allreduce(res, comm))

utils.mkdir(args.odir)

solver = cg.CG(A, dof.zip(rhs))
for i in range(200):
	solver.step()
	if comm.rank == 0:
		if solver.i % 20 == 0:
			map = dof.unzip(solver.x)
			enmap.write_map(args.odir + "/map%03d.fits" % solver.i, map)
		print "%4d %15.7e" % (solver.i, solver.err)

if comm.rank == 0:
	enmap.write_map(args.odir + "/map.fits", dof.unzip(solver.x))
