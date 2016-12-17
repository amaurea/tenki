"""This program reads in a set of pickup submaps for the same
scanning pattern and attempts to coadd them in a way that maximizes
the S/N of stationary features."""
import numpy as np, argparse, h5py
from enlib import enmap, fft, utils, cg, zipper, mpi, bunch
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("template")
parser.add_argument("odir")
parser.add_argument("-n", "--nmax",     type=int,   default=0)
parser.add_argument("-F", "--fiducial", action="store_true")
parser.add_argument("-H", "--highpass", type=float, default=0, help="Highpass at this wavelength in pixels")
parser.add_argument("--nompi", action="store_true")
args  = parser.parse_args()

bsize  = 10
nrow, ncol = 33, 32
comm = mpi.COMM_WORLD if not args.nompi else mpi.COMM_SELF
ifiles= args.ifiles
if args.nmax: ifiles = ifiles[:args.nmax]

def window(m, w=50):
	m = m.copy()
	a = (1-np.cos(np.linspace(0,np.pi,w)))/2
	m[...,:w] *= a
	m[...,-w:] *= a[::-1]
	return m

def apply_highpass(m, pix):
	fm = fft.rfft(m)
	fm[...,:fm.shape[-1]/pix] = 0
	return fft.ifft(fm, m, normalize=True)

def estimate_nmat1(diff):
	# Give zero weight to tods with too many empty
	# columns
	zfrac  = float(np.sum(np.all(diff==0,0)))/diff.shape[1]
	diff   = window(diff)
	fdiff  = fft.rfft(diff)
	power  = np.abs(fdiff)**2
	Nmat   = 1/utils.block_mean_filter(power, bsize)
	#Nmat   = power.copy()
	#for i, p in enumerate(power):
	#	Nmat[i] = 1/utils.block_mean_filter(p, bsize)
	#	Nmat[i][~np.isfinite(Nmat[i])] = 0
	#	if np.any(Nmat[i] > 0.01):
	#			print "A", np.where(Nmat[i]>0.01)
	#			print "B", np.max(Nmat[i])
	#			j = np.where(Nmat[i]>0.01)[0][0]
	#			print "C", Nmat[i,j-1:j+2]
	#			print "D", p[j-15:]
	#			1/0
	Nmat[~np.isfinite(Nmat)] = 0
	if zfrac > 0.02: Nmat[:] = 0
	#print np.min(Nmat[Nmat!=0]), np.max(Nmat[Nmat!=0])
	return Nmat

def V(x):
	res = np.zeros((ncol,nrow,x.shape[-1]),x.dtype)
	res[:] = x[:,None,:]
	return res.reshape(nrow*ncol,-1)
def VT(x):
	return np.sum(x.reshape(ncol,nrow,-1),1)

def apply_nmat1(Nmat, vec):
	res = vec*0
	fft.irfft(Nmat*fft.rfft(window(vec)), res, normalize=True)
	return window(res)

def get_slice(m, template):
	corner = np.floor(template.sky2pix(m.pix2sky([0,0]))).astype(int)
	rout = [max(0,corner[1]),min(template.shape[-1],corner[1]+m.shape[-1])]
	rin  = [0,rout[1]-rout[0]]
	pslice = (Ellipsis,slice(None),slice(rout[0],rout[1]))
	m = m[...,rin[0]:rin[1]]
	return m, pslice

estimate_nmat = estimate_nmat1
apply_nmat    = apply_nmat1

template = enmap.read_map(args.template)
while template.ndim > 2: template = template[0]

# Separate out the mpi tasks that didn't receive any files
comm_work = comm.Split(comm.rank < len(ifiles), comm.rank if comm.rank < len(ifiles) else comm.rank - len(ifiles))
if comm.rank < len(ifiles):
	rhs = template*0
	data = []
	for ind in range(comm_work.rank, len(ifiles), comm_work.size):
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
		# Apply highpass-filter if requested
		if args.highpass:
			apply_highpass(mavg, args.highpass)
		rhs[pslice] += apply_nmat(Nmat, mavg)
		data.append(bunch.Bunch(Nmat=Nmat, pslice=pslice))

	# Get the total rhs
	rhs = utils.allreduce(rhs, comm_work)

	# Set up our degrees of freedom
	dof = zipper.ArrayZipper(rhs.copy())

	def A(x):
		map = dof.unzip(x)
		res = map*0
		for d in data:
			locmap = map[d.pslice]
			locmap = apply_nmat(d.Nmat, locmap)
			res[d.pslice] += locmap
		return dof.zip(utils.allreduce(res, comm_work))

	utils.mkdir(args.odir)

	solver = cg.CG(A, dof.zip(rhs))
	for i in range(100):
		solver.step()
		if comm_work.rank == 0:
			#if solver.i % 20 == 0:
			#	map = dof.unzip(solver.x)
			#	enmap.write_map(args.odir + "/map%03d.fits" % solver.i, map)
			print "%4d %15.7e" % (solver.i, solver.err)

if comm.rank == 0:
	enmap.write_map(args.odir + "/map.fits", dof.unzip(solver.x))
