"""Create maps in azimuth phase, in order to investigate magnetic pickup. We assume
that relative azimuth is all that matters, so time, elevation and azimuth center
can be ignored."""
import numpy as np, mpi4py.MPI, sys, os, time, h5py
from enact import filedb, data
from enlib import enmap, config, utils, array_ops, fft, errors

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("selector")
parser.add_argument("odir")
parser.add_argument("-f", "--filter", type=float, default=0)
args = parser.parse_args()

utils.mkdir(args.odir)

comm = mpi4py.MPI.COMM_WORLD
ids  = filedb.scans[args.selector].fields["id"]
nbin = 300

def white_est(tod): return np.std(tod[:,1:]-tod[:,:-1],1)
def highpass(tod, f, srate=400):
	tod = tod.copy()
	ft = fft.rfft(tod)
	ft[:,:int(f/float(srate)*tod.shape[1])] = 0
	fft.ifft(ft, tod, normalize=True)
	return tod

# cumulative az bin
crhs = np.zeros([3,nbin])
cdiv = np.zeros([3,3,nbin])
trhs, tdiv = crhs.copy(), cdiv.copy()

def output_cum(si):
	comm.Allreduce(crhs, trhs)
	comm.Allreduce(cdiv, tdiv)
	if comm.rank == 0:
		tsig  = array_ops.solve_multi(tdiv, trhs, axes=[0,1])
		with h5py.File(args.odir + "/cum%04d.hdf" % si, "w") as hfile:
			hfile["crhs"] = trhs
			hfile["cdiv"] = tdiv
			hfile["csig"] = tsig

for si in range(comm.rank, len(ids), comm.size):
	entry = filedb.data[ids[si]]
	print "Reading %s" % entry.id
	try:
		d = data.read(entry, fields=["gain","polangle","tconst","boresight","cut","tod"])
		d = data.calibrate(d)
	except (IOError,errors.DataMissing) as e:
		print "Skipping [%s]" % e.message
		output_cum(si)
		continue
	print "Computing pol tod"
	ndet, nsamp = d.tod.shape
	# Estimate white noise level (rough)
	rms = white_est(d.tod[:,:50000])
	weight = (1-d.cut.to_mask())/rms[:,None]**2
	# Project whole time-stream to TQU
	comps = np.zeros([3,ndet])
	comps[0] = 1
	comps[1] = np.cos(+2*d.polangle)
	comps[2] = np.sin(-2*d.polangle)
	polrhs = comps.dot(d.tod*weight)
	if args.filter:
		polrhs = highpass(polrhs, args.filter)
	poldiv = np.einsum("ad,bd,di->abi",comps,comps,weight)
	poltod = array_ops.solve_multi(poldiv,polrhs,axes=[0,1])
	# Bin by az
	print "Computing az bin"
	az    = d.boresight[1]
	abox  = np.array([np.min(az),np.max(az)])
	pix   = np.minimum(np.floor((az-abox[0])/(abox[1]-abox[0])*nbin).astype(int),nbin-1)
	arhs  = np.array([np.bincount(pix, r, minlength=nbin) for r in polrhs])
	adiv  = np.zeros([3,3,nbin])
	for i in range(3):
		for j in range(3):
			adiv[i,j] = np.bincount(pix, poldiv[i,j], minlength=nbin)
	asig  = array_ops.solve_multi(adiv, arhs, axes=[0,1])
	with h5py.File(args.odir + "/"+ entry.id, "w") as hfile:
		hfile["poltod"] = poltod
		hfile["polrhs"] = polrhs
		hfile["poldiv"] = poldiv
		hfile["asig"] = asig
		hfile["arhs"] = arhs
		hfile["adiv"] = adiv
		hfile["az"] = az
		hfile["pix"] = pix
	# Update cumulative
	crhs += arhs
	cdiv += adiv

	output_cum(si)
