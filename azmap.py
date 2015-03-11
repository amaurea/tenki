"""Create maps in azimuth phase, in order to investigate magnetic pickup. We assume
that relative azimuth is all that matters, so time, elevation and azimuth center
can be ignored."""
import numpy as np, mpi4py.MPI, sys, os, time, h5py
from enact import filedb, data
from enlib import enmap, config, utils, array_ops, fft, errors, gapfill, rangelist
from scipy.interpolate import UnivariateSpline

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
my_arhs, tot_arhs = np.zeros([2,3,nbin])
my_adiv, tot_adiv = np.zeros([2,3,3,nbin])
my_prhs, tot_prhs = np.zeros([2,3,nbin])
my_pdiv, tot_pdiv = np.zeros([2,3,3,nbin])
my_acc_arhs, tot_acc_arhs = np.zeros([2,nbin])
my_acc_adiv, tot_acc_adiv = np.zeros([2,nbin])
my_acc_prhs, tot_acc_prhs = np.zeros([2,nbin])
my_acc_pdiv, tot_acc_pdiv = np.zeros([2,nbin])

def output_cum(si):
	comm.Allreduce(my_arhs, tot_arhs)
	comm.Allreduce(my_adiv, tot_adiv)
	comm.Allreduce(my_prhs, tot_prhs)
	comm.Allreduce(my_pdiv, tot_pdiv)
	comm.Allreduce(my_acc_arhs, tot_acc_arhs)
	comm.Allreduce(my_acc_adiv, tot_acc_adiv)
	comm.Allreduce(my_acc_prhs, tot_acc_prhs)
	comm.Allreduce(my_acc_pdiv, tot_acc_pdiv)
	if comm.rank == 0:
		tot_asig  = array_ops.solve_multi(tot_adiv, tot_arhs, axes=[0,1])
		tot_psig  = array_ops.solve_multi(tot_pdiv, tot_prhs, axes=[0,1])
		tot_acc_asig = tot_acc_arhs/tot_acc_adiv
		tot_acc_psig = tot_acc_prhs/tot_acc_pdiv
		with h5py.File(args.odir + "/cum%04d.hdf" % si, "w") as hfile:
			hfile["arhs"] = tot_arhs
			hfile["adiv"] = tot_adiv
			hfile["asig"] = tot_asig
			hfile["prhs"] = tot_prhs
			hfile["pdiv"] = tot_pdiv
			hfile["psig"] = tot_psig
			hfile["acc/arhs"] = tot_acc_arhs
			hfile["acc/adiv"] = tot_acc_adiv
			hfile["acc/asig"] = tot_acc_asig
			hfile["acc/prhs"] = tot_acc_prhs
			hfile["acc/pdiv"] = tot_acc_pdiv
			hfile["acc/psig"] = tot_acc_psig
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

	if args.filter: d.tod = highpass(d.tod, args.filter)

	# Project whole time-stream to TQU
	comps = np.zeros([3,ndet])
	comps[0] = 1
	comps[1] = np.cos(+2*d.polangle)
	comps[2] = np.sin(-2*d.polangle)
	polrhs = comps.dot(d.tod*weight)
	poldiv = np.einsum("ad,bd,di->abi",comps,comps,weight)
	poltod = array_ops.solve_multi(poldiv,polrhs,axes=[0,1])
	del weight
	print "Computing az bin"
	az    = d.boresight[1]
	def build_bins(a, nbin):
		box = np.array([np.min(a),np.max(a)])
		return np.minimum(np.floor((a-box[0])/(box[1]-box[0])*nbin).astype(int),nbin-1)
	def bin_by_pix(a, pix, nbin):
		a = np.asarray(a)
		if a.ndim == 0: a = np.full(len(pix), a)
		fa = a.reshape(-1,a.shape[-1])
		fo = np.zeros([fa.shape[0],nbin])
		for i in range(len(fa)):
			fo[i] = np.bincount(pix, fa[i], minlength=nbin)
		return fo.reshape(a.shape[:-1]+(nbin,))
	# Gapfill poltod in regions with far too few hits, to
	# avoid messing up the poltod power spectrum
	mask = poldiv[0,0] < np.mean(poldiv[0,0])*0.1
	for i in range(poltod.shape[0]):
		poltod[i] = gapfill.gapfill_copy(poltod[i], rangelist.Rangelist(mask))

	# Calc phase which is equal to az while az velocity is positive
	# and 2*max(az) - az while az velocity is negative
	x = np.arange(len(az))
	az_spline = UnivariateSpline(x, az, s=1e-4)
	daz  = az_spline.derivative(1)(x)
	ddaz = az_spline.derivative(2)(x)
	phase = az.copy()
	phase[daz<0] = 2*np.max(az)-az[daz<0]
	# Bin by az
	apix = build_bins(az, nbin)
	arhs = bin_by_pix(polrhs, apix, nbin)
	adiv = bin_by_pix(poldiv, apix, nbin)
	asig = array_ops.solve_multi(adiv, arhs, axes=[0,1])
	aacc_rhs = bin_by_pix(ddaz, apix, nbin)
	aacc_div = bin_by_pix(1, apix, nbin)
	aacc_sig = aacc_rhs/aacc_div
	# Bin by phase
	ppix = build_bins(phase, nbin)
	prhs = bin_by_pix(polrhs, ppix, nbin)
	pdiv = bin_by_pix(poldiv, ppix, nbin)
	psig = array_ops.solve_multi(pdiv, prhs, axes=[0,1])
	pacc_rhs = bin_by_pix(ddaz, ppix, nbin)
	pacc_div = bin_by_pix(1, ppix, nbin)
	pacc_sig = pacc_rhs/pacc_div
	with h5py.File(args.odir + "/"+ entry.id, "w") as hfile:
		hfile["poltod"] = poltod
		hfile["polrhs"] = polrhs
		hfile["poldiv"] = poldiv
		hfile["az"]    = az
		hfile["asig"] = asig
		hfile["arhs"] = arhs
		hfile["adiv"] = adiv
		hfile["aacc_sig"] = aacc_sig
		hfile["aacc_rhs"] = aacc_rhs
		hfile["aacc_div"] = aacc_div
		hfile["phase"] = phase
		hfile["psig"] = psig
		hfile["prhs"] = prhs
		hfile["pdiv"] = pdiv
		hfile["pacc_sig"] = pacc_sig
		hfile["pacc_rhs"] = pacc_rhs
		hfile["pacc_div"] = pacc_div
	# Update cumulative
	my_arhs += arhs
	my_adiv += adiv
	my_prhs += prhs
	my_pdiv += pdiv
	my_acc_arhs += aacc_rhs
	my_acc_adiv += aacc_div
	my_acc_prhs += pacc_rhs
	my_acc_pdiv += pacc_div

	output_cum(si)
