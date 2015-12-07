import numpy as np, argparse, bunch, os, time, sys
from enlib import pmat, config, utils, interpol, coordinates, bench
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("--el1", type=float, default=15)
parser.add_argument("--el2", type=float, default=85)
parser.add_argument("--del", type=float, default=5, dest="delta_el")
parser.add_argument("--az1", type=float, default=0)
parser.add_argument("--az2", type=float, default=180)
parser.add_argument("--daz", type=float, default=10)
parser.add_argument("--t",   type=float, default=55000)
parser.add_argument("--wt",  type=float, default=15)
parser.add_argument("--waz", type=float, default=20)
parser.add_argument("--wel", type=float, default=2)
parser.add_argument("--ntest", type=int, default=1000)
args = parser.parse_args()

# Hardcode an arbitrary site
site = bunch.Bunch(
	lat  = -22.9585,
	lon  = -67.7876,
	alt  = 5188.,
	T    = 273.15,
	P    = 550.,
	hum  = 0.2,
	freq = 150.,
	lapse= 0.0065)

acc = config.get("pmat_accuracy")
max_size = config.get("pmat_interpol_max_size")
max_time = config.get("pmat_interpol_max_time")

nel = int(np.rint((args.el2-args.el1)/args.delta_el))+1
naz = int(np.rint((args.az2-args.az1)/args.daz))+1

def hor2cel(hor):
	shape = hor.shape[1:]
	hor = hor.reshape(hor.shape[0],-1)
	tmp = coordinates.transform("hor", "cel", hor[1:], time=hor[0], site=site, pol=True)
	res = np.zeros((4,)+tmp.shape[1:])
	res[0] = utils.rewind(tmp[0], tmp[0,0])
	res[1] = tmp[1]
	res[2] = np.cos(2*tmp[2])
	res[3] = np.sin(2*tmp[2])
	res = res.reshape(res.shape[:1]+shape)
	return res

def eval_ipol(ipol, hor):
	n = hor.shape[1]
	dtype = hor.dtype
	pix   = np.zeros((1,n,2),dtype=dtype)
	phase = np.zeros((1,n,3),dtype=dtype)
	det_pos = np.zeros((1,3))
	det_comps = np.array([[1,1,0]])
	comps = np.zeros(1)
	rbox, nbox, ys = pmat.extract_interpol_params(ipol, dtype)
	pmat.get_core(dtype).translate(
		hor, pix.T, phase.T,
		det_pos.T, det_comps.T,
		comps,
		rbox.T, nbox, ys.T)
	return np.concatenate([pix[0].T,phase[0,:,1:].T],0)

for iaz in range(naz):
	for iel in range(nel):
		az_mid = (args.az1 + args.daz*iaz)*utils.degree
		el_mid = (args.el1 + args.delta_el*iel)*utils.degree
		t_mid  = args.t
		# Bounds
		box = np.array([
			[t_mid -args.wt/2./60/24,  t_mid +args.wt/2./60/24],
			[az_mid-utils.degree*args.waz/2, az_mid+utils.degree*args.waz/2],
			[el_mid-utils.degree*args.wel/2, el_mid+utils.degree*args.wel/2]]).T
		# Build an interpolator
		wbox = utils.widen_box(box)
		errlim = np.array([utils.arcsec, utils.arcsec, utils.arcmin, utils.arcmin])*acc
		t1 = time.time()
		ipol, obox, ok, err = interpol.build(hor2cel, interpol.ip_grad, wbox, errlim,
			maxsize=max_size, maxtime=max_time, return_obox=True, return_status=True)
		t2 = time.time()
		# Choose some points to evaluate the model at
		hor_test = box[0,:,None] + np.random.uniform(0,1,size=(3,args.ntest))*(box[1]-box[0])[:,None]
		cel_exact = hor2cel(hor_test)
		cel_interpol = ipol(hor_test)
		cel_interpol2 = eval_ipol(ipol, hor_test)

		diff = np.max(np.abs(cel_interpol-cel_exact),1)
		print (" %9.4f"*(2+4+1)) % (
				(az_mid/utils.degree,el_mid/utils.degree) +
				tuple(diff/errlim) + (t2-t1,))
		sys.stdout.flush()
