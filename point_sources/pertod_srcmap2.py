import numpy as np, sys, os
from enlib import utils
with utils.nowarn(): import h5py
from enlib import config, pmat, mpi, errors, gapfill, enmap, bench
from enlib import fft, array_ops, sampcut, cg
from enact import filedb, actscan, actdata, cuts, nmat_measure
config.set("pmat_cut_type",  "full")
parser = config.ArgumentParser()
parser.add_argument("sel")
parser.add_argument("srcs")
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("tag", nargs="?")
parser.add_argument("-R", "--dist", type=float, default=4)
parser.add_argument("-y", "--ypad", type=float, default=3)
parser.add_argument("-s", "--src",  type=int,   default=None, help="Only analyze given source")
parser.add_argument("-c", "--cont", action="store_true")
parser.add_argument("-m", "--model",type=str, default="joneig")
args = parser.parse_args()

comm = mpi.COMM_WORLD
filedb.init()
R    = args.dist * utils.arcmin
ypad = args.ypad * utils.arcmin
csize= 100
config.set("pmat_ptsrc_cell_res", 2*(R+ypad)/utils.arcmin)
config.set("pmat_interpol_pad", 5+ypad/utils.arcmin)

dtype = np.float32
area  = enmap.read_map(args.area).astype(dtype)
ncomp = 3
shape = area.shape[-2:]
utils.mkdir(args.odir)
prefix = args.odir + "/"
if args.tag:  prefix += args.tag + "_"

# Set up a dummy beam that represents our source mask. It will
# go linearly from 1 to 0 at 2R, letting us use 0.5 as the 1R cutoff.
beam = np.array([[0,1],[2*R,0]]).T

# Get the source positions
srcs = np.loadtxt(args.srcs)
src_param = np.zeros((len(srcs),8))
src_param[:,0:2] = srcs[:,1::-1]*utils.degree
src_param[:,2]   = 1
src_param[:,5:7] = 1

# Find out which sources are hit by each tod
db = filedb.scans.select(filedb.scans[args.sel])
tod_srcs = {}
for sid, src in enumerate(srcs):
	if args.src is not None and sid != args.src: continue
	for id in db["hits([%.6f,%6f])" % tuple(src[:2])]:
		if not id in tod_srcs: tod_srcs[id] = []
		tod_srcs[id].append(sid)

# Prune those those that are done
if args.cont:
	good = []
	for id in tod_srcs:
		bid = id.replace(":","_")
		ndone = 0
		for sid in tod_srcs[id]:
			if os.path.exists("%s%s_src%03d_map.fits" % (prefix, bid, sid)) or os.path.exists("%s%s_empty.txt" % (prefix, bid)):
				ndone += 1
		if ndone < len(tod_srcs[id]):
			good.append(id)
	tod_srcs = {id:tod_srcs[id] for id in good}
ids = sorted(tod_srcs.keys())

def smooth(tod, srate, fknee=10, alpha=10):
	ft   = fft.rfft(tod)
	freq = fft.rfftfreq(tod.shape[-1])*srate
	flt  = 1/(1+(freq/fknee)**alpha)
	ft  *= flt
	fft.ifft(ft, tod, normalize=True)
	return tod

def calc_model_joneig(tod, cut, srate=400):
	return smooth(gapfill.gapfill_joneig(tod, cut, inplace=False), srate)

def calc_model_constrained(tod, cut, srate=400, mask_scale=0.3, lim=3e-4, maxiter=50, verbose=False):
	ft = fft.rfft(tod) * tod.shape[1]**-0.5
	iN = nmat_measure.detvecs_jon(ft, srate)
	iV = iN.ivar*mask_scale
	def A(x):
		x   = x.reshape(tod.shape)
		Ax  = iN.apply(x.copy())
		Ax += sampcut.gapfill_const(cut, x*iV[:,None], 0, inplace=True)
		return Ax.reshape(-1)
	b  = sampcut.gapfill_const(cut, tod*iV[:,None], 0, inplace=True).reshape(-1)
	x0 = sampcut.gapfill_linear(cut, tod).reshape(-1)
	solver = cg.CG(A, b, x0)
	while solver.i < maxiter and solver.err > lim:
		solver.step()
		if verbose:
			print "%5d %15.7e" % (solver.i, solver.err)
	return solver.x.reshape(tod.shape)

for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	bid   = id.replace(":","_")
	entry = filedb.data[id]
	# Read the tod as usual
	try:
		with bench.show("read"):
			d = actdata.read(entry)
		with bench.show("calibrate"):
			d = actdata.calibrate(d, exclude=["autocut"])
		# Replace the beam with our dummy beam
		d.beam = beam
		if d.ndet == 0 or d.nsamp < 2: raise errors.DataMissing("no data in tod")
	except errors.DataMissing as e:
		print "Skipping %s (%s)" % (id, e.message)
		# Make a dummy output file so we can skip this tod in the future
		with open("%s%s_empty.txt" % (prefix, bid),"w"): pass
		continue
	print "Processing %s [ndet:%d, nsamp:%d, nsrc:%d]" % (id, d.ndet, d.nsamp, len(tod_srcs[id]))
	# Very simple white noise model. This breaks if the beam has been tod-smoothed by this point.
	with bench.show("ivar"):
		tod  = d.tod
		del d.tod
		tod -= np.mean(tod,1)[:,None]
		tod  = tod.astype(dtype)
		diff = tod[:,2:]-tod[:,:-2]
		diff = diff[:,:diff.shape[-1]/csize*csize].reshape(d.ndet,-1,csize)
		ivar = 1/(np.median(np.mean(diff**2,-1),-1)/2**0.5)
		del diff
	with bench.show("actscan"):
		scan = actscan.ACTScan(entry, d=d)
	with bench.show("pmat1"):
		psrc = pmat.PmatPtsrc(scan, src_param)
		pcut = pmat.PmatCut(scan)
		junk = np.zeros(pcut.njunk, dtype)
	with bench.show("source mask"):
		# Find the samples where the sources live
		src_mask = np.zeros(tod.shape, np.bool)
		# Allow elongating the mask vertically
		nypad = 1
		dypad = R/2
		if ypad > 0:
			nypad = int((2*ypad)//dypad)+1
			dypad = (2*ypad)/(nypad-1)
		# Hack: modify detector offsets to apply the y broadening
		detoff = scan.offsets.copy()
		src_tod = tod*0
		for yi in range(nypad):
			yoff    = -ypad + yi*dypad
			psrc.scan.offsets = detoff.copy()
			psrc.scan.offsets[:,2] -= yoff
			psrc.forward(src_tod, src_param, tmul=0)
			src_mask |= src_tod > 0.5
		del src_tod
		# Undo the hack here
		psrc.scan.offsets = detoff
		src_cut  = sampcut.from_mask(src_mask)
		del src_mask
	with bench.show("atm model"):
		if   args.model == "joneig":
			model = calc_model_joneig(tod, src_cut, d.srate)
		elif args.model == "constrained":
			model = calc_model_constrained(tod, src_cut, d.srate, verbose=True)
	with bench.show("atm subtract"):
		tod -= model
		del model
		tod  = tod.astype(dtype, copy=False)
	# Should now be reasonably clean of correlated noise.
	# Proceed to make simple binned map for each point source. We need a separate
	# pointing matrix for each because each has its own local coordinate system.
	tod *= ivar[:,None]
	pcut.backward(tod, junk)
	for sid in tod_srcs[id]:
		rhs  = enmap.zeros((ncomp,)+shape, area.wcs, dtype)
		div  = enmap.zeros((ncomp,ncomp)+shape, area.wcs, dtype)
		with bench.show("pmat %s" % sid):
			pmap = pmat.PmatMap(scan, area, sys="hor:%.6f_%.6f:cel/0_0:hor" % tuple(srcs[sid,:2]))
		with bench.show("rhs %s" % sid):
			pmap.backward(tod, rhs)
		with bench.show("hits"):
			for i in range(ncomp):
				div[i,i] = 1
				pmap.forward(tod, div[i])
				tod *= ivar[:,None]
				pcut.backward(tod, junk)
				div[i] = 0
				pmap.backward(tod, div[i])
		with bench.show("map %s" % sid):
			idiv = array_ops.eigpow(div, -1, axes=[0,1], lim=1e-5, fallback="scalar")
			map  = enmap.map_mul(idiv, rhs)
		with bench.show("write"):
			enmap.write_map("%s%s_src%03d_map.fits" % (prefix, bid, sid), map)
			enmap.write_map("%s%s_src%03d_rhs.fits" % (prefix, bid, sid), rhs)
			enmap.write_map("%s%s_src%03d_div.fits" % (prefix, bid, sid), div)
		del rhs, div, idiv, map
	del d, scan, pmap, pcut, tod, junk
