import numpy as np, sys, os, h5py
from enlib import config, pmat, mpi, errors, gapfill, utils, enmap, bench
from enlib import fft, array_ops
from enact import filedb, actscan, actdata, cuts, nmat_measure

parser = config.ArgumentParser(os.environ["HOME"]+"./enkirc")
parser.add_argument("planet")
parser.add_argument("area")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("tag", nargs="?")
parser.add_argument("-R", "--dist",    type=float, default=0.2)
parser.add_argument("-e", "--equator", action="store_true")
parser.add_argument("-c", "--cont",    action="store_true")
parser.add_argument("-m", "--model",   type=str, default="joneig")
args = parser.parse_args()

comm = mpi.COMM_WORLD
filedb.init()
ids  = filedb.scans[args.sel]
R    = args.dist * utils.degree
csize= 100

dtype= np.float32
area = enmap.read_map(args.area).astype(dtype)
ncomp= 3
shape= area.shape[-2:]
model_fknee = 10
model_alpha = 10
sys = "hor:"+args.planet
if args.equator: sys += "/0_0"
utils.mkdir(args.odir)
prefix = args.odir + "/"
if args.tag: prefix += args.tag + "_"

def gapfill_nmat(tod, cut, srate, nit=4, npass=1, inplace=False, verbose=False):
	tod   = np.asarray(tod)
	if not inplace: tod = tod.copy()
	with bench.show("gapfill linear", verbose):
		gapfill.gapfill_linear(tod, cut)
	for p in range(npass):
		with bench.show("build nmat %d" % p, verbose):
			ft    = fft.rfft(tod) * tod.shape[1]**-0.5
			inmat = nmat_measure.detvecs_jon(ft, srate)
			nmat  = inmat.calc_inverse()
			del ft
		for it in range(nit):
			with bench.show("copy %d %d" % (p, it)):
				work = tod.copy()
			with bench.show("forward %d %d" % (p, it)):
				inmat.apply(work)
			with bench.show("gapfill constant %d %d" % (p, it)):
				gapfill.gapfill_constant(work, cut)
			with bench.show("backward %d %d" % (p, it)):
				nmat.apply(work)
			with bench.show("insert %d %d" % (p, it)):
				cut.insert_sampes(tod, cut.extract_samples(work))
	return tod

for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	bid   = id.replace(":","_")
	entry = filedb.data[id]
	oname = "%s%s_map.fits" % (prefix, bid)
	if args.cont and os.path.isfile(oname):
		print "Skipping %s (already done)" % (id)
		continue
	# Read the tod as usual
	try:
		with bench.show("read"):
			d = actdata.read(entry)
		with bench.show("calibrate"):
			d = actdata.calibrate(d, exclude=["autocut"])
		if d.ndet == 0 or d.nsamp < 2: raise errors.DataMissing("no data in tod")
	except errors.DataMissing as e:
		print "Skipping %s (%s)" % (id, e.message)
		continue
	print "Processing %s" % id
	# Very simple white noise model
	with bench.show("ivar"):
		tod  = d.tod
		del d.tod
		tod -= np.mean(tod,1)[:,None]
		tod  = tod.astype(dtype)
		diff = tod[:,1:]-tod[:,:-1]
		diff = diff[:,:diff.shape[-1]/csize*csize].reshape(d.ndet,-1,csize)
		ivar = 1/(np.median(np.mean(diff**2,-1),-1)/2**0.5)
		del diff
	# Generate planet cut
	with bench.show("planet cut"):
		planet_cut = cuts.avoidance_cut(d.boresight, d.point_offset, d.site,
				args.planet, R)
	# Subtract atmospheric model
	with bench.show("atm model"):
		if args.model == "joneig":
			model = gapfill.gapfill_joneig(tod, planet_cut, inplace=False)
		elif args.model == "nmat":
			model = gapfill_nmat(tod, planet_cut, d.srate, inplace=False, verbose=True)
	# Estimate noise level
	asens = np.sum(ivar)**-0.5 / d.srate**0.5
	with bench.show("smooth"):
		ft   = fft.rfft(model)
		freq = fft.rfftfreq(model.shape[-1])*d.srate
		flt  = 1/(1+(freq/model_fknee)**model_alpha)
		ft  *= flt
		fft.ifft(ft, model, normalize=True)
		del ft, flt, freq
	with bench.show("atm subtract"):
		tod -= model
		del model
		tod  = tod.astype(dtype, copy=False)
	# Should now be reasonably clean of correlated noise.
	# Proceed to make simple binned map
	with bench.show("actscan"):
		scan = actscan.ACTScan(entry, d=d)
	with bench.show("pmat"):
		pmap = pmat.PmatMap(scan, area, sys=sys)
		pcut = pmat.PmatCut(scan)
		rhs  = enmap.zeros((ncomp,)+shape, area.wcs, dtype)
		div  = enmap.zeros((ncomp,ncomp)+shape, area.wcs, dtype)
		junk = np.zeros(pcut.njunk, dtype)
	with bench.show("rhs"):
		tod *= ivar[:,None]
		pcut.backward(tod, junk)
		pmap.backward(tod, rhs)
	with bench.show("hits"):
		for i in range(ncomp):
			div[i,i] = 1
			pmap.forward(tod, div[i])
			tod *= ivar[:,None]
			pcut.backward(tod, junk)
			div[i] = 0
			pmap.backward(tod, div[i])
	with bench.show("map"):
		idiv = array_ops.eigpow(div, -1, axes=[0,1], lim=1e-5)
		map  = enmap.map_mul(idiv, rhs)
	# Estimate central amplitude
	c = np.array(map.shape[-2:])/2
	crad  = 50
	mcent = map[:,c[0]-crad:c[0]+crad,c[1]-crad:c[1]+crad]
	mcent = enmap.downgrade(mcent, 4)
	amp   = np.max(mcent)
	print amp, asens
	print "%s amp %7.3f asens %7.3f" % (id, amp/1e6, asens)
	with bench.show("write"):
		enmap.write_map("%s%s_map.fits" % (prefix, bid), map)
		enmap.write_map("%s%s_rhs.fits" % (prefix, bid), rhs)
		enmap.write_map("%s%s_div.fits" % (prefix, bid), div)
	del d, scan, pmap, pcut, tod, map, rhs, div, idiv, junk
