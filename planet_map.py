import numpy as np, sys, os, h5py
from enlib import config, pmat, mpi, errors, gapfill, utils, enmap, bench
from enlib import fft, array_ops
from enact import filedb, actscan, actdata, cuts

parser = config.ArgumentParser(os.environ["HOME"]+"./enkirc")
parser.add_argument("planet")
parser.add_argument("area")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("tag", nargs="?")
parser.add_argument("-R", "--dist", type=float, default=0.2)
parser.add_argument("-e", "--equator", action="store_true")
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
	except errors.DataMissing as e:
		print "Skipping %s (%s)" % (id, e.message)
		continue
	print "Processing %s" % id
	# Very simple white noise model
	with bench.show("ivar"):
		tod  = d.tod
		diff = tod[:,1:]-tod[:,:-1]
		diff = diff[:,:diff.shape[-1]/csize*csize].reshape(d.ndet,-1,csize)
		ivar = 1/(np.median(np.mean(diff**2,-1),-1)/2**0.5)
	# Generate planet cut
	with bench.show("planet cut"):
		planet_cut = cuts.avoidance_cut(d.boresight, d.point_offset, d.site,
				args.planet, R)
	# Subtract atmospheric model
	with bench.show("atm model"):
		model= gapfill.gapfill_joneig(tod, planet_cut, inplace=False)
	with bench.show("smooth"):
		ft   = fft.rfft(model)
		freq = fft.rfftfreq(model.shape[-1])*d.srate
		flt  = 1/(1+(freq/model_fknee)**model_alpha)
		ft  *= flt
		fft.ifft(ft, model, normalize=True)
		del ft, flt, freq
	with bench.show("atm subtract"):
		tod -= model
		tod  = tod.astype(dtype)
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
	with bench.show("write"):
		enmap.write_map("%s%s_map.fits" % (prefix, bid), map)
		enmap.write_map("%s%s_rhs.fits" % (prefix, bid), rhs)
		enmap.write_map("%s%s_div.fits" % (prefix, bid), div)
	del scan, pmap, pcut, tod, map, rhs, div, idiv
