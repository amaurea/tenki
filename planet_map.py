import numpy as np, sys, os, h5py
from enlib import config, pmat, mpi, errors, gapfill, utils, enmap, bench
from enlib import fft, array_ops
from enact import filedb, actscan, actdata, cuts, nmat_measure

config.set("pmat_cut_type",  "full")

parser = config.ArgumentParser(os.environ["HOME"]+"./enkirc")
parser.add_argument("planet")
parser.add_argument("area")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("tag", nargs="?")
parser.add_argument("-R", "--dist",    type=float, default=0.2)
parser.add_argument("-e", "--equator", action="store_true")
parser.add_argument("-c", "--cont",    action="store_true")
parser.add_argument("--sim",           type=str,   default=None, help="Passing a sel here sets up simulation mode. The simulations will consist of data from the sim sel TODs with the scanning pattern of the real TODs, and with the signal read off from the area map")
parser.add_argument("--noiseless",      action="store_true", help="Replace signal with simulation instead of adding them. This can be used to get noise free transfer functions")
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

if args.sim:
	sim_ids = filedb.scans[args.sim][:len(ids)]
	if area.ndim == 2:
		tmp = enmap.zeros((ncomp,)+shape, area.wcs, dtype)
		tmp[0] = area
		area = tmp

def smooth(tod, srate):
	ft   = fft.rfft(tod)
	freq = fft.rfftfreq(tod.shape[-1])*srate
	flt  = 1/(1+(freq/model_fknee)**model_alpha)
	ft  *= flt
	fft.ifft(ft, tod, normalize=True)
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
		if not args.sim:
			with bench.show("read"):
				d = actdata.read(entry)
		else:
			sim_id    = sim_ids[ind]
			sim_entry = filedb.data[sim_id]
			with bench.show("read"):
				d  = actdata.read(entry, ["boresight"])
				d += actdata.read(sim_entry, exclude=["boresight"])
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
	# Estimate noise level
	asens = np.sum(ivar)**-0.5 / d.srate**0.5
	with bench.show("actscan"):
		scan = actscan.ACTScan(entry, d=d)
	with bench.show("pmat"):
		pmap = pmat.PmatMap(scan, area, sys=sys)
		pcut = pmat.PmatCut(scan)
		rhs  = enmap.zeros((ncomp,)+shape, area.wcs, dtype)
		div  = enmap.zeros((ncomp,ncomp)+shape, area.wcs, dtype)
		junk = np.zeros(pcut.njunk, dtype)
	# Generate planet cut
	with bench.show("planet cut"):
		planet_cut = cuts.avoidance_cut(d.boresight, d.point_offset, d.site,
				args.planet, R)
	if args.sim:
		if args.noiseless: tod_orig = tod.copy()
		with bench.show("inject"):
			pmap.forward(tod, area)
	# Compute atmospheric model
	with bench.show("atm model"):
		model  = smooth(gapfill.gapfill_joneig(tod,      planet_cut, inplace=False), d.srate)
	if args.sim and args.noiseless:
		model -= smooth(gapfill.gapfill_joneig(tod_orig, planet_cut, inplace=False), d.srate)
		tod   -= tod_orig
		del tod_orig
	with bench.show("atm subtract"):
		tod -= model
		del model
		tod  = tod.astype(dtype, copy=False)
	# Should now be reasonably clean of correlated noise.
	# Proceed to make simple binned map
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
	print "%s amp %7.3f asens %7.3f" % (id, amp/1e6, asens)
	with bench.show("write"):
		enmap.write_map("%s%s_map.fits" % (prefix, bid), map)
		enmap.write_map("%s%s_rhs.fits" % (prefix, bid), rhs)
		enmap.write_map("%s%s_div.fits" % (prefix, bid), div)
	del d, scan, pmap, pcut, tod, map, rhs, div, idiv, junk
