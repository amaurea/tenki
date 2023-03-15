import numpy as np
from pixell import enmap, utils, mpi, fft
from enlib import pmat, sampcut, config, errors, pulsar, array_ops, log, gapfill
from enact import filedb, actdata, actscan, cuts
config.default("verbosity", 1, "Verbosity for output. Higher means more verbose. 0 outputs only errors etc. 1 outputs INFO-level and 2 outputs DEBUG-level messages.")
config.default("eig_limit", 1e-3, "Smallest relative eigenvalue to invert in eigenvalue inversion. Ones smaller than this are set to zero.")
parser = config.ArgumentParser()
parser.add_argument("name_or_coords")
parser.add_argument("sel")
parser.add_argument("area")
parser.add_argument("odir")
parser.add_argument("tag", nargs="?", default=None)
parser.add_argument("-T", "--timing-file", type=str,   default="crabtime.txt")
parser.add_argument("-E", "--ephemeris",   type=str,   default="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de200.bsp")
parser.add_argument("-n", "--nbin",        type=int,   default=10)
parser.add_argument(      "--fknee",       type=float, default=3)
parser.add_argument(      "--alpha",       type=float, default=-10)
parser.add_argument("-R", "--rad",         type=float, default=0.2)
parser.add_argument("-F", "--filter-type", type=str,   default="planet")
args = parser.parse_args()

def lowpass_tod(tod, srate, fknee=3, alpha=-10):
	ft   = fft.rfft(tod)
	freq = fft.rfftfreq(tod.shape[-1])*srate
	with utils.nowarn():
		flt  = 1/(1+(freq/fknee)**-alpha)
	ft  *= flt
	fft.ifft(ft, tod, normalize=True)
	return tod

def planet_filter(scan, coords, tod, R=0.2*utils.degree, fknee=3, alpha=-10):
	planet_cut = cuts.avoidance_cut(scan.d.boresight, scan.d.point_offset, scan.d.site, coords, R)
	model      = gapfill.gapfill_joneig(tod, planet_cut, inplace=False)
	model      = lowpass_tod(model, srate=scan.srate, fknee=fknee, alpha=alpha)
	tod       -= model

filedb.init()
comm       = mpi.COMM_WORLD
ids        = filedb.scans[args.sel]
dtype      = np.float32
ncomp      = 3
nbin       = args.nbin
shape, wcs = enmap.read_map_geometry(args.area)
# Set up our pulsar timing
pulstime   = pulsar.PulsarTiming(args.timing_file)
pulseph    = args.ephemeris
# Pulsar coordinates. ra,dec
known_pulsars = {
		"crab": [83.63322, 22.01446],
}
try: coords = np.array(known_pulsars[args.name_or_coords])*utils.degree
except KeyError: coords = utils.parse_floats(args.name_or_coords)*utils.degree
# Map coordinate system centered on pulsar
sys = "equ:%.6f_%.6f/0_0" % (coords[0]/utils.degree,coords[1]/utils.degree)

utils.mkdir(args.odir)
prefix = args.odir + "/"
if args.tag: prefix += args.tag + "_"

# Set up logging
log_level = log.verbosity2level(config.get("verbosity"))
L = log.init(level=log_level, rank=comm.rank)

# Set up our output maps. These will be reshaped to something sensible later
rhs = enmap.zeros((nbin*ncomp,)     +shape[-2:], wcs, dtype)
div = enmap.zeros((ncomp,nbin*ncomp)+shape[-2:], wcs, dtype)

L.info("Processing %d scans" % len(ids))

for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	# Setup the scan
	entry = filedb.data[id]
	try:
		scan = actscan.ACTScan(entry)
		if scan.ndet == 0 or scan.nsamp < 2: raise errors.DataMissing("no data in tod")
	except errors.DataMissing as e:
		L.debug("%s skipped (%s)" % (id, str(e)))
		continue
	L.debug("%s read" % id)
	# Determine the phase
	ctime = utils.mjd2ctime(scan.mjd0) + scan.boresight[:,0]
	#from enlib import coordinates
	#jodrell_site = coordinates.default_site.copy()
	#jodrell_site.lon = -2.307139
	#jodrell_site.lat = 53.23625
	#jodrell_site.alt = 0
	#
	#ctime_test  = 603295716.09406399727
	#ctime_test += -0.637457
	#print(pulsar.obstime2phase(ctime_test, coords, pulstime, ephem=pulseph, site=jodrell_site))
	phase = pulsar.obstime2phase(ctime, coords, pulstime, ephem=pulseph, interp=True)
	bini  = utils.nint(phase*nbin) % nbin
	# Set up our pointing matrix
	pmap = pmat.PmatMap(scan, rhs, split=bini, sys=sys)
	L.debug("%s pmat" % id)
	# Get the time-ordered data
	tod  = scan.get_samples(verbose=False)
	tod  = utils.deslope(tod)
	tod  = tod.astype(dtype)
	L.debug("%s tod" % id)
	if args.filter_type == "planet":
		# Filter from planet mapmaker. Gets rid of correlated noise
		# without biasing small central region.
		planet_filter(scan, coords, tod, R=args.rad*utils.degree, fknee=args.fknee, alpha=args.alpha)
	else:
		# Lowpass filter. This won't affect our signal much since we're
		# looking for a 30 Hz signal
		freq  = fft.rfftfreq(scan.nsamp, 1/scan.srate)
		ftod  = fft.rfft(tod)
		ftod /= 1 + (np.maximum(freq,freq[1]/2)/args.fknee)**args.alpha
		fft.irfft(ftod, tod, normalize=True)
	L.debug("%s filtered" % id)
	# Estimate noise per detector. Should be white noise by now. Using median
	# of means to be robust to bright signal
	det_ivar = np.median(utils.block_reduce(tod**2, 100, inclusive=False),-1)**-1
	# Update RHS
	sampcut.gapfill_const(scan.cut, tod, 0, inplace=True)
	tod *= det_ivar[:,None]
	pmap.backward(tod, rhs)
	L.debug("%s rhs" % id)
	# Update div
	for i in range(ncomp):
		one = div[0]*0
		one[i::ncomp] = 1
		pmap.forward(tod, one)
		tod *= det_ivar[:,None]
		sampcut.gapfill_const(scan.cut, tod, 0, inplace=True)
		pmap.backward(tod, div[i])
		del one
	L.debug("%s div" % id)
	del scan, tod, pmap

# Done processing tods. Reduce
L.info("Reducing")
rhs = utils.allreduce(rhs, comm)
div = utils.allreduce(div, comm)
if comm.rank == 0:
	L.info("Solving")
	# Reshape to sensible shape
	rhs = rhs.reshape((nbin,ncomp)+rhs.shape[-2:])
	div = div.reshape((ncomp,nbin,ncomp)+div.shape[-2:])
	div = enmap.samewcs(np.ascontiguousarray(np.moveaxis(div, 0, 1)), div)
	# Solve for the map
	idiv = array_ops.eigpow(div,   -1, axes=[-4,-3], lim=config.get("eig_limit"), fallback="scalar")
	map  = enmap.samewcs(array_ops.matmul(idiv, rhs, axes=[-4,-3]),rhs)
	del idiv
	L.info("Writing")
	# Output results
	enmap.write_map(prefix + "map.fits",  map)
	enmap.write_map(prefix + "ivar.fits", div[:,0,0])
	L.info("Done")
