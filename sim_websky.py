# The goal of this program is to read in the WebSky cluster catalog
# and produce an enmap for the given geometry, frequency and beam.
# We want to reproduce what Nemo does, but with my cluster profile
# evaluation code
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("halos",   help="Raw WebSky catalog pkcs file")
parser.add_argument("geometry",help="Geometry template. A file containing (at least) a fits header specifying the map shape and projection to use")
parser.add_argument("ofile",   help="Output map")
parser.add_argument("-f", "--freq",  type=float, default=98, help="Frequency to simulate at")
parser.add_argument("-b", "--beam",  type=str,   default="2.1", help="The beam to use. Either a number, which is interpreted as a FWHM in arcmin, or a file name, which should be to a beam transform file with the format [l,b(l)]")
parser.add_argument("-n", "--nhalo", type=int,   default=0, help="Number of halos to use. 0 for unlimited")
parser.add_argument("-B", "--bsize", type=int,   default=100000)
parser.add_argument("-V", "--vmin",  type=float, default=0.001)
args = parser.parse_args()
import numpy as np, pyccl, time
from astropy.io import fits
from pixell import enmap, utils, bunch, pointsrcs
from enlib import mpi, clusters

dtype   = np.float32 # only float32 supported by fast srcsim
comm    = mpi.COMM_WORLD
bsize   = args.bsize
margin  = 100
beta_range = [-14,-3]
# This is the profile cutoff in µK. It defaults to 1e-3, i.e. 1 nK. This
# should be much lower than necessary, but I set it this low because the
# profile building step is the bottleneck for most objects anyway. If that
# step could be sped up, then this program could be made much faster by increasing
# this number. But for now it's practically free to keep it this low.
vmin    = args.vmin

nhalo   = clusters.websky_pkcs_nhalo(args.halos)
if args.nhalo:
	nhalo = min(nhalo, args.nhalo)

cosmology   = pyccl.Cosmology(Omega_c=0.2589, Omega_b=0.0486, h=0.6774, sigma8=0.8159, n_s=0.9667, transfer_function="boltzmann_camb")
shape, wcs  = enmap.read_map_geometry(args.geometry)
freq        = args.freq*1e9
omap        = enmap.zeros(shape[-2:], wcs, dtype)
rht         = utils.RadialFourierTransform()
# Read the beam from one of the two formats
try:
	sigma = float(args.beam)*utils.fwhm*utils.arcmin
	lbeam = np.exp(-0.5*rht.l**2*sigma**2)
except ValueError:
	l, bl = np.loadtxt(args.beam, usecols=(0,1), ndmin=2).T
	lbeam = np.interp(rht.l, l, bl)
prof_builder= clusters.ProfileBattagliaFast(cosmology=cosmology, beta_range=beta_range)
mass_interp = clusters.MdeltaTranslator(cosmology)

# We use this to decide if it's worth it to prune
# objects outide our patch or now. This pruning takes
# some extra calculations which aren't necessary if we're
# fullsky or close to it
fullsky = enmap.area(shape, wcs)/(4*np.pi) > 0.8

# Loop over halos
nblock  = (nhalo+bsize-1)//bsize
tget    = 0
tprof   = 0
tpaint  = 0
ntot    = 0
for bi in range(comm.rank, nblock, comm.size):
	i1    = bi*bsize
	i2    = min((bi+1)*bsize, nhalo)
	t1    = time.time()
	data  = clusters.websky_pkcs_read(args.halos, num=i2-i1, offset=i1)
	# Prune the ones outside our area
	if not fullsky:
		pixs  = enmap.sky2pix(shape, wcs, utils.rect2ang(data.T[:3])[::-1])
		good  = np.all((pixs >= -margin) & (pixs < np.array(shape[-2:])[:,None]+margin), 0)
		data  = data[good]
	ngood = len(data)
	if ngood == 0: continue
	# Compute physical quantities
	cat    = clusters.websky_decode(data, cosmology, mass_interp); del data
	t2     = time.time(); tget = t2-t1
	# Evaluate the y profile
	rprofs  = prof_builder.y(cat.m200[:,None], cat.z[:,None], rht.r)
	# convolve with beam
	lprofs  = rht.real2harm(rprofs)
	lprofs  *= lbeam
	rprofs  = rht.harm2real(lprofs)
	r, rprofs = rht.unpad(rht.r, rprofs)
	# and factor out peak value
	yamps   = rprofs[:,0].copy()
	rprofs /= yamps[:,None]
	# Prepare for painting
	amps   = (yamps * utils.tsz_spectrum(freq) / utils.dplanck(freq) * 1e6).astype(dtype)
	poss   = np.array([cat.dec,cat.ra]).astype(dtype)
	profiles = [np.array([r,prof]).astype(dtype) for prof in rprofs]; del rprofs
	prof_ids = np.arange(len(profiles)).astype(np.int32)
	# And paint
	ntot += ngood
	t3 = time.time(); tprof  = t3-t2
	pointsrcs.sim_objects(shape, wcs, poss, amps, profiles, prof_ids=prof_ids, omap=omap, vmin=vmin)
	t4 = time.time(); tpaint = t4-t3
	# Print our status
	print("%3d %4d/%d ndone %6.1fk get %6.3f ms prof %6.3f ms draw %6.3f tot %6.3f each maxamp %7.2f" % (
		comm.rank, bi+1, nblock, ntot/1e3, tget/ngood*1e3, tprof/ngood*1e3, tpaint/ngood*1e3,
		(tget+tprof+tpaint)/ngood*1e3, np.max(np.abs(amps))))

print("%4d Reducing" % comm.rank)

if comm.size > 1:
	comm.Barrier()
	if comm.rank == 0:
		omap_full = np.zeros_like(omap)
		comm.Reduce(omap, omap_full)
		omap = omap_full
	else:
		comm.Reduce(omap, None)
		del omap
	comm.Barrier()
if comm.rank == 0:
	enmap.write_map(args.ofile, omap)
	print("Done")
