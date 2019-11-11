import numpy as np, argparse
from enlib import enmap, curvedsky, lensing, powspec, utils, mpi
parser = argparse.ArgumentParser()
parser.add_argument("template")
parser.add_argument("powspec")
parser.add_argument("ofile")
parser.add_argument("-L", "--lensed", action="store_true")
parser.add_argument("-g", "--geometry", type=str,  default="curved")
parser.add_argument("-l", "--lmax",     type=int,  default=0)
parser.add_argument("-m", "--maplmax",  type=int,  default=0)
parser.add_argument("-s", "--seed",     type=int,  default=None)
parser.add_argument("-b", "--beam",     type=float,default=0)
parser.add_argument("-N", "--nsim",     type=int,  default=1)
parser.add_argument("--ncomp",          type=int,  default=3)
parser.add_argument("-v", "--verbose",  action="count", default=0)
parser.add_argument("-q", "--quiet",    action="count", default=0)
parser.add_argument("--method",         type=str,  default="auto")
parser.add_argument("-D", "--direct",   action="store_true")
parser.add_argument("-B", "--bits",     type=int, default=64)
parser.add_argument("-S", "--sigma",    type=float, default=0)
args = parser.parse_args()

comm = mpi.COMM_WORLD
shape, wcs = enmap.read_map_geometry(args.template)
shape = (args.ncomp,)+shape[-2:]
lmax = args.lmax or None
maplmax = args.maplmax or None
seed = args.seed if args.seed is not None else np.random.randint(0,100000000)
dtype= {32:np.float32, 64:np.float64}[args.bits]
verbose = args.verbose - args.quiet > 0
def make_beam(nl, bsize):
	l = np.arange(nl)
	return np.exp(-l*(l+1)*bsize**2)

for i in range(comm.rank, args.nsim, comm.size):
	if args.lensed:
		ps = powspec.read_camb_full_lens(args.powspec).astype(dtype)
		if args.geometry == "curved":
			if args.beam:
				raise NotImplementedError("Beam not supported for curved lensed sims yet")
			m, = lensing.rand_map(shape, wcs, ps, lmax=lmax, maplmax=maplmax, seed=(seed,i), verbose=verbose, dtype=dtype)
		else:
			maps = enmap.rand_map((shape[0]+1,)+shape[1:], wcs, ps, spin=[0,0,2])
			enmap.write_map("moo.fits", maps)
			phi, unlensed = maps[0], maps[1:]
			m = lensing.lens_map_flat(unlensed, phi)
			if args.beam:
				beam = make_beam(ps.shape[-1], args.beam*utils.arcmin*utils.fwhm)
				l    = m.modlmap()
				beam2d = np.interp(l, np.arange(len(beam)), beam)
				m    = enmap.ifft(enmap.fft(m)*beam2d).real
	else:
		ps = powspec.read_spectrum(args.powspec).astype(type)
		beam = make_beam(ps.shape[-1], args.beam*utils.arcmin*utils.fwhm)
		ps *= beam
		if args.geometry == "curved":
			m = curvedsky.rand_map(shape, wcs, ps, lmax=lmax, seed=(seed,i), method=args.method, direct=args.direct, dtype=dtype, verbose=verbose)
		else:
			m = enmap.rand_map(shape, wcs, ps)
	if args.sigma:
		m += np.random.standard_normal(m.shape)*args.sigma

	if args.nsim == 1:
		if verbose: print "Writing %s" % args.ofile
		enmap.write_map(args.ofile, m)
	else:
		if verbose: print "Writing %s" % (args.ofile % i)
		enmap.write_map(args.ofile % i, m)
