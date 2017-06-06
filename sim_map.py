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
parser.add_argument("-v", "--verbosity", action="count", default=0)
parser.add_argument("--method",         type=str,  default="auto")
parser.add_argument("-e", "--extent",   type=str,  default="subgrid")
args = parser.parse_args()

comm = mpi.COMM_WORLD
enmap.extent_model.append(args.extent)
imap = enmap.read_map(args.template)
lmax = args.lmax or None
maplmax = args.maplmax or None
shape, wcs = (args.ncomp,)+imap.shape[-2:], imap.wcs
seed = args.seed if args.seed is not None else np.random.randint(0,100000000)
def make_beam(nl, bsize):
	l = np.arange(nl)
	return np.exp(-l*(l+1)*bsize**2)

for i in range(comm.rank, args.nsim, comm.size):
	if args.lensed:
		ps = powspec.read_camb_full_lens(args.powspec)
		if args.beam:
			raise NotImplementedError("Beam not supported for lensed sims yet")
		if args.geometry == "curved":
			m, = lensing.rand_map(shape, wcs, ps, lmax=lmax, maplmax=maplmax, seed=(seed,i), verbose=args.verbosity)
		else:
			maps = enmap.rand_map((shape[0]+1,)+shape[1:], wcs, ps)
			phi, unlensed = maps[0], maps[1:]
			m = lensing.lens_map_flat(unlensed, phi)
	else:
		ps = powspec.read_spectrum(args.powspec)
		beam = make_beam(ps.shape[-1], args.beam*utils.arcmin*utils.fwhm)
		ps *= beam
		if args.geometry == "curved":
			m = curvedsky.rand_map(shape, wcs, ps, lmax=lmax, seed=(seed,i), method=args.method)
		else:
			m = enmap.rand_map(shape, wcs, ps)

	if args.nsim == 1:
		enmap.write_map(args.ofile, m)
	else:
		enmap.write_map(args.ofile % i, m)
