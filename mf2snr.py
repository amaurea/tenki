import argparse
parser = argparse.ArgumentParser()
parser.add_argument("rhos", nargs="+")
parser.add_argument("--tol", type=float, default=0.01)
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils, mpi

comm = mpi.COMM_WORLD
rhofiles = sum([sorted(utils.glob(ifile)) for ifile in args.rhos],[])

for ind in range(comm.rank, len(rhofiles), comm.size):
	rhofile = rhofiles[ind]
	kapfile = utils.replace(rhofile, "_rho", "_kappa")
	ofile   = utils.replace(rhofile, "_rho", "_snr")
	rho     = enmap.read_map(rhofile)
	kappa   = enmap.read_map(kapfile)
	kappa   = np.maximum(kappa, np.max(kappa)*args.tol)
	snr     = rho/kappa**0.5
	enmap.write_map(ofile, snr)
	print(ofile)
