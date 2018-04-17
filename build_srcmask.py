from __future__ import division, print_function
import numpy as np, argparse, os, glob
from scipy import ndimage
from enlib import enmap, utils, pointsrcs, mpi
from enact import files
parser = argparse.ArgumentParser()
parser.add_argument("ifile")
parser.add_argument("srcs")
parser.add_argument("beam")
parser.add_argument("ofile")
parser.add_argument("-m", "--mask", type=float, default=0)
parser.add_argument("-a", "--apod", type=int,   default=16)
args = parser.parse_args()

comm = mpi.COMM_WORLD
srcs = pointsrcs.read(args.srcs)
beam = files.read_beam(args.beam)
beam[0] *= utils.degree

def build_single(ifile, srcs, beam, ofile, mask_level=0, apod_size=16):
	imap   = enmap.read_map(ifile)
	omap, oslice = pointsrcs.sim_srcs(imap.shape[-2:], imap.wcs, srcs, beam, return_padded=True)
	if mask_level:
		mask = omap > mask_level
		omap = 1-np.cos(np.minimum(1,ndimage.distance_transform_edt(1-mask)/16.0)*np.pi)
		omap = enmap.samewcs(omap, imap)
	omap = omap[oslice]
	enmap.write_map(ofile, omap)

if os.path.isdir(args.ifile):
	utils.mkdir(args.ofile)
	ifiles = sorted(glob.glob(args.ifile + "/tile*.fits"))[::-1]
	for ind in range(comm.rank, len(ifiles), comm.size):
		ifile = ifiles[ind]
		print(ifile)
		ofile = args.ofile + "/" + os.path.basename(ifile)
		build_single(ifile, srcs, beam, ofile, args.mask, args.apod)
else:
	print(args.ifile)
	build_single(args.ifile, srcs, beam, args.ofile, args.mask, args.apod)
