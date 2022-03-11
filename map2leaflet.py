import numpy as np, argparse, os, glob
from enlib import retile, mpi
parser = argparse.ArgumentParser()
parser.add_argument("imaps", nargs="+")
parser.add_argument("odir")
parser.add_argument(      "--slice", type=str, default=None)
parser.add_argument("-z", "--nzoom", type=int, default=7)
parser.add_argument(      "--z1",    type=int, default=0)
parser.add_argument("-v", "--verbose", type=int, default=0)
args = parser.parse_args()

imaps = sum([sorted(glob.glob(ifile)) for ifile in args.imaps],[])

comm = mpi.COMM_WORLD
for ind in range(comm.rank, len(imaps), comm.size):
	imap = imaps[ind]
	print("%4d/%d %s" % (ind+1, len(imaps), imap))
	retile.leaftile(imap, "%s/%s" % (args.odir, os.path.basename(imap)), monolithic=True, verbose=args.verbose, comm=mpi.COMM_SELF, slice=args.slice, lrange=[args.z1,args.z1-args.nzoom+1])
