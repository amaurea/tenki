import numpy as np, argparse, os
from enlib import retile, mpi
parser = argparse.ArgumentParser()
parser.add_argument("idir")
parser.add_argument("odir")
parser.add_argument("--slice", type=str, default=None)
parser.add_argument("--nompi", action="store_true")
args = parser.parse_args()
comm = mpi.COMM_WORLD
if args.nompi: comm = mpi.COMM_SELF
monolithic = os.path.isfile(args.idir)
retile.leaftile(args.idir, args.odir, verbose=True, comm=comm, monolithic=monolithic, slice=args.slice)
