import numpy as np, argparse
from enlib import retile
parser = argparse.ArgumentParser()
parser.add_argument("idir")
parser.add_argument("ofile")
parser.add_argument("--slice", type=str, default=None)
parser.add_argument("--dtype", type=str, default=None)
args = parser.parse_args()
dtype = np.dtype(args.dtype) if args.dtype else None
retile.monolithic(args.idir, args.ofile, verbose=True, slice=args.slice, dtype=dtype)
