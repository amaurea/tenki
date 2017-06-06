import numpy as np, argparse
from enlib import retile
parser = argparse.ArgumentParser()
parser.add_argument("idir")
parser.add_argument("ofile")
parser.add_argument("--slice", type=str, default=None)
args = parser.parse_args()
retile.monolithic(args.idir, args.ofile, verbose=True, slice=args.slice)
