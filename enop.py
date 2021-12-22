import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ifile")
parser.add_argument("ofile")
parser.add_argument("--op", type=str, default=None)
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils

m = enmap.read_map(args.ifile)
if args.op is not None:
	m = eval(args.op, {"m":m,"enmap":enmap,"utils":utils,"np":np},np.__dict__)
enmap.write_map(args.ofile, m)
