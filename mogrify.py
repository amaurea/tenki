import argparse
parser = argparse.ArgumentParser()
parser.add_argument("code")
parser.add_argument("ifiles", nargs="+")
parser.add_argument("--suffix", type=str, default="_trf")
parser.add_argument("-v", "--verbose",   default=1, action="count")
parser.add_argument("-q", "--quiet",     default=0, action="count")
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils

verbose = args.verbose - args.quiet

for ifile in args.ifiles:
	if verbose > 0: print(ifile)
	toks  = ifile.split(".")
	ofile = ".".join(toks[:-1]) + args.suffix + "." + toks[-1]
	map   = enmap.read_map(ifile)
	env   = {}
	env.update(np.__dict__)
	env.update(utils.__dict__)
	env.update(enmap.__dict__)
	env.update({"np":np, "utils":utils, "enmap":enmap, "m": map})
	try: map = eval(args.code, env)
	except SyntaxError:
		exec(code, env)
		map = env["m"]
	enmap.write_map(ofile, map)
	del map
