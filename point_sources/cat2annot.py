import argparse
parser = argparse.ArgumentParser()
parser.add_argument("icat")
parser.add_argument("annot", nargs="?", default="/dev/stdout")
parser.add_argument("-s", "--snlim", type=float, default=0)
parser.add_argument("-a", "--alim",  type=float, default=0)
parser.add_argument("-p", "--positive", type=str, default="#f00")
parser.add_argument("-n", "--negative", type=str, default="#00f")
parser.add_argument("-r", "--radius",   type=str, default="s")
parser.add_argument("-w", "--width",    type=int, default=2)
parser.add_argument("-N", "--number", action="store_true")
parser.add_argument("-f", "--fontsize", type=int, default=16)
args = parser.parse_args()

def read_cat(ifile):
	if ifile.endswith(".fits"):
		import numpy as np
		from enlib import dory, utils
		cat = dory.read_catalog(ifile)
		return np.array([cat.ra/utils.degree, cat.dec/utils.degree, cat.amp[:,0]/cat.damp[:,0], cat.amp[:,0]]).T
	else:
		res = []
		with open(ifile, "r") as ifile:
			for line in ifile:
				if line.startswith("#") or len(line) == 0: continue
				toks = line.split()
				res.append(map(float, toks[:4]))
		return res

def get_radius(expr, sn, amp):
	return int(abs(0.5+eval(expr, {"s":sn, "a":amp})))

cat = read_cat(args.icat)

with open(args.annot, "w") as ofile:
	for i, (ra,dec,sn,amp) in enumerate(cat):
		if abs(sn) < args.snlim: continue
		color = args.positive if sn >= 0 else args.negative
		if not color: continue
		r = get_radius(args.radius, sn, amp)
		ofile.write("circle %12.7f %12.7f %3d %3d %5d %2d %s\n" % (dec, ra, 0, 0, r, args.width, color))
		if args.number:
			ofile.write("text %12.7f %12.7f %3d %3d %s %2d %s\n" % (dec, ra, r, r, i+1, args.fontsize, color))
