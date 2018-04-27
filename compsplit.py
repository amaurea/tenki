from __future__ import division, print_function
import numpy as np, argparse, os
from enlib import enmap
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("-s", "--slice", type=str, default=None)
parser.add_argument("-S", "--symmetric", action="store_true")
parser.add_argument("-v", "--verbose",   action="store_true")
args = parser.parse_args()

def get_num_digits(n): return int(np.log10(n))+1
def split_file_name(fname):
	"""Split a file name into directory, base name and extension,
	such that fname = dirname + "/" + basename + "." + ext."""
	dirname  = os.path.dirname(fname)
	if len(dirname) == 0: dirname = "."
	base_ext = os.path.basename(fname)
	# Find the extension. Using the last dot does not work for .fits.gz.
	# Using the first dot in basename does not work for foo2.5_bar.fits.
	# Will therefore handle .gz as a special case.
	if base_ext.endswith(".gz"):
		dot = base_ext[:-3].rfind(".")
	else:
		dot  = base_ext.rfind(".")
	if dot < 0: dot = len(base_ext)
	base = base_ext[:dot]
	ext  = base_ext[dot+1:]
	return dirname, base, ext

for ifile in args.ifiles:
	m = enmap.read_map(ifile)
	if args.slice:
		m = eval("m" + args.slice)
	N = m.shape[:-2]
	for i, comp in enumerate(m.preflat):
		I = np.unravel_index(i, N) if len(N) > 0 else []
		if args.symmetric and np.any(np.sort(I) != I):
			continue
		ndigits  = [get_num_digits(n) for n in N]
		compname = "_".join(["%0*d" % (ndig,ind) for ndig,ind in zip(ndigits,I)]) if len(N) > 0 else ""
		dir, base, ext = split_file_name(ifile)
		oname = dir + "/" + base + "_" + compname + "." + ext
		if args.verbose: print(oname)
		enmap.write_map(oname, comp)
