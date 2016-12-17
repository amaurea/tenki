import numpy as np, argparse, os
from enlib import enmap, utils
from enact import files
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("ofile")
args = parser.parse_args()

maps = []
for ifile in args.ifiles:
	ibase = os.path.basename(ifile)
	print ibase
	data  = enmap.read_map(ifile)
	maps.append(data)

maps = enmap.samewcs(np.array(maps),maps[0])

print np.sum(maps**2)
print maps.shape

masked = np.ma.MaskedArray(maps, mask=maps==0)
print np.sum(masked**2)

medmap = np.ma.median(masked,0)
medmap = enmap.samewcs(np.asarray(medmap),maps)

# And run through again, subtracting it
enmap.write_map(args.ofile, medmap)
