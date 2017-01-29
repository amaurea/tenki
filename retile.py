# Combine tiles of a dmap into a set of larger-scale tiles
import numpy as np, argparse, os, re
from enlib import enmap, utils, mpi
parser = argparse.ArgumentParser()
parser.add_argument("idir")
parser.add_argument("odir")
parser.add_argument("-c", "--combine",    type=int, default=2)
parser.add_argument("-d", "--downsample", type=int, default=2)
args = parser.parse_args()

comm = mpi.COMM_WORLD
ozpad= args.zpad % 2 == 1
izpad= 0

utils.mkdir(args.odir)
ifiles = os.listdir(args.idir)
# Find the dimensions of the tiling
indig, nitile = [0,0], [0,0]
for ifile in ifiles:
	m = re.match(r"tile(\d+)_(\d+)\.fits", ifile)
	if m:
		for i in range(2):
			indig[i]   = max(indig[i],   len(m.group(i+1)))
			nitile[i] = max(nitile[i], int(m.group(i+1)))
			if indig[i] > 1 and m.group(i+1)[0] == "0":
				izpad = 1
nitile = np.array(nitile)+1

notile = (nitile+1)/args.combine
oxys = [(oy,ox) for oy in range(notile[0]) for ox in range(notile[1])]
for i in range(comm.rank, len(oxys), comm.size):
	oy, ox = oxys[i]
	# Read the input tiles corresponding to this
	ymaps = []
	for dy in range(args.combine):
		iy = oy*args.combine+dy
		if iy >= nitile[0]: continue
		xmaps = []
		for dx in range(args.combine):
			ix = ox*args.combine+dx
			if ix >= nitile[1]: continue
			itname = args.idir + "/tile%0*d_%0*d.fits" % (ndig[0],iy,ndig[1],ix)
			m = enmap.read_map(itname)
			xmaps.append(m)
		ymaps.append(xmaps)
	omap = enmap.tile_maps(ymaps)
	print omap.shape
	if args.downsample > 1:
		omap = enmap.downgrade(omap, args.downsample)
	otname = args.odir + "/tile%0*d_%0*d.fits" % (ndig[0],oy,ndig[1],ox)
	enmap.write_map(otname, omap)
	print otname
