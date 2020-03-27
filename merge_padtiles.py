from __future__ import division, print_function
import numpy as np, argparse, os
from enlib import enmap, retile, utils, mpi
parser = argparse.ArgumentParser()
parser.add_argument("idir")
parser.add_argument("odir")
parser.add_argument("-p", "--pad",  type=int, default=240, help="Number of pixels of padding surrounding the main part of each tile.")
parser.add_argument("-E", "--edge", type=int, default=120, help="Specifies the number of pixels at the edge of the padding to ignore. The rest of the padding will be used to avoid tile discontinuities.")
parser.add_argument("-N", "--ncomp", type=int, default=None, help="Force the map to have this number of components by inserting blank ones as required")
parser.add_argument("-c", "--cont", action="store_true")
args = parser.parse_args()

utils.mkdir(args.odir)
comm = mpi.COMM_WORLD

class MapReader:
	def __init__(self, pathfmt, ncache=9, crop=0, nphi=0, ncomp=None):
		self.pathfmt = pathfmt
		self.cache   = []
		self.crop    = crop
		self.nphi    = nphi
		self.ncache  = ncache
		self.ncomp   = ncomp
	def read(self,y,x):
		if self.nphi: x = x % self.nphi
		for c in self.cache:
			if c[0] == (y,x): return c[1]
		else:
			while len(self.cache) >= self.ncache:
				del self.cache[0]
			fname = self.pathfmt % {"y":y,"x":x}
			if os.path.isfile(fname):
				m = enmap.read_map(fname)
				if self.crop:
					m = m[...,self.crop:-self.crop,self.crop:-self.crop]
				if self.ncomp:
					m = m.preflat
					extra = np.tile(m[:1]*0, (self.ncomp-len(m),1,1))
					m = enmap.samewcs(np.concatenate([m,extra],0),m)
			else:
				m = None
			self.cache.append([(y,x),m])
			return m

def combine_tiles(tiles, weight, dims=(-2,-1)):
	ncont = len(weight)
	if len(dims) > 1:
		tiles = [combine_tiles(row, weight, dims[1:]) for row in tiles]
	# At this point we only have to deal with a 1d combine
	if tiles[1] is None: return None
	map  = tiles[1].copy()
	ndim = map.ndim
	dim  = dims[0] % ndim
	sw   = (slice(None),)+(None,)*(ndim-dim-1)
	s1   = (slice(None),)*dim + (slice(0,ncont),)
	s2   = (slice(None),)*dim + (slice(-ncont,None),)
	if tiles[0] is not None:
		map[s1] = tiles[1][s1]*weight[::-1][sw] + tiles[0][s2]*weight[sw]
	if tiles[2] is not None:
		map[s2] = tiles[1][s2]*weight[sw] + tiles[2][s1]*weight[::-1][sw]
	map = map[(slice(None),)*dim + (slice(ncont//2,-ncont//2),)]
	return map

# The neighboring tiles overlap by ncontext = pad - edge.
# 1111111333333222222222. Will do linear interpolation in
# overlapping region.

# Find our input tiles
ipathfmt = args.idir + "/tile%(y)03d_%(x)03d.fits"
tile1, tile2 = retile.find_tile_range(ipathfmt)
reader = MapReader(ipathfmt, crop=args.edge, nphi=tile2[1], ncomp=args.ncomp)
# Precompute edge weights:
ncontext = args.pad - args.edge
weight   = 1-np.arange(2*ncontext)*1.0/(2*ncontext)

# Loop through tiles
utils.mkdir(args.odir)
for y in range(tile1[0], tile2[0])[comm.rank::comm.size]:
	for x in range(tile1[1], tile2[1]):
		ofile = args.odir + "/tile%(y)03d_%(x)03d.fits" % {"y":y,"x":x}
		if args.cont and os.path.isfile(ofile): continue
		print(ofile)
		tiles = [[reader.read(y+dy,x+dx) for dx in range(-1,2)] for dy in range(-1,2)]
		map = combine_tiles(tiles, weight)
		enmap.write_map(ofile, map)
