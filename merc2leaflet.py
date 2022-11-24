import argparse, sys
parser = argparse.ArgumentParser()
parser.add_argument("ifile")
parser.add_argument("odir")
parser.add_argument("-T", "--tsize", type=int, default=256)
args = parser.parse_args()
import numpy as np
from pixell import enmap, utils, mpi

comm = mpi.COMM_WORLD

# Check for problems first, and compute number of levels
dtype = np.float32
tsize = args.tsize
shape, wcs = enmap.read_map_geometry(args.ifile)
nlevel = utils.nint(np.log2(shape[-1]/tsize))+1
tmp = tsize * 2**(nlevel-1)
if tmp != shape[-1] or tmp != shape[-2]:
	sys.stderr.write("Map size must be power of two times the tile size, but was %s\n" % str(shape))
	sys.exit(1)

if comm.rank == 0:
	sys.stderr.write("Reading %s\n" % args.ifile)
	map = enmap.read_map(args.ifile).astype(dtype, copy=False)
utils.mkdir(args.odir)

oi = 0
for i in range(nlevel):
	ny = shape[-2]//2**i//tsize
	nx = shape[-1]//2**i//tsize
	tdir = "%s/%d" % (args.odir, nlevel-1-i)
	utils.mkdir(tdir)
	for ty in range(ny):
		for tx in range(nx):
			if comm.size == 1:
				tile = map[...,ty*tsize:(ty+1)*tsize,tx*tsize:(tx+1)*tsize]
				enmap.write_map("%s/tile_%d_%d.fits" % (tdir, ny-1-ty, tx), tile)
			else:
				# We do it this way to avoid having the sender block too much.
				# If the sender spends time writing itself, then the others end
				# up waiting while it blocks.
				write_rank = oi % (comm.size-1) + 1
				if comm.rank == 0:
					tile = map[...,ty*tsize:(ty+1)*tsize,tx*tsize:(tx+1)*tsize]
					tile = np.ascontiguousarray(tile)
					tile.dtype = utils.fix_dtype_mpi4py(tile.dtype)
					comm.Send(tile, dest=write_rank, tag=oi)
				elif comm.rank == write_rank:
					# We should write this tile. Set up the buffer to receive it to
					tgeo = enmap.Geometry(shape, wcs)[ty*tsize:(ty+1)*tsize,tx*tsize:(tx+1)*tsize]
					tile = enmap.zeros(*tgeo, dtype=dtype)
					comm.Recv(tile, source=0, tag=oi)
					enmap.write_map("%s/tile_%d_%d.fits" % (tdir, ny-1-ty, tx), tile)
			if comm.rank == 0:
				sys.stderr.write("\r%2d %3d %3d" % (i, ty, tx))
			oi += 1
	if comm.rank == 0:
		map = enmap.downgrade(map, 2)
comm.Barrier()
if comm.rank == 0:
	sys.stderr.write("\nDone\n")
