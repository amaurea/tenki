# This program takes in a full-resolution CMB map and splits it into
# a hierarchy of tiles suitable for e.g. leaflet maps.
import numpy as np, argparse
from enlib import enmap, utils
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+", help="Input maps to use. Must have compatible pixelization")
parser.add_argument("odir")
parser.add_argument("-s", "--tile-size", type=int, default=256)
args = parser.parse_args()
utils.mkdir(args.odir)

# Read the first map and use it to determine the coordinate system.
# I assume a cylindrical projection.
ts  = args.tile_size
wcs = enmap.read_map(args.ifiles[0]).wcs
pix = wcs.wcs_world2pix([[-180,-90],[180,90]],0)
nx, ny = np.round(np.abs((pix[1]-pix[0]))).astype(int)
nz = int(np.ceil(np.log2(max(ny,nx))-np.log2(ts)))

# Expand reference wcs to fit whole sky. We assume that pixels
# are compatible with [0,0] for now.
wcs.wcs.crval = [0,0]
wcs.wcs.crpix = [nx/2+1,ny/2+1]

maps = [enmap.read_map(ifile) for ifile in args.ifiles]
# Find the displacement from global to local pixels for each map
# pix_loc = pix_glob + off
offs = [m.wcs.wcs_world2pix(wcs.wcs.crval[0],wcs.wcs.crval[1],0) for m in maps]
offs = (np.array(offs)-wcs.wcs.crpix+1).astype(int)

# Ok, loop through each zoom level
for iz in range(nz):
	# And each tile at this zoom
	tpix  = np.mgrid[:ts,:ts]
	tile  = enmap.zeros(maps[0].shape[:-2]+(ts,ts),wcs)
	nyz, nxz = ny/2**iz, nx/2**iz
	zoffs = offs/2**iz
	for y in range(0,nyz,ts):
		yinfos = []
		for map, zoff in zip(maps,zoffs):
			ypix  = np.arange(ts)+y+zoff[1]
			ymask = (ypix < map.shape[-2]) & (ypix >= 0)
			yany  = np.any(ymask)
			yinfos.append((ypix,ymask,yany))
		if not np.any([yinfo[2] for yinfo in yinfos]): continue
		for x in range(0,nxz,ts):
			xinfos = []
			for map, zoff in zip(maps, zoffs):
				xpix  = np.arange(ts)+x+zoff[0]
				xmask = (xpix < map.shape[-1]) & (xpix >= 0)
				xany  = np.any(xmask)
				xinfos.append((xpix,xmask,xany))
			if not np.any([xinfo[2] for xinfo in xinfos]): continue
			print "%5d %5d %5d" % (iz, y, x)
			tile[:] = 0
			for map, yinfo, xinfo in zip(maps, yinfos, xinfos):
				ypix, ymask, yany = yinfo
				xpix, xmask, xany = xinfo
				if not yany or not xany: continue
				tile[...,ymask[:,None]&xmask[None,:]] += map[...,ypix[ymask][:,None],xpix[xmask][None,:]].reshape(map.shape[:-2]+(-1,))
			tile.wcs.wcs.crpix = wcs.wcs.crpix - [x,y]
			utils.mkdir("%s/%d/%d" % (args.odir, nz-iz-1,(nyz-y-1)/ts))
			enmap.write_map("%s/%d/%d/%d.fits" % (args.odir, nz-iz-1, (nyz-y-1)/ts, x/ts), tile)

	# Downgrade all maps
	for i, map in enumerate(maps):
		maps[i] = enmap.downgrade(map,2)
