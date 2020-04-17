# Download a section of a webplotted multitile map to a fits file
from __future__ import division, print_function
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("tilepath")
parser.add_argument("ofile")
parser.add_argument("-b", "--box",   type=str,   default="-4:4,4:-4")
parser.add_argument("-T", "--tsize", type=int,   default=675)
parser.add_argument("-r", "--res",   type=float, default=0.5)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()
import numpy as np, requests, io, asyncio, concurrent.futures
from pixell import enmap, utils
from PIL import Image

def download(url, verbose=False):
	if verbose: print(url)
	return requests.get(url).content

def download_all(urls, verbose=False):
	loop = asyncio.get_event_loop()
	async def helper():
		with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
			futures = [ loop.run_in_executor(executor, download, url, verbose) for url in urls ]
		return list(await asyncio.gather(*futures))
	return loop.run_until_complete(helper())

def unpack(imap):
	# Read the metadata row
	meta, qmap = imap[0], imap[1:]
	nbyte   = meta[0]
	quantum = meta[1:9].view(np.float64)[0]
	# Undo plane stacking
	qmap = qmap.reshape(nbyte,-1,qmap.shape[1])
	qmap = np.moveaxis(qmap, 0,-1)
	mask = np.all(qmap==0xff,-1)
	wmap = np.zeros(qmap.shape[:2]+(8,), np.uint8)
	wmap[:,:,:nbyte] = qmap
	wmap = wmap.view(np.uint64)
	# Back to twos complement
	neg  = (wmap & 1) == 1
	wmap >>= 1
	wmap = wmap.view(np.int64)
	wmap[neg] = -wmap[neg]
	# And to real units
	omap = wmap*quantum
	omap[mask] = 0
	omap, mask = omap[...,0], mask[...,0]
	return omap, mask

# Hardcoded geometry and tiling
def webget(tilepath, box, tsize=675, res=0.5*utils.arcmin, dtype=np.float64, verbose=False):
	# Build the geometry representing the tiles web map. This is a normal
	# fullsky geometry, except that the origin is in the top-left corner
	# instead of the bottom-left, so we flip the y axis.
	geo        = enmap.Geometry(*enmap.fullsky_geometry(res=res))[::-1]
	ntile      = np.array(geo.shape)//tsize
	pbox       = enmap.subinds(*geo, box, cap=False, noflip=True)
	# Set up output geometry with the same ordering as the input one. We will
	# flip it to the final ordering at the end
	ogeo       = geo[pbox[0,0]:pbox[1,0],pbox[0,1]:pbox[1,1]]
	omap       = enmap.zeros(*ogeo, dtype=dtype)
	# Loop through tiles
	t1         = pbox[0]//tsize
	t2         = (pbox[1]+tsize-1)//tsize
	urls       = []
	for ty in range(t1[0], t2[0]):
		ty = ty % ntile[0]
		for tx in range(t1[1], t2[1]):
			tx = tx % ntile[1]
			url    = tilepath.format(y=ty, x=tx)
			urls.append(url)
	datas = download_all(urls, verbose=verbose)
	di    = 0
	for ty in range(t1[0], t2[0]):
		ty = ty % ntile[0]
		y1, y2 = ty*tsize, (ty+1)*tsize
		for tx in range(t1[1], t2[1]):
			tx = tx % ntile[1]
			x1, x2 = tx*tsize, (tx+1)*tsize
			tgeo   = geo[y1:y2,x1:x2]
			data   = datas[di]; di += 1
			imgdata= np.array(Image.open(io.BytesIO(data)))
			mapdata, mask = unpack(imgdata)
			map    = enmap.enmap(mapdata, tgeo.wcs)
			omap.insert(map)
	# Flip omap to normal ordering
	omap = omap[::-1]
	return omap

dtype= np.float32
box  = np.array([[float(w) for w in word.split(":")] for word in args.box.split(",")]).T*utils.degree
omap = webget(args.tilepath, box, tsize=args.tsize, res=args.res*utils.arcmin, dtype=dtype, verbose=args.verbose)
enmap.write_map(args.ofile, omap)
