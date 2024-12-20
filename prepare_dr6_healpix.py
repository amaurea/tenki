import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("-N", "--nside", type=int, default=8192)
parser.add_argument("-l", "--lmax",  type=int, default=21000)
parser.add_argument("-c", "--cont",  action="store_true")
args = parser.parse_args()
import numpy as np, os, healpy
from pixell import enmap, utils, curvedsky, reproject, mpi, bunch

act_fields = set("TELESCOP", "INSTRUME", "RELEASE", "SEASON", "PATCH", "ARRAY", "FREQ", "ACTTAGS", "BUNIT", "EXTNAME")

def extract_act_fields(header):
	fields = []
	for key, val in header.items():
		print(type(val), val)
		if key in act_fields:
			fields.append([key]+val)
	return fields

comm    = mpi.COMM_WORLD
ifnames = sum([sorted(utils.glob(ifname)) for ifname in args.ifiles],[])

tasks = []
for fi, ifname in enumerate(ifnames):
	ofname= utils.replace(ifname, ".fits", "_healpix.fits")
	if args.cont and os.path.isfile(ofname): continue
	isvar = ifname.endswith("_ivar.fits") or ifname.endswith("_xlink.fits")
	tasks.append(bunch.Bunch(ifname=ifname, ofname=ofname, isvar=isvar))

for ind in range(comm.rank, len(tasks), comm.size):
	task    = tasks[ind]
	map     = enmap.read_map(task.ifname)
	headers = enmap.read_map_headers(task.ifname)
	fields  = extract_act_fields(headers)
	dtype   = map.dtype
	if task.isvar:
		omap = reproject.map2healpix(map, nside=args.nside, rot="equ,gal", method="spline", spin=[0], order=0, extensive=True)
	else:
		mask  = map.preflat[0]!=0
		omask = reproject.map2healpix(omask, nside=args.nside, rot="equ,gal", method="spline", order=0)
		del mask
		omap  = reproject.map2healpix(map, nside=args.nside, rot="equ,gal", method="harm", lmax=args.lmax)
		omap *= omask
		del omask
	del map
	healpy.write_map(task.ofname, omap, dtype=dtype, coord="G", extra_header=fields, overwrite=True)
	del omap
	print(task.ofname)