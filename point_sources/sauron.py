import argparse, os, sys
help_general = """Usage:
  sauron find  ifile[s] odir
  sauron fit   icat ifile[s] odir

Sauron is a multifrequency, multi-template matched filter based
point source and cluster finder."""
if len(sys.argv) < 2:
	sys.stderr.write(help_general + "\n")
	sys.exit(1)
mode = sys.argv[1]

parser = argparse.ArgumentParser(description=help_general)
parser.add_argument("mode", choices=["find","fit","recat"])
if mode == "find":
	parser.add_argument("ifiles", nargs="+", help="The mapdata files to analyse, one for each frequency")
	parser.add_argument("odir", help="The directory to write the output to. Will be created if necessary")
elif mode == "fit":
	parser.add_argument("icat", help="The input point catalog for flux fitting. Must have the same format as what 'find' would produce.")
	parser.add_argument("ifiles", nargs="+", help="The mapdata files to analyse, one for each frequency")
	parser.add_argument("odir", help="The directory to write the output to. Will be created if necessary")
elif mode == "recat":
	parser.add_argument("icat")
	parser.add_argument("ocat")
else: pass
parser.add_argument("--snr1",  type=float, default=8)
parser.add_argument("--snr2",  type=float, default=4)
parser.add_argument("--nmat1", type=str, default="constcorr")
parser.add_argument("--nmat2", type=str, default="constcorr")
parser.add_argument("--nmat-smooth", type=str, default="angular")
parser.add_argument("--cmb",   type=str, default=None)
parser.add_argument("--slice", type=str, default=None)
parser.add_argument("--box",   type=str, default=None)
parser.add_argument("-c", "--cont",   action="store_true")
parser.add_argument("-t", "--tshape",  type=str, default="500,1000")
parser.add_argument("-C", "--comps",   type=str, default="auto")
parser.add_argument(      "--sim-cat", type=str, default=None)
parser.add_argument(      "--sim-noise", action="store_true")
parser.add_argument("-m", "--mask",    type=str, default=None)
parser.add_argument("-T", "--tiling",  type=str, default="tiled")
args = parser.parse_args()
import numpy as np, time
from pixell import enmap, utils, bunch, analysis, uharm, powspec, pointsrcs, curvedsky, mpi
from enlib import array_ops, sauron
from enlib import mapdata_simple as mapdata
from scipy import ndimage

if args.mode == "find" or args.mode == "fit":
	utils.mkdir(args.odir)
	comm = mpi.COMM_WORLD
	tshape= np.zeros(2,int)+utils.parse_ints(args.tshape)

	dtype  = np.float32
	sel    = utils.parse_slice(args.slice)
	box    = utils.parse_box(args.box)*utils.degree if args.box  else None
	cl_cmb = powspec.read_spectrum(args.cmb)     if args.cmb     else None
	sim_cat= pointsrcs.read_sauron(args.sim_cat) if args.sim_cat else None
	icat   = pointsrcs.read_sauron(args.icat)    if args.mode == "fit" else None

	# Get map dimensions
	shape, wcs = enmap.read_map_geometry(mapdata.read_info(args.ifiles[0]).map)
	if args.comps == "auto":
		if len(shape) == 2: comps = "T"
		else: comps = "TQU"[:shape[-3]]
	else: comps = args.comps

	if args.tiling == "tiled":
		sauron.search_maps_tiled(args.ifiles, args.odir, mode=args.mode, icat=icat, tshape=tshape, sel=sel, box=box, cl_cmb=cl_cmb, nmat1=args.nmat1, nmat2=args.nmat2, snr1=args.snr1, snr2=args.snr2, dtype=dtype, verbose=True, cont=args.cont, comm=comm, comps=comps, sim_cat=sim_cat, sim_noise=args.sim_noise, mask=args.mask)
	elif args.tiling == "single":
		res = sauron.search_maps(args.ifiles, mode=args.mode, icat=icat, sel=sel, box=box, cl_cmb=cl_cmb, nmat1=args.nmat1, nmat2=args.nmat2, snr1=args.snr1, snr2=args.snr2, dtype=dtype, verbose=True, comps=comps, sim_cat=sim_cat, sim_noise=args.sim_noise, mask=args.mask)
		sauron.write_results(args.odir, res)
	else:
		print("Unrecognized tiling '%s'" % args.tiling)
		sys.exit(1)

elif args.mode == "recat":
	icat   = pointsrcs.read_sauron(args.icat)
	pointsrcs.write_sauron(args.ocat, icat)
