import numpy as np, argparse, os, imp, time
from scipy import ndimage
from enlib import enmap, retile, utils, jointmap, bunch, cg, fft, mpi
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("sel")
parser.add_argument("template", nargs="?")
parser.add_argument("odir")
# Let's use a 4x4 degree test patch with 1 degree of padding on each side and
# another 1 degree of apodization
parser.add_argument("-b", "--box",        type=str,   default="-5:-1,36:32")
parser.add_argument("-t", "--tilesize",   type=int,   default=480)
parser.add_argument("-p", "--pad",        type=int,   default=240)
parser.add_argument("-a", "--apod-val",   type=float, default=2e-1)
parser.add_argument("-A", "--apod-alpha", type=float, default=5)
parser.add_argument("-E", "--apod-edge",  type=float, default=120)
parser.add_argument(      "--kxrad",      type=float, default=90)
parser.add_argument(      "--kx-ymax-scale", type=float, default=1)
parser.add_argument(      "--highpass",   type=float, default=200)
parser.add_argument(      "--cg-tol",     type=float, default=1e-6)
parser.add_argument(      "--max-ps",     type=float, default=0)
parser.add_argument("-M", "--mode",       type=str,   default="weight")
parser.add_argument("-c", "--cont",       action="store_true")
parser.add_argument("-v", "--verbose",    action="store_true")
parser.add_argument("-i", "--maxiter",    type=int,   default=100)
parser.add_argument(      "--ncomp",      type=int,   default=3)
parser.add_argument("-s", "--slice",      type=str,   default=None)
args = parser.parse_args()

# Why did I use this?
enmap.extent_model.append("intermediate")

dtype = np.float64
comm  = mpi.COMM_WORLD
ncomp = args.ncomp
utils.mkdir(args.odir)

config   = jointmap.read_config(args.config)
datasets = jointmap.get_datasets(config, args.sel)
# Get the reference beam, which is the highest beam at each l
ref_beam = datasets[0].beam
for dataset in datasets[1:]:
	ref_beam = np.maximum(ref_beam, dataset.beam)

if args.template is None:
	# Process a single box
	box  = np.array([[float(w) for w in fromto.split(":")] for fromto in args.box.split(",")]).T*utils.degree
	utils.mkdir(args.odir)
	coadder = jointmap.AutoCoadder.read(datasets, box, pad=args.pad, verbose=True, dtype=dtype,
			cache_dir=args.odir, read_cache=args.cont)
	coadder.analyze(ref_beam, mode=args.mode,
			apod_val=args.apod_val, apod_alpha=args.apod_alpha, apod_edge=args.apod_edge,
			filter_kxrad=args.kxrad, filter_kx_ymax_scale=args.kx_ymax_scale, filter_highpass=args.highpass)
	if args.slice: coadder.set_slice(args.slice)
	coadder.calc_precon()
	coadder.calc_rhs()
	res = coadder.solve(verbose=True, cg_tol=args.cg_tol, dump_dir=args.odir, maxiter=args.maxiter)
	if res is None: print "No data found"
	else:
		enmap.write_map(args.odir + "/map.fits", res.map)
		enmap.write_map(args.odir + "/div.fits", res.div)
else:
	# We will loop over tiles in the area defined by template
	shape, wcs = jointmap.read_geometry(args.template)
	pre   = (ncomp,)
	shape = pre + shape
	tshape = np.array([args.tilesize,args.tilesize])
	ntile  = np.floor((shape[-2:]+tshape-1)/tshape).astype(int)
	tyx = [(y,x) for y in range(ntile[0]-1,-1,-1) for x in range(ntile[1])]
	for i in range(comm.rank, len(tyx), comm.size):
		y, x = tyx[i]
		if args.cont and os.path.isfile(args.odir + "/map_padtile%(y)03d_%(x)03d.fits" % {"y":y,"x":x}):
			print "%3d skipping %3d %3d" % (comm.rank, y, x)
			continue
		print "%3d processing %3d %3d" % (comm.rank, y, x)
		tpos = np.array(tyx[i])
		pbox = np.array([tpos*tshape,np.minimum((tpos+1)*tshape,shape[-2:])])
		box  = enmap.pix2sky(shape, wcs, pbox.T).T
		# Set up the coadder
		coadder = jointmap.AutoCoadder.read(datasets, box, pad=args.pad, verbose=True, dtype=dtype, ncomp=ncomp)
		if coadder is not None:
			coadder.analyze(ref_beam, mode=args.mode,
					apod_val=args.apod_val, apod_alpha=args.apod_alpha, apod_edge=args.apod_edge,
					filter_kxrad=args.kxrad, filter_kx_ymax_scale=args.kx_ymax_scale, filter_highpass=args.highpass)
			if args.slice: coadder.set_slice(args.slice)
			coadder.calc_precon()
			coadder.calc_rhs()
			res = coadder.solve(verbose=True, cg_tol=args.cg_tol, maxiter=args.maxiter,
					dump_dir = args.odir if comm.rank==0 else None)
		if coadder is None or res is None:
			res = make_dummy_tile(shape, wcs, box, pad=args.pad)
		enmap.write_map(args.odir + "/map_padtile%(y)03d_%(x)03d.fits" % {"y":y,"x":x}, res.map)
		enmap.write_map(args.odir + "/div_padtile%(y)03d_%(x)03d.fits" % {"y":y,"x":x}, res.div)
