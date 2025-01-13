from __future__ import print_function, division
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("idir")
parser.add_argument("odir")
parser.add_argument("-i", "--iteration", type=str, default="individual")
parser.add_argument("-a", "--allow-nonstandard",   action="store_true")
parser.add_argument("-c", "--cont",                action="store_true")
parser.add_argument("-d", "--dry",                 action="store_true")
parser.add_argument("-O", "--output",    type=str, default="map,ivar,sens,xlink,hits,totmap,totsens,totxlink,tothits")
parser.add_argument("--exclude",         type=str, default=None)
parser.add_argument("--only",            type=str, default=None)
parser.add_argument("--repixwin",        type=str, default=None)
parser.add_argument("--suffix",          type=str, default="")
args = parser.parse_args()
import numpy as np, glob, re, os, shutil, sys
from enlib import enmap, utils, retile, bunch, mpi

comm = mpi.COMM_WORLD
outputs = set(args.output.split(","))
verbose = True
dtype   = np.float32

def get_ref(ivar):
	v = ivar[ivar>0]
	if len(v > 1000): return np.median(v[::1+utils.floor(len(v)/1000)])
	elif len(v) > 1: return np.max(v)
	else: return 1

# Optional pixel window correction
def apply_fourier(map, ivar, op, tol=0.01, inplace=True):
	# To make fourier operations safe, we want to whiten areas with
	# very high hitcount variance, which will be areas where the hitcount
	# is low.
	ref = get_ref(ivar)*tol
	low = (ivar>0)&(ivar<ref)
	whiten = (ivar[low]/ref)**0.5
	# Apply edge whitening
	if not inplace:
		map = map.copy()
	map[:,low] *= whiten
	for I in utils.nditer(map.shape[:-2]):
		# Actually apply our fourier operation
		fmap     = enmap.fft(map[I])
		fmap     = op(fmap)
		map[I]   = enmap.ifft(fmap).real
		del fmap
	# Undo edge whitening
	map[:,low] /= whiten
	# Remask
	map[:,ivar<=0] = 0
	return map

if args.repixwin:
	iwin, owin = utils.parse_ints(args.repixwin)
	def mapfix(map, ivar):
		return apply_fourier(map, ivar,
			lambda fmap: enmap.apply_window(enmap.unapply_window(fmap, order=iwin, nofft=True), order=owin, nofft=True)
		)
else:
	def mapfix(map, ivar): return map

# Look for map files in the input directory
datasets = {}
for fname in utils.glob(args.idir + "/*map????.fits"):
	fname = os.path.basename(fname)
	if args.exclude and re.search(args.exclude, fname): continue
	if args.only    and not re.search(args.only, fname): continue
	m = re.search(r"^(.*)_(\d+)way_(\d+)_sky_map(\d\d\d\d).fits", fname)
	if not m:
		if comm.rank == 0: print("Skipping unrecognized map name: " + fname)
		continue
	prefix = m.group(1)
	nway   = int(m.group(2))
	sub    = int(m.group(3))
	it     = int(m.group(4))
	if prefix not in datasets:
		datasets[prefix] = [bunch.Bunch(it=0, maxit=0, its=set(), name="") for i in range(nway)]
	datasets[prefix][sub].maxit = max(datasets[prefix][sub].maxit, it)
	datasets[prefix][sub].its.add(it)
	datasets[prefix][sub].name = "_".join(fname.split("_")[:-2])

nchar = max([len(key) for key in datasets])

# Find the iteration count to use for each dataset
for key in sorted(datasets.keys()):
	d     = datasets[key]
	minit = min([sub.it for sub in d])
	if   args.iteration == "individual":
		for i in range(len(d)): d[i].it = d[i].maxit
	elif args.iteration == "min":
		for i in range(len(d)): d[i].it = minit
	else:
		try:
			for i in range(len(d)):
				it = int(args.iteration)
				if it in d[i].its:
					d[i].it = int(args.iteration)
		except ValueError:
			raise ValueError(args.iteration)
	
	# Do we have any zeros at this point? If so, warn and skip the dataset
	nzero = sum([sub.it == 0 for sub in d])
	if nzero > 0:
		if comm.rank == 0:
			desc = " %4d"*len(d) % tuple([sub.it for sub in d])
			print("%-*s with %s has only %d/%d splits. Skipping" % (nchar, key, desc, len(d)-nzero, len(d)))
		del datasets[key]
	elif comm.rank == 0:
		print("%-*s using" % (nchar, key) + " %4d"*len(d) % tuple([sub.it for sub in d]))

# Check if we follow the standard format
onames = {}
for key in list(datasets.keys()):
	m = re.match(r"s\d\d_\w+_pa\d_f\d\d\d_(no)?hwp_(day|night|daynight)\b.*", key)
	if not m:
		if args.allow_nonstandard: onames[key] = key
		else: del datasets[key]
	else:
		toks = key.split("_")
		if toks[1] == "cmb": toks[1] = "advact"
		onames[key] = "_".join(toks[:6])

def read_map(ifile, slice=None):
	if not os.path.isdir(ifile):
		map = enmap.read_map(ifile)
		if slice: map = eval("map"+slice)
	else:
		map = retile.read_monolithic(ifile, slice=slice, verbose=False)
	map = map.astype(dtype, copy=False)
	return map

def copy_mono(ifile, ofile, slice=None):
	if args.cont and os.path.exists(ofile): return
	if verbose: print("%3d copy_mono %s" % (comm.rank, ofile))
	if args.dry: return
	tfile = ofile + ".tmp"
	map   = read_map(ifile, slice=slice)
	enmap.write_map(tfile, map)
	shutil.move(tfile, ofile)

def add_mono(ifiles, ofile, slice=None, factors=None):
	if args.cont and os.path.exists(ofile): return
	if verbose: print("%3d add_mono %s" % (comm.rank, ofile))
	if args.dry: return
	tfile = ofile + ".tmp"
	if factors is None:
		omap  = read_map(ifiles[0], slice=slice)
		for ifile in ifiles[1:]:
			omap += read_map(ifile, slice=slice)
	else:
		omap  = read_map(ifiles[0], slice=slice)*factors[0]
		for i in range(1, len(ifiles)):
			omap += read_map(ifiles[i], slice=slice)*factors[i]
	enmap.write_map(tfile, omap)
	shutil.move(tfile, ofile)

def coadd_mono(imapfiles, idivfiles, omapfile, odivfile=None, op=lambda map,ivar:map):
	if args.cont and os.path.exists(omapfile) and (odivfile is None or os.path.exists(odivfile)): return
	if verbose: print("%3d coadd_mono %s" % (comm.rank, omapfile))
	if args.dry: return
	omap, odiv = 0, 0
	tmapfile = omapfile + ".tmp"
	if odivfile: tdivfile = odivfile + ".tmp"
	if len(imapfiles) != 1:
		for imapfile, idivfile in zip(imapfiles, idivfiles):
			imap = read_map(imapfile)
			idiv = read_map(idivfile, slice=".preflat[0]")
			omap += imap*idiv
			odiv += idiv
			del imap, idiv
		mask = odiv > 0
		omap[...,mask] /= odiv[mask]
	else:
		omap = read_map(imapfiles[0])
		odiv = read_map(idivfiles[0], slice=".preflat[0]")
	omap = op(omap, odiv)
	enmap.write_map(tmapfile, omap)
	if odivfile: enmap.write_map(tdivfile, odiv)
	shutil.move(tmapfile, omapfile)
	if odivfile: shutil.move(tdivfile, odivfile)

def copy_plain(ifile, ofile):
	if args.cont and os.path.exists(ofile): return
	if verbose: print("%3d copy_plain %s" % (comm.rank, ofile))
	if args.dry: return
	tfile = ofile + ".tmp"
	shutil.copyfile(ifile, tfile)
	shutil.move(tfile, ofile)

def cat_files(ifiles, ofile):
	if args.cont and os.path.exists(ofile): return
	if verbose: print("%3d cat %s" % (comm.rank, ofile))
	if args.dry: return
	with open(ofile, "w") as ofh:
		for ifile in ifiles:
			with open(ifile, "r") as ifh:
				shutil.copyfileobj(ifh, ofh)

def link(ifile, ofile):
	if args.cont and os.path.exists(ofile): return
	if verbose: print("%d link %s" % (comm.rank, ofile))
	os.symlink(ifile, ofile)

queue = []
def schedule(func, *fargs, **kwargs):
	queue.append([func, fargs, kwargs])

utils.mkdir(args.odir)
# We can now process each dataset
all_files = [os.path.basename(p) for p in glob.glob(args.idir + "/*")]
for iname in sorted(datasets.keys()):
	d = datasets[iname]
	obase = args.odir + "/" + onames[iname]
	# Per split stuff
	for si, sub in enumerate(d):
		ipre = args.idir + "/" + sub.name + "_"
		opre = obase + "_%dway_set%d_" % (len(d),si) + args.suffix
		if "map"  in outputs:
			schedule(coadd_mono, [ipre + "sky_map%04d.fits" % sub.it], [ipre + "sky_div.fits"], opre + "map.fits", opre + "ivar.fits", op=mapfix)
		if "ivar" in outputs and "map" not in outputs:
			schedule(copy_mono, ipre + "sky_div.fits", opre + "ivar.fits", slice=".preflat[0]")
		if "div" in outputs:
			schedule(copy_mono, ipre + "sky_div.fits", opre + "div.fits")
		if "hits"  in outputs:
			schedule(copy_mono, ipre + "sky_hits.fits", opre + "hits.fits")
		if "xlink" in outputs:
			schedule(copy_mono, ipre + "sky_crosslink.fits", opre + "xlink.fits")
		if "sens" in outputs:
			schedule(copy_plain, ipre + "noise.txt", opre + "sens.txt")
	# Coadds
	opre = obase + "_%dway_coadd_" % len(d) + args.suffix
	if "totmap" in outputs:
		imaps = [args.idir + "/" + sub.name + "_sky_map%04d.fits" % sub.it for sub in d]
		idivs = [args.idir + "/" + sub.name + "_sky_div.fits" for sub in d]
		schedule(coadd_mono, imaps, idivs, opre + "map.fits", opre + "ivar.fits", op=mapfix)
	if "tothits" in outputs:
		imaps = [args.idir + "/" + sub.name + "_sky_hits.fits" for sub in d]
		schedule(add_mono, imaps, opre + "hits.fits")
	if "totxlink" in outputs:
		imaps = [args.idir + "/" + sub.name + "_sky_crosslink.fits" for sub in d]
		schedule(add_mono, imaps, opre + "xlink.fits")
	if "toticov" in outputs:
		imaps = [args.idir + "/" + sub.name + "_sky_icov.fits" for sub in d]
		schedule(add_mono,  imaps, opre + "icov.fits", slice=".preflat[0]")
		schedule(copy_plain, args.idir + "/" + d[0].name + "_sky_icov_pix.txt", opre + "icov_pix.txt")
	if "totsens"  in outputs:
		ifiles = [args.idir + "/" + sub.name + "_noise.txt" for sub in d]
		schedule(cat_files, ifiles, opre + "sens.txt")

# Process the scheduled items
np.random.seed(0)
inds = np.random.permutation(len(queue))
for i in range(comm.rank, len(queue), comm.size):
	ind = inds[i]
	func, fargs, kwargs = queue[ind]
	try:
		func(*fargs, **kwargs)
	except Exception as e:
		sys.stderr.write("Exception in %s\n" % func.__name__)
		sys.stderr.write(str(fargs) + "\n")
		sys.stderr.write(str(kwargs) + "\n")
		raise
