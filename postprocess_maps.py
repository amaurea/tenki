import numpy as np, argparse, glob, re, os, shutil
from enlib import enmap, utils, retile, bunch, mpi
parser = argparse.ArgumentParser()
parser.add_argument("idir")
parser.add_argument("odir")
parser.add_argument("-i", "--iteration", type=str, default="individual")
parser.add_argument("-a", "--allow-nonstandard",   action="store_true")
parser.add_argument("-c", "--cont",                action="store_true")
parser.add_argument("-O", "--output",    type=str, default="map,ivar,sens,xlink,totmap,totsens,totxlink")
args = parser.parse_args()

comm = mpi.COMM_WORLD
outputs = set(args.output.split(","))

# Look for map files in the input directory
datasets = {}
for fname in glob.glob(args.idir + "/*map????.fits"):
	fname = os.path.basename(fname)
	m = re.search(r"^(.*)_(\d+)way_(\d+)_sky_map(\d\d\d\d).fits", fname)
	if not m:
		if comm.rank == 0: print "Skipping unrecognized map name: " + fname
		continue
	prefix = m.group(1)
	nway   = int(m.group(2))
	sub    = int(m.group(3))
	it     = int(m.group(4))
	if prefix not in datasets:
		datasets[prefix] = [bunch.Bunch(it=0, name="") for i in range(nway)]
	datasets[prefix][sub].it   = max(datasets[prefix][sub].it, it)
	datasets[prefix][sub].name = "_".join(fname.split("_")[:-2])

nchar = max([len(key) for key in datasets])

# Find the iteration count to use for each dataset
for key in sorted(datasets.keys()):
	d     = datasets[key]
	minit = min([sub.it for sub in d])
	if   args.iteration == "individual": pass
	elif args.iteration == "min":
		for i in range(len(d)): d[i].it = minit
	else:
		try:
			for i in range(len(d)):
				d[i].it = int(args.iteration)
		except ValueError:
			raise ValueError(args.iteration)
	
	# Do we have any zeros at this point? If so, warn and skip the dataset
	nzero = sum([sub.it == 0 for sub in d])
	if nzero > 0:
		if comm.rank == 0: print "%-*s only has %d/%d splits. Skipping" % (nchar, key, len(d)-nzero, len(z))
		del datasets[key]
	if comm.rank == 0:
		print "%-*s using" % (nchar, key) + " %4d"*len(d) % tuple([sub.it for sub in d])

# Check if we follow the standard format
onames = {}
for key in datasets.keys():
	m = re.match(r"s\d\d_\w+_pa\d_f\d\d\d_(no)?hwp_(day|night|daynight)\b.*", key)
	if not m:
		if comm.rank == 0:
			print "%s does not follow the standard name format" % key
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
	return map

def copy_mono(ifile, ofile, slice=None):
	if args.cont and os.path.exists(ofile): return
	tfile = ofile + ".tmp"
	map   = read_map(ifile, slice=slice)
	enmap.write_map(tfile, map)
	shutil.move(tfile, ofile)

def add_mono(ifiles, ofile, slice=None):
	if args.cont and os.path.exists(ofile): return
	tfile = ofile + ".tmp"
	omap  = read_map(ifiles[0], slice=slice)
	for ifile in ifiles[1:]:
		omap += read_map(ifile, slice=slice)
	enmap.write_map(tfile, omap)
	shutil.move(tfile, ofile)

def coadd_mono(imapfiles, idivfiles, omapfile, odivfile):
	if args.cont and os.path.exists(omapfile): return
	omap, odiv = None, None
	tmapfile = omapfile + ".tmp"
	tdivfile = odivfile + ".tmp"
	for imapfile, idivfile in zip(imapfiles, idivfiles):
		imap = read_map(imapfile)
		idiv = read_map(idivfile, slice=".preflat[0]")
		if omap is None: omap, odiv = imap*0, idiv*0
		omap += imap*idiv
		odiv += idiv
	mask = odiv > 0
	omap[...,mask] /= odiv[mask]
	enmap.write_map(tmapfile, omap)
	enmap.write_map(tdivfile, odiv)
	shutil.move(tmapfile, omapfile)
	shutil.move(tdivfile, odivfile)

def copy_plain(ifile, ofile):
	if args.cont and os.path.exists(ofile): return
	tfile = ofile + ".tmp"
	shutil.copyfile(ifile, tfile)
	shutil.move(tfile, ofile)

def cat_files(ifiles, ofile):
	if args.cont and os.path.exists(ofile): return
	with open(ofile, "w") as ofh:
		for ifile in ifiles:
			with open(ifile, "r") as ifh:
				shutil.copyfileobj(ifh, ofh)

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
		opre = obase + "_%dway_set%d_" % (len(d),si)
		print ipre, opre
		if "ivar" in outputs:
			schedule(copy_mono, ipre + "sky_div.fits", opre + "ivar.fits", slice=".preflat[0]")
		if "map"  in outputs:
			schedule(copy_mono, ipre + "sky_map%04d.fits" % sub.it, opre + "map.fits")
		if "hits"  in outputs:
			schedule(copy_mono, ipre + "sky_hits.fits", opre + "hits.fits")
		if "xlink" in outputs:
			schedule(copy_mono, ipre + "sky_crosslink.fits", opre + "xlink.fits", slice="[1:]")
		if "icov" in outputs:
			schedule(copy_mono,  ipre + "sky_icov.fits", opre + "icov.fits", slice=".preflat[0]")
			schedule(copy_plain, ipre + "sky_icov_pix.txt", opre + "icov_pix.txt")
		if "sens" in outputs:
			schedule(copy_plain, ipre + "noise.txt", opre + "sens.txt")
	# Coadds
	opre = obase + "_%dway_coadd_" % len(d)
	if "totmap" in outputs:
		imaps = [args.idir + "/" + sub.name + "_sky_map%04d.fits" % sub.it for sub in d]
		idivs = [args.idir + "/" + sub.name + "_sky_div.fits" for sub in d]
		schedule(coadd_mono, imaps, idivs, opre + "map.fits", opre + "ivar.fits")
	if "totxlink" in outputs:
		imaps = [args.idir + "/" + sub.name + "_sky_crosslink.fits" for sub in d]
		schedule(add_mono, imaps, opre + "xlink.fits", slice="[1:]")
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
	print comm.rank, func.__name__, fargs
	func(*fargs, **kwargs)
