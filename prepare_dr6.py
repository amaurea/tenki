import argparse
parser = argparse.ArgumentParser()
parser.add_argument("odir")
parser.add_argument("-c", "--cont", action="store_true")
args = parser.parse_args()
import numpy as np, os
from pixell import enmap, utils, bunch, mpi, wcsutils

release = "dr6.02"

idir  = "/gpfs/fs0/project/r/rbond/sigurdkn/actpol/maps/dr6v4_20230316/release"
cdir  = "/gpfs/fs0/project/r/rbond/sigurdkn/actpol/map_coadd/20240322"
cdir2 = "/gpfs/fs0/project/r/rbond/sigurdkn/actpol/map_coadd/20240323_simple"
mdir  = "/gpfs/fs0/project/r/rbond/sigurdkn/actpol/masks"

arrs    = ["pa5_f090", "pa6_f090", "pa4_f150", "pa5_f150", "pa6_f150", "pa4_f220"]
arrs    = ["pa5_f090"]
gains   = [     1.011,      1.009,         1.,      0.986,      0.970,      1.043]
poleffs = [     0.953,      0.971,         1.,      0.955,      0.968,      0.907]

datasets = [
	["cmb_night",              "std",         "night", "AA", 3, 4, None],
	["cmb_night_null_pwv1",    "null-pwv1",   "night", "AA", 3, 4, None],
	["cmb_night_null_pwv2",    "null-pwv2",   "night", "AA", 3, 4, None],
	["cmb_night_null_el1",     "null-el1",    "night", "AA", 3, 4, None],
	["cmb_night_null_el2",     "null-el2",    "night", "AA", 3, 4, None],
	["cmb_night_null_el3",     "null-el3",    "night", "AA", 3, 4, None],
	["cmb_night_null_t1",      "null-t1",     "night", "AA", 3, 2, None],
	["cmb_night_null_t2",      "null-t2",     "night", "AA", 3, 2, None],
	["cmb_night_null_inout1",  "null-inout1", "night", "AA", 3, 4, None],
	["cmb_night_null_inout2",  "null-inout2", "night", "AA", 3, 4, None],
	["cmb_daydeep",            "std",         "day",   "DS", 3, 4, [[1260,  11039],[ 4109, 24765]]],
	["cmb_daydeep",            "std",         "day",   "DN", 3, 4, [[7831, -10698],[10034,  2213]]],
	["cmb_daywide",            "std",         "day",   "AA", 3, 4, None],
	["galaxy_night",           "std",         "night", "GC", 3, 2, None],
	["bridge_night",           "std",         "night", "BR", 3, 2, None],
	["deep5_night",            "std",         "night", "D5", 3, 4, None],
]

# How about a filename format like
# act_dr6.02_{aa_night,aa_day,dd_day} etc?
# Actually, daydeep should probably be split - it has lots of empty pixels

maptypes = [
	["map",           1, (0,-1,-1), (1, 1,-1), "FREQ-MAP"],
	["map_srcfree",   1, (0,-1,-1), (1, 1,-1), "FREQ-MAP"],
	["ivar",         -2, (0)      , (1)      , "IVAR-MAP"],
	["xlink",        -2, (0, 0, 0), (1, 1, 1), "XLINK-MAP"],
]

comm = mpi.COMM_WORLD
def proot(msg):
	if comm.rank == 0:
		print(msg)

utils.mkdir(args.odir)
# Do the fast stuff in a serial nested loop. Then we will
# do the heavy stuff in a final flattened mpi-loop at the end.
# First set up the standard maps
tasks = []
for di, dset in enumerate(datasets):
	iname, typ, tim, patch, npass, nsplit, pixbox = datasets[di]
	for ai, arr in enumerate(arrs):
		for mi, mtype in enumerate(maptypes):
			splits = ["set%d" % si for si in range(nsplit)]+["coadd"]
			for si, split in enumerate(splits):
				tname, gexp, pexp, sign, extname = mtype
				pexp = np.array(pexp)
				sign = np.array(sign)
				ifname = "%s/%s_%s_%dpass_%dway_%s_%s.fits" % (idir, iname, arr, npass, nsplit, split, tname)
				ofname = "%s/act_%s_%s_%s_%s_%s_%dway_%s_%s.fits" % (args.odir, release, typ, patch, tim, arr, nsplit, split, tname)
				if not os.path.isfile(ifname):
					proot("Warning: %s not found. Skipping" % ifname)
					continue
				if args.cont and os.path.isfile(ofname): continue

				toks = [typ,tim]
				if "day" in tim: toks.append("prelim")
				if patch in ["D5","GC","BR"]: toks.append("bonus")

				extra  = bunch.Bunch(
					TELESCOP = "act",
					INSTRUME = "advact",
					RELEASE  = (release, "Data release tag"),
					SEASON   = ("s17s22", "Observation seasons"),
					PATCH    = (patch, "Survey patch"),
					ARRAY    = (arr.split("_")[0], "Telescope array"),
					FREQ     = (arr.split("_")[1], "Frequency tag"),
					ACTTAGS  = ",".join(toks),
					BUNIT    = ("uK" if gexp == 1 else "uK^%.0f" % gexp, "Physical (pixel) units"),
					EXTNAME  = (extname, "Extension name"),
				)
				# Only include polcconv for cases where it makes sense, to
				# avoid incorrect flipping of the U axis for cases where it
				# doesn't matter
				if sign.ndim == 1 and len(sign) == 3 and sign[2] < 0:
					extra["POLCCONV"] = ("IAU", "Polarization convention")
				tasks.append(bunch.Bunch(
					ifname=ifname, ofname=ofname, extra=extra, pixbox=pixbox,
					mul=(gains[ai]**gexp * poleffs[ai]**pexp * sign)[...,None,None]))

# Next set up the coadd maps. Here we have map types map[3], map_srcfree[3] and ivar[3],
# and datasets {act,act_planck}, {s08,s17}_{f090,f150,f220}_{night,daynight}
bands = ["f090", "f150", "f220"]
times = ["night", "daynight"]
teles = ["act", "act_planck"]
seasons = ["s17_s22", "s08_s22"]
sdirs   = [cdir, cdir2]
svers   = ["dr6.02", "dr4dr6"]
maptypes = [
	["map",           1, (0,-1,-1), (1, 1,-1), "FREQ-MAP"],
	["map_srcfree",   1, (0,-1,-1), (1, 1,-1), "FREQ-MAP"],
	["ivar",         -2, (0)      , (1)      , "IVAR-MAP"],
]

for si, season in enumerate(seasons):
	for tei, tele in enumerate(teles):
		for ti, tim in enumerate(times):
			for bi, band in enumerate(bands):
				for mi, mtype in enumerate(maptypes):
					tname, gexp, pexp, sign, extname = mtype
					pexp = np.array(pexp)
					sign = np.array(sign)
					ifname = "%s/%s_%s_%s_%s_%s.fits" % (sdirs[si], tele, season, band, tim, tname)
					ofname = "%s/%s_%s_coadd_AA_%s_%s_%s.fits" % (args.odir, tele.replace("_","-"), svers[si], tim, band, tname)
					if not os.path.isfile(ifname):
						proot("Warning: %s not found. Skipping" % ifname)
						continue
					if args.cont and os.path.isfile(ofname): continue

					toks = [tim]
					if "day" in tim: toks += ["prelim"]
					instruments = []
					if "s08" in season: instruments += ["mbac","actpol"]
					instruments.append("advact")
					if "planck" in teles: instruments += ["planckHFI"]

					extra  = bunch.Bunch(
						TELESCOP = tele.replace("_","+"),
						INSTRUME = "+".join(instruments),
						RELEASE  = (svers[si], "Data release tag"),
						SEASON   = (season.replace("_",""), "Observation seasons"),
						PATCH    = ("AA", "Survey patch"),
						FREQ     = (arr.split("_")[1], "Frequency tag"),
						ACTTAGS  = ",".join(toks),
						BUNIT    = ("uK" if gexp == 1 else "uK^%.0f" % gexp, "Physical (pixel) units"),
						EXTNAME  = (extname, "Extension name"),
					)
					if sign.ndim == 1 and len(sign) == 3 and sign[2] < 0:
						extra["POLCCONV"] = ("IAU", "Polarization convention")
					tasks.append(bunch.Bunch(
						ifname=ifname, ofname=ofname, extra=extra, pixbox=None,
						mul=np.array(sign)[...,None,None]))

# Other maps
def add_other(ifname, ofname, pixbox=None, mul=1, extra=None):
	if not os.path.isfile(ifname):
		return proot("Warning: %s not found. Skipping" % ifname)
	if args.cont and os.path.isfile(ofname): return
	tasks.append(bunch.Bunch(
		ifname=ifname, ofname=ofname, extra=extra, mul=mul, pixbox=pixbox))
add_other("%s/srcsamp/srcsamp_mask_fejer1.fits" % mdir, "%s/srcsamp_mask.fits" % args.odir, extra=bunch.Bunch(RELEASE=(release, "Data release tag"), PATCH=("AA", "Survey patch"), EXTNAME=("srcsamp-mask")))

# Finally loop over everything with mpi
for ind in range(comm.rank, len(tasks), comm.size):
	task = tasks[ind]
	map    = enmap.read_map(task.ifname, pixbox=task.pixbox)
	map   *= task.mul
	if task.pixbox is not None:
		# Make sure crval isn't too far away
		map.wcs = wcsutils.fix_wcs(map.wcs, n=map.shape[-1])
	enmap.write_map(task.ofname, map, extra=task.extra, allow_modify=True)
	del map
	print("Wrote %s" % task.ofname)
