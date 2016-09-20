"""Given a Ninkasi output directory, combines splits, input-used, beams etc. To
form a full, coadded map."""
import numpy as np, argparse, os, re
from enlib import enmap, utils
parser = argparse.ArgumentParser()
parser.add_argument("idir")
parser.add_argument("odir", nargs="?", default=".")
parser.add_argument("-I", "--iteration", type=str, default="auto")
parser.add_argument("-s", "--single", action="count", default=0)
parser.add_argument("-t", "--total",  action="count", default=1)
parser.add_argument("--wpoly", type=str, default="")
parser.add_argument("--set",   type=str, default="_set_")
args = parser.parse_args()

utils.mkdir(args.odir)

fnames = os.listdir(args.idir)
# Identify which datasets are included. There is one for each set_0 weights_I.fits file
groups = {}
for fname in fnames:
	m = re.match(r"(.*)%s[0-9]_(.*)weights_I.fits" % args.set, fname)
	if not m:
		m = re.match(r"(.*)%s[0-9]_(.*)div.fits" % args.set, fname)
		if not m: continue
	if re.match(r".*single_weights_I.fits", fname): continue
	pre = m.group(1)
	post = m.group(2)
	tag = pre + args.set + "%s_" + post
	if tag not in groups:
		groups[tag] = {"nsub": 0}
	groups[tag]["nsub"] += 1

def vread(fname):
	print "Reading %s" % fname
	return enmap.read_map(fname)
def vwrite(fname, m):
	print "Writing %s" % fname
	enmap.write_map(fname, m)

tags = sorted(groups.keys())
print "Found %d map sets" % len(tags)
# Process each group
for tag in tags:
	group = groups[tag]
	nsub = group["nsub"]
	# Gather information about each
	maxit, iused, beam, noise, div = np.inf, np.inf, np.inf, np.inf, np.inf
	for sub in range(nsub):
		stag = tag % sub
		smaxit, siused, sbeam, snoise, sdiv = 0,0,0,0,0
		for fname in fnames:
			m = re.match(stag + args.wpoly + "([0-9]+)_I.fits", fname)
			if m: smaxit = max(smaxit, int(m.group(1)))
			m = re.match(stag + "input_used_I.fits", fname)
			if m: siused = 1
			m = re.match(stag + "beam_test.fits", fname)
			if m: sbeam = 1
			m = re.match(stag + "noise.fits", fname)
			if m: snoise = 1
			m = re.match(stag + "div.fits", fname)
			if m: sdiv = 1
	maxit = int(min(maxit, smaxit))
	iused = int(min(iused, siused))
	beam  = int(min(beam,  sbeam))
	noise = int(min(noise, snoise))
	div   = int(min(div,   sdiv))
	step  = maxit if args.iteration == "auto" else int(args.iteration)
	# Read in each map for each group
	tmap = None
	tweight = None
	for sub in range(nsub):
		pre = args.idir + "/" + tag % sub
		opre = args.odir + "/" + tag % sub
		# Weight map
		if div:
			weight = vread(pre + "div.fits")
			if weight.ndim == 4: weight = weight[0,0]
		elif noise:
			weight = vread(pre + "noise.fits")
			weight[weight>0] = weight[weight>0]**-2
		else:
			weight = vread(pre + "weights_I.fits")
		# CMB map
		map = [vread(pre + args.wpoly + "%d_%s.fits" % (step, comp)) for comp in ["I","Q","U"]]
		map = enmap.enmap(map, map[0].wcs)
		# Input used, if available
		if iused and not args.wpoly:
			for i, comp in enumerate(["I","Q","U"]):
				base = vread(pre + "input_used_%s.fits" % comp)
				map[i,weight>0] += base[weight>0]
		nosrc = map.copy()
		# Point source template, if available
		if beam:
			srcs = vread(pre + "beam_test.fits")
			map[0][weight>0] += srcs[weight>0]
			if args.single % 2 > 0:
				vwrite(opre + "nosrc_%04d.fits" % step, nosrc)
		if args.single % 2 > 0:
			vwrite(opre + "map_%04d.fits"   % step, map)
			vwrite(opre + "div.fits", weight)
		if tmap is None:
			tmap = map*0
			tnosrc = nosrc*0
			tweight = weight*0
		tmap += map*weight[None]
		tnosrc += nosrc*weight[None]
		tweight += weight
	tmap[:,tweight>0] /= tweight[None][:,tweight>0]
	tnosrc[:,tweight>0] /= tweight[None][:,tweight>0]
	opre = args.odir + "/" + tag % "tot"
	if args.total % 2 > 0:
		if beam:
			vwrite(opre + "nosrc_%04d.fits" % step, tnosrc)
		vwrite(opre + "map_%04d.fits" % step, tmap)
		vwrite(opre + "div.fits", tweight)
