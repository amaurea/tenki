"""Given a set of maps representing independent observations of
the the same patch of sky, along with hitcounts map for each of
these, estimates the white noise level in the maps and the conversion
factor between hitcounts and noise (e.g. the sensitivity). Uses this
to output calibrated hitcount maps in the unit inverse variance per
pixel."""
from __future__ import division, print_function
import numpy as np, argparse, sys
from enlib import enmap, utils
parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", help="map hit odiv map hit odiv ..., or map map ... hit hit ... odiv odiv ... if -T is passed")
parser.add_argument("-b", "--blocksize", type=int, default=3)
parser.add_argument("--srate", type=float, default=400)
parser.add_argument("-T", "--transpose", action="store_true")
parser.add_argument("-D", "--dry-run",   action="store_true")
parser.add_argument("-v", "--verbose",   action="store_true")
parser.add_argument("-c", "--component", type=int, default=0)
args = parser.parse_args()
hitlim = 0.05
bs = args.blocksize

nmap = len(args.files)//3
if args.transpose:
	mapfiles = args.files[nmap*0:nmap*1]
	hitfiles = args.files[nmap*1:nmap*2]
	divfiles = args.files[nmap*2:nmap*3]
else:
	mapfiles = args.files[0::3]
	hitfiles = args.files[1::3]
	divfiles = args.files[2::3]

def map_to_blocks(map, n):
	m = map[...,:map.shape[-2]//n*n,:map.shape[-1]//n*n]
	m = m.reshape(m.shape[:-2]+(m.shape[-2]//n,n,m.shape[-1]//n,n))
	return m

def calc_map_block_ivar(map, n):
	m = map_to_blocks(map, n)
	vmap = np.var(m, axis=(-3,-1))
	vmap[vmap!=0] = 1/vmap[vmap!=0]
	return enmap.samewcs(vmap, map[...,::n,::n])
def calc_map_block_mean(map, n):
	m = map_to_blocks(map, n)
	return enmap.samewcs(np.mean(m, axis=(-3,-1)), map[...,::n,::n])

print("Reading map %s" % (mapfiles[0]))
map = enmap.read_map(mapfiles[0]).preflat[args.component]
print("Reading hit %s" % (hitfiles[0]))
hit = enmap.read_map(hitfiles[0])
# We assume that the hitcount maps are 2d
def get_bias(bs):
	# Get bias factor we get from estimating the ratio
	# via medmean. Determined empirically based on white noise
	# simulations. We will also have biases from not actually having
	# white noise, but those will be map-dependent.
	biases = [0, 0, 0.74, 0.89, 0.94, 0.96, 0.97, 0.98, 0.99, 0.99, 0.99]
	return 1.0 if bs >= len(biases) else biases[bs]

def quant(a, q): return np.percentile(a, q*100)
qlim = 0.95

print("Measuring sensitivities")
ratios = []
for i, (mapfile, hitfile) in enumerate(zip(mapfiles[1:], hitfiles[1:])):
	print("Reading map %s" % mapfile)
	map2 = enmap.read_map(mapfile).preflat[args.component]
	print("Reading hit %s" % hitfile)
	hit2 = enmap.read_map(hitfile)
	# Compute variances for the current map minus the previous map
	dmap = (map2-map)/2
	dhit = hit2+hit
	vmap = calc_map_block_ivar(dmap, bs)
	mask  = (hit>quant(hit,qlim)*hitlim) & (hit2>quant(hit2,qlim)*hitlim)
	mask &= (hit<quant(hit,qlim)) & (hit2<quant(hit2,qlim))
	# Reduce dhit and mask to vmap's resolution
	dhit = calc_map_block_mean(dhit, bs)
	mask = calc_map_block_mean(mask, bs)>0
	ratio = utils.medmean(vmap[mask]/dhit[mask])
	# And compute the sensitivity
	ratio *= get_bias(bs)**2
	ratios.append(ratio)
	sens = (ratio*args.srate)**-0.5
	print("%d-%d %7.2f %7.2f" % (i+1,i, ratio, sens))
	map, hit = map2, hit2

# Ratio has units 1/(uK^2*sample), and gives us the conversion
# factor between hitmaps and inverse variance maps, which we call
# div maps by convention from tenki.
ratio = np.mean(ratios)
print("mean %7.2f" % (ratio*args.srate)**-0.5)

if args.dry_run: sys.exit()

# Ok, now that we have the mean conversion factor between hits and divs,
# loop through all hits again, and output div maps
for i, (hitfile, divfile) in enumerate(zip(hitfiles, divfiles)):
	hit = enmap.read_map(hitfile)
	div = hit*ratio
	print("Writing %s" % divfile)
	enmap.write_map(divfile, div)
