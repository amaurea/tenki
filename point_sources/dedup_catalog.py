import argparse
parser = argparse.ArgumentParser()
parser.add_argument("icat")
parser.add_argument("ocat")
parser.add_argument("-b", "--beam",    type=str,   default="1.4", help="Beam in arcmin or beam(l)")
parser.add_argument("-s", "--snmin",   type=float, default=5)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-S", "--sort",    action="store_true")
parser.add_argument(      "--bscale",  type=float, default=1)
parser.add_argument(      "--artrad",  type=float, default=20)
parser.add_argument(      "--artnum",  type=float, default=7)
parser.add_argument(      "--artpen",  type=float, default=2)
args = parser.parse_args()
import numpy as np
from enlib import utils, dory

beam      = dory.get_beam(args.beam)
beam_prof = dory.get_beam_profile(beam)
beam_area = dory.calc_beam_profile_area(beam_prof)

# Apply ad-hoc beam scaling
beam_prof[0] *= args.bscale

cat   = dory.read_catalog(args.icat)
nread = len(cat)

# We will remove sources that are weaker than the surroundings' contribution that area, so
# get the total flux at each source's position
flux  = dory.eval_flux_at_srcs(cat, beam_prof, verbose=args.verbose)
# We will also penalize source detection in areas with too many sources. We can
# do this with the same function, if we modify the beam a bit
r_dummy   = np.linspace(0, args.artrad*utils.arcmin, 10000)
b_dummy   = r_dummy*0+1; b_dummy[-1] = 0
cat_dummy = cat.copy()
sn        = np.abs(cat.amp[:,0])/cat.damp[:,0]
cat_dummy.flux[:,0] = sn > args.snmin
nnear = dory.eval_flux_at_srcs(cat_dummy, np.array([r_dummy,b_dummy]), verbose=args.verbose)
nmax1 = np.max(nnear)
# Use nnear to get a per-source S/N threshold
snmin = np.where(nnear < args.artnum, args.snmin, args.snmin*args.artpen)
nart  = np.sum(nnear >= args.artnum)

cat.status = 0
cat.status[flux<0] = 2
cat.status[(flux>0)&(cat.flux[:,0]>flux*0.5)] = 1
# Reject anything weaker than snmin
cat.status[np.abs(cat.flux[:,0]/cat.dflux[:,0])<snmin] = 0
nbad, nsrc, nsz = np.bincount(cat.status, minlength=3)[:3]

#cat_dummy.flux[:,0] = cat.status > 0
#nnear2 = dory.eval_flux_at_srcs(cat_dummy, np.array([r_dummy,b_dummy]), verbose=args.verbose)
#nmax2 = np.max(nnear2)
#inds = np.argsort(nnear2)[::-1][:2000]
#for i, ind in enumerate(inds):
#	print("%5d %5d %5.0f  %8.3f %8.3f %8.3f" % (i, ind, nnear2[ind], cat_dummy.ra[ind]/utils.degree, cat_dummy.dec[ind]/utils.degree, cat.amp[ind,0]/cat.damp[ind,0]))

cat = cat[cat.status>0]
if args.sort:
	sn = cat.amp[:,0]/cat.damp[:,0]
	cat = cat[np.argsort(sn)[::-1]]

print("%d read, %d art, %d cut, %d src, %d sz" % (nread, nart, nbad, nsrc, nsz))
dory.write_catalog(args.ocat, cat)
