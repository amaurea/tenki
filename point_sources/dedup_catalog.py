import argparse
parser = argparse.ArgumentParser()
parser.add_argument("icat")
parser.add_argument("ocat")
parser.add_argument("-b", "--beam",    type=str,   default="1.4", help="Beam in arcmin or beam(l)")
parser.add_argument("-s", "--snmin",   type=float, default=5)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument(      "--bscale",  type=float, default=1)
args = parser.parse_args()
import numpy as np
from enlib import utils, dory

beam      = dory.get_beam(args.beam)
beam_prof = dory.get_beam_profile(beam)
beam_area = dory.calc_beam_profile_area(beam_prof)

# Apply ad-hoc beam scaling
beam_prof[0] *= args.bscale

cat   = dory.read_catalog(args.icat)
flux  = dory.eval_flux_at_srcs(cat, beam_prof, verbose=args.verbose)
nread = len(cat)

cat.status = 0
cat.status[flux<0] = 2
cat.status[(flux>0)&(cat.flux[:,0]>flux*0.5)] = 1
# Reject anything weaker than snmin
cat.status[np.abs(cat.flux[:,0]/cat.dflux[:,0])<args.snmin] = 0
nbad, nsrc, nsz = np.bincount(cat.status)[:3]
cat = cat[cat.status>0]

print("%d read, %d cut, %d src, %d sz" % (nread, nbad, nsrc, nsz))
dory.write_catalog(args.ocat, cat)
