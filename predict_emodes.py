import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("ispec")
parser.add_argument("omap")
parser.add_argument("-l", "--lmax", type=int, default=10000)
parser.add_argument("-m", "--mode", type=str, default="auto")
parser.add_argument("-s", "--spin", type=int, default=2)
args = parser.parse_args()
import numpy as np, time
from pixell import enmap, utils, curvedsky, powspec, uharm

imap = enmap.read_map(args.imap).preflat[0]
C    = powspec.read_spectrum(args.ispec)

# For each lm we have a [TT,TE,ET,EE]-covmat. We want the maximum-likelihood estimate
# for E given T only.
#  -2logL = log|2piC| + a'C"a = log|2piC| + t C"_TT t + 2 t C"_TE e + e C"_EE e
#    = log|2piC| + (e + C"_EE" C"_TE t)'C"_EE(e + ...) - t'C"_TE C"_EE" C"_TE t + t'C"_TT t
# So e|t is Gaussian distributed with expectation value
#  ê = -C"_EE" C"_TE t
# and covariance
#  Ê = C"_EE"

spin = 0 if args.spin == 0 else [0,args.spin]

# Prepare our T→E predicting filter
iC   = utils.eigpow(C, -1, axes=(0,1))
f    = np.zeros_like(iC[0,0])
mask = iC[1,1]>0
f[mask] = -1/iC[1,1,mask]*iC[1,0,mask]

# Apply it to our T map to get an E predction
uht  = uharm.UHT(imap.shape, imap.wcs, mode=args.mode, lmax=args.lmax)
iharm = uht.map2harm(imap)
eharm = uht.hmul(uht.lprof2hprof(f), iharm)
# Build full TEB harm
oharm = np.zeros_like(iharm, shape=(3,)+iharm.shape)
oharm[0] = iharm
oharm[1] = eharm
# Transform to TQU
omap = uht.harm2map(oharm, spin=spin)
# And output
enmap.write_map(args.omap, omap)
