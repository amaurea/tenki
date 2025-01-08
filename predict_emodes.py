import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("ispec")
parser.add_argument("omap")
parser.add_argument("-l", "--lmax", type=int, default=10000)
args = parser.parse_args()
import numpy as np, time
from pixell import enmap, utils, curvedsky, powspec

imap = enmap.read_map(args.imap)
C    = powspec.read_spectrum(args.ispec)

# For each lm we have a [TT,TE,ET,EE]-covmat. We want the maximum-likelihood estimate
# for E given T only.
#  -2logL = log|2piC| + a'C"a = log|2piC| + t C"_TT t + 2 t C"_TE e + e C"_EE e
#    = log|2piC| + (e + C"_EE" C"_TE t)'C"_EE(e + ...) - t'C"_TE C"_EE" C"_TE t + t'C"_TT t
# So e|t is Gaussian distributed with expectation value
#  ê = -C"_EE" C"_TE t
# and covariance
#  Ê = C"_EE"

iC   = utils.eigpow(C, -1, axes=(0,1))
f    = np.zeros_like(iC[0,0])
mask = iC[1,1]>0
f[mask] = -1/iC[1,1,mask]*iC[1,0,mask]

alm    = curvedsky.map2alm(imap, lmax=args.lmax)
alm[1] = curvedsky.almxfl(alm[0], f)
alm[2] = 0

omap = curvedsky.alm2map(alm, np.zeros_like(imap))
omap[0] = imap[0]
enmap.write_map(args.omap, omap)
