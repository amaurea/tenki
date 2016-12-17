"""This program reads in a pickup map and outputs a mask specifying the
regions to cut. This mask can then be turned into a per-scanning-pattern
cut list in mask2cut"""
import numpy as np, argparse
from scipy import ndimage
from enlib import enmap, fft
parser = argparse.ArgumentParser()
parser.add_argument("imap")
parser.add_argument("omap")
parser.add_argument("-T", "--threshold", type=float, default=500)
parser.add_argument("-H", "--high-threshold", type=float, default=4000)
parser.add_argument("-W", "--minwidth",  type=int,   default=2)
parser.add_argument("-w", "--widen",     type=int,   default=3)
parser.add_argument("-m", "--maxmean",   type=int,   default=250)
args = parser.parse_args()

imap = enmap.read_map(args.imap)
ngood = int(np.sum(np.any(imap!=0,1)))

# Smooth a bit in az direction to reduce noise
m  = imap
fm = fft.redft00(m)
nf = fm.shape[1]
fm[:,nf/4:] = 0
fft.redft00(fm, m, normalize=True)

def fixwidth(row):
	row = ndimage.distance_transform_edt(row) > args.minwidth
	if np.any(row):
		row = ndimage.distance_transform_edt(1-row) < args.minwidth + args.widen
	return row

# Look for areas where abs value is above threshold
mask = np.abs(m) > args.threshold
# Shrink and grow a bit to avoid single-pixel masking
for row in mask: row[:] = fixwidth(row)

# Make sure very high values are masked anyway
mask2 = np.abs(m) > args.high_threshold

# Also look at the mean value per az, to find fainter excesses that
# would still tend to coadd into the same location in the maps.
# But prevent single high values from dominating by capping the value
mval = np.sum(np.minimum(np.abs(m), args.threshold), 0)/ngood
mask3= fixwidth(mval > args.maxmean)
mask3 = np.tile(mask3, (m.shape[0],1))

# Output an image indicating what to mask for each reason.
omap = mask + 2*mask2 + 4*mask3

enmap.write_map(args.omap, omap)
