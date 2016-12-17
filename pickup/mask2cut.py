import numpy as np, argparse
from enlib import enmap, utils
parser = argparse.ArgumentParser()
parser.add_argument("imasks", nargs="+")
args = parser.parse_args()

maps = [enmap.read_map(fname) for fname in args.imasks]
nrow, ncol = 33, 32

for r in range(nrow):
	for c in range(ncol):
		uid = c + ncol*r
		i   = r + nrow*c
		# Loop through each map for this detector
		az_ranges = []
		for mi, map in enumerate(maps):
			pix_ranges = utils.mask2range(map[i]>0)
			if len(pix_ranges) == 0: continue
			az_ranges.append(map.pix2sky([pix_ranges*0,pix_ranges])[1]/utils.degree)
		if len(az_ranges) == 0: az_ranges = np.zeros((1,0,2))
		az_ranges = np.concatenate(az_ranges,0)
		order = np.argsort(az_ranges[:,0])
		az_ranges = az_ranges[order]
		# And output our cut
		cut = "%3d  " % uid + " ".join(["%.2f:%.2f" % tuple(a) for a in az_ranges])
		print cut
