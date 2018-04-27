import numpy as np, argparse
from enlib import enmap, utils, bench
parser = argparse.ArgumentParser()
parser.add_argument("div")
parser.add_argument("ofile", nargs="?", default="/dev/stdout")
parser.add_argument("-d", "--downgrade", type=int, default=1)
parser.add_argument("-t", "--thin", type=int, default=1000)
parser.add_argument("-A", "--area-model", type=str, default="exact", help="How to model pixel area. exact: Compute shape of each pixel. average: Use a single average number for all")
args = parser.parse_args()

div = enmap.read_fits(args.div)
if args.downgrade:
	div  = enmap.downgrade(div, args.downgrade)
	div *= args.downgrade**2

div = div.reshape((-1,)+div.shape[-2:])[0]
# Individual pixel area
if args.area_model == "average":
	pix_area = div*0 + div.area()/div.size*(180*60/np.pi)**2
else:
	pos   = div.posmap()
	diffs = utils.rewind(pos[:,1:,1:]-pos[:,:-1,:-1],0)
	pix_area = np.abs(diffs[0]*diffs[1])*np.cos(pos[0,:-1,:-1])
	del diffs
	# Go to square arcmins
	pix_area /= utils.arcmin**2
	# Pad to recover edge pixels
	pix_area = np.concatenate([pix_area,pix_area[-1:]],0)
	pix_area = np.concatenate([pix_area,pix_area[:,-1:]],1)

# Flatten everything
div = div.reshape(-1)
pix_area = pix_area.reshape(-1)

with utils.nowarn():
	rms  = (pix_area/div)**0.5
	inds = np.argsort(rms, axis=None)
	rms  = rms[inds]
	area = np.cumsum(pix_area[inds])/3600
	mask = np.isfinite(rms)
	rms, area = rms[mask], area[mask]

np.savetxt(args.ofile, np.array([area[::args.thin],rms[::args.thin]]).T, fmt="%9.3f %15.4f")
