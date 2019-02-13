# Project a single pickup map into horizontal coordinates
import numpy as np, os
from enlib import enmap, utils, config, array_ops
from enact import actdata, filedb
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("pickup_map")
parser.add_argument("template")
parser.add_argument("sel_repr")
parser.add_argument("el", type=float)
parser.add_argument("ofile")
args  = parser.parse_args()

filedb.init()
nrow, ncol = 33, 32
# Read our template, which represents the output horizontal coordinates
template = enmap.read_map(args.template)
# Use our representative selector to get focalplane offsets and polangles
entry = filedb.data[filedb.scans[args.sel_repr][0]]
d = actdata.read(entry, ["boresight", "point_offsets", "polangle"])
d.boresight[2] = args.el # In degrees, calibrated in next step
d = actdata.calibrate(d, exclude=["autocut"])

def reorder(map, nrow, ncol, dets):
	return enmap.samewcs(map[utils.transpose_inds(dets,nrow,ncol)],map)

# Read our map, and give each row a weight
pickup = enmap.read_map(args.pickup_map)
pickup = reorder(pickup, nrow, ncol, d.dets)
weight = np.median((pickup[:,1:]-pickup[:,:-1])**2,-1)
weight[weight>0] = 1/weight[weight>0]

# Find the output pixel for each input pixel
baz = pickup[:1].posmap()[1,0]
bel = baz*0 + args.el * utils.degree
ipoint = np.array([baz,bel])

opoint = ipoint[:,None,:] + d.point_offset.T[:,:,None]
opix   = template.sky2pix(opoint[::-1]).astype(int) # [{y,x},ndet,naz]
opix   = np.rollaxis(opix, 1) # [ndet,{y,x},naz]

omap = enmap.zeros((3,)+template.shape[-2:], template.wcs)
odiv = enmap.zeros((3,3)+template.shape[-2:], template.wcs)
for det in range(d.ndet):
	omap += utils.bin_multi(opix[det], template.shape[-2:], weight[det]*pickup[det]) * d.det_comps[det,:,None,None]
	odiv += utils.bin_multi(opix[det], template.shape[-2:], weight[det]) * d.det_comps[det,:,None,None,None] * d.det_comps[det,None,:,None,None]

odiv = enmap.samewcs(array_ops.eigpow(odiv,   -1, axes=[0,1]), odiv)
omap = enmap.samewcs(array_ops.matmul(odiv, omap, axes=[0,1]), omap)
enmap.write_map(args.ofile, omap)
