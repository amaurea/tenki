import numpy as np, argparse
from enlib import enmap
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs=2)
parser.add_argument("ofile")
parser.add_argument("-b", "--binsize", type=int, default=3)
parser.add_argument("-s", "--smooth",  type=float, default=30)
parser.add_argument("--div", type=float, default=1.0)
args = parser.parse_args()
b  = args.binsize
smooth = args.smooth * np.pi/180/60/(8*np.log(2))
m  = [enmap.read_map(f) for f in args.ifiles]
dm = (m[1]-m[0])/2

pixarea = dm.area()/np.product(dm.shape[-2:])*(180*60/np.pi)**2


# Compute standard deviation in bins
dm = dm[...,:dm.shape[-2]/b*b,:dm.shape[-1]/b*b]
dm_blocks = dm.reshape(dm.shape[:-2]+(dm.shape[-2]/b,b,dm.shape[-1]/b,b))
var  = np.std(dm_blocks,axis=(-3,-1))**2*pixarea/args.div
# This reshaping stuff messes up the wcs, which doesn't notice
# that we now have bigger pixels. So correct that.
var  = enmap.samewcs(var, dm[...,::b,::b])

typ = np.median(var[var!=0])
var[~np.isfinite(var)] = 0
var = np.minimum(var,typ*1e6)

svar = enmap.smooth_gauss(var, smooth)
sigma = svar**0.5

enmap.write_map(args.ofile, sigma)
