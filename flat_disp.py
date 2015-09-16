# This program computes a simple estimation of the amount of distortion the
# flat-sky approximation involves
import numpy as np, argparse
from enlib import enmap
parser = argparse.ArgumentParser()
parser.add_argument("omap")
parser.add_argument("-D", "--diameter", type=float, default=30)
parser.add_argument("-n", "--npix",     type=int,   default=800)
parser.add_argument("--proj",           type=str,   default="cea")
args = parser.parse_args()

r = args.diameter*np.pi/180/2
shape, wcs = enmap.geometry(pos=[[-r,-r],[r,r]], shape=(args.npix, args.npix), proj=args.proj)

def linpos(n,b): return b[0] + (np.arange(n)+0.5)*(b[1]-b[0])/n
alpha = enmap.zeros((2,)+shape, wcs)
pos   = enmap.posmap(shape, wcs)
# I'm not sure how to compute this in general, so here's a specialization
# for cylindrical projections
if args.proj == "cea" or args.proj =="car":
	dec   = pos[0,:,0]
	ra    = pos[1,0,:]
	lindec= linpos(shape[0],enmap.box(shape,wcs)[:,0])
	scale = 1/np.cos(dec)-1
	alpha[0] = (lindec-dec)[:,None]
	alpha[1] = ra[None,:]*scale[:,None]
enmap.write_map(args.omap, alpha*180*60/np.pi)
