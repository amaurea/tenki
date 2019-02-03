from __future__ import division, print_function
import numpy as np, argparse
from enlib import enmap
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("ofile")
parser.add_argument("-s", "--slice", type=str, default="")
parser.add_argument("-a", "--apod",  type=int, default=50)
parser.add_argument("-p", "--pregrade",  type=int, default=1)
parser.add_argument("-d", "--downgrade", type=int, default=1)
parser.add_argument("-L", "--lmax-scale", type=float, default=np.inf)
parser.add_argument("-O", "--order", type=int, default=3)
args = parser.parse_args()

# Compute total 2d spectrum for the given input files, which must
# all have compatible geometry

nfile   = len(args.ifiles)
ftot    = None
ps_auto = None
for ifile in args.ifiles:
	print("Reading %s" % ifile)
	m = enmap.read_map(ifile)
	m = eval("m"+args.slice)
	m = m.apod(args.apod)
	m = enmap.map2harm(m)
	if ftot is None:
		ftot = m*0
		ps_auto = enmap.zeros(m.shape[:1]+m.shape, m.wcs, m.dtype)
	ftot += m
	ps_auto += m[:,None]*np.conj(m[None,:])

print("Computing cross spectrum")
# Compute auto spectrum
ps_auto /= nfile**2
ftot /= nfile
# Compute total spectrum
ps_tot = ftot[:,None]*np.conj(ftot[None,:])
if len(args.ifiles) > 1:
	# Subtract to get cross spectrum
	ps_cross = ps_tot - ps_auto
else:
	ps_cross = ps_tot
del ps_tot, ps_auto, ftot
ps_cross = enmap.downgrade(ps_cross, args.pregrade)
print(ps_cross.shape)

print("Normalizing")
l = np.sum(ps_cross.lmap()**2,0)**0.5
l = np.minimum(l, args.lmax_scale)
ps_cross *= ps_cross.area() / ps_cross.npix
ps_cross *= l**2/(2*np.pi)

print("Recentering")
# Center l=0
def recenter(m, shape):
	return np.roll(np.roll(m, -shape[-2]//2, -2), -shape[-1]//2, -1)
ps_cross = recenter(ps_cross, ps_cross.shape)
# Replace WCS, so that we can plot it with proper axes
print("Updating WCS")
ly,lx = enmap.laxes(ps_cross.shape, ps_cross.wcs)
lbox = [[np.min(ly),np.min(lx)],[np.max(ly),np.max(lx)]]
lshape = (len(ly),len(lx))
_, lwcs = enmap.geometry(pos=lbox, shape=lshape, proj="plain")
ps_cross.wcs = lwcs

# Project onto an equal-aspect ratio coordinate system
print("Projecting")
oshape, owcs = enmap.geometry(pos=lbox, shape=(np.max(lshape),np.max(lshape)), proj="plain")
ospec = ps_cross.project(oshape, owcs, order=3)

print("Downgrading")
ospec = enmap.downgrade(ospec, args.downgrade)

# Keep only the real part for now
ospec = ospec.real

print("Writing")
enmap.write_map(args.ofile, ospec)
