import numpy as np, argparse
from enlib import enmap
parser = argparse.ArgumentParser()
parser.add_argument("srcs")
parser.add_argument("area")
args = parser.parse_args()

srcs = np.loadtxt(args.srcs)
area = enmap.read_map(args.area)

pos  = srcs[:,[3,5]].T*np.pi/180
pix  = area.sky2pix(pos)

for p in pix.T:
	print "0 0 0 %9.2f %9.2f" % tuple(p)
