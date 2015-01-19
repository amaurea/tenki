import numpy as np, argparse
from enlib import enmap
parser = argparse.ArgumentParser()
parser.add_argument("srcs")
parser.add_argument("area")
parser.add_argument("-c", "--columns", type=str, default="3,5")
parser.add_argument("-p", "--pad", type=int, default=3)
args = parser.parse_args()

srcs = np.loadtxt(args.srcs)
area = enmap.read_map(args.area)

cols = [int(w) for w in args.columns.split(",")]

pos  = srcs[:,cols].T*np.pi/180
pix  = area.sky2pix(pos)

for p in pix.T:
	print ("0 "*args.pad + " %9.2f %9.2f") % tuple(p)
