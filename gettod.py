import numpy as np, os, h5py
from enlib import config
from enact import data, filedb

parser = config.ArgumentParser(os.environ["HOME"]+"/.enkirc")
parser.add_argument("query")
parser.add_argument("ofile")
parser.add_argument("-d", "--dets", type=str, default=None)
parser.add_argument("-D", "--absdets", type=str, default=None)
parser.add_argument("-c", "--calib", action="store_true")
parser.add_argument("--nofft", action="store_true")
args = parser.parse_args()

filedb.init()
id = filedb.scans[args.query].ids[0]
entry = filedb.data[id]
subdets = None
absdets = None
if args.absdets is not None:
	absdets = [int(w) for w in args.absdets.split(",")]
elif args.dets is not None:
	subdets = [int(w) for w in args.dets.split(",")]
else:
	subdets = [0]

d = data.read(entry, fields=["gain","tconst","cut","tod","boresight", "noise_cut","polangle","point_offsets","site"], subdets=subdets, absdets=absdets)
if args.calib: d = data.calibrate(d, nofft=args.nofft)

with h5py.File(args.ofile, "w") as hfile:
	hfile["tod"] = d.tod
	hfile["az"]  = d.boresight[1]
	hfile["el"]  = d.boresight[2]