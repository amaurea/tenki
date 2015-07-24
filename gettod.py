import numpy as np, os
from enlib import config
from enact import data, filedb

parser = config.ArgumentParser(os.environ["HOME"]+"/.enkirc")
parser.add_argument("query")
parser.add_argument("-d", "--dets", type=str, default="0")
parser.add_argument("-c", "--calib", action="store_true")
args = parser.parse_args()

filedb.init()
id = filedb.scans[args.query].ids[0]
entry = filedb.data[id]

d = data.read(entry, fields=["gain","tconst","cut","tod","boresight", "noise_cut","polangle","point_offsets","site"], subdets=[int(w) for w in args.dets.split(",")])
if args.calib: d = data.calibrate(d)

np.savetxt("/dev/stdout", d.tod.T, fmt="%18.9e")


