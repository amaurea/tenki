print "A"
import numpy as np, argparse
print "B"
from enlib import enmap, pmat
print "C"
from enact import filedb, data
print "D"
parser = argparse.ArgumentParser()
parser.add_argument("id")
parser.add_argument("area")
parser.add_argument("--di", type=int, default=0, help="Index into array of accepted detectors to use.")
args = parser.parse_args()

print "E"
area  = parser.add_argument("area")
entry = filedb.data[args.id]
print "F"

# First get the raw samples
d        = data.read(entry, subdets=[args.di])
print "G"
raw_tod  = d.tod[0,d.sample_offset:d.cutafter]
raw_bore = d.boresight[:,d.sample_offset:d.cutafter].T
# Then some calibrated samples
d        = data.calibrate(d)
print "H"
cal_tod  = d.tod[0]
cal_bore = d.boresight.T
# And a proper ACTScan
scan = data.ACTScan(entry, subdets=[args.di])
print "I"
# Detector pointing
det_ipoint = scan.boresight + scan.offsets[0]

# Build pointing translation to angles and pixels
pang = pmat.PmatMap(scan, None)
print "J"
ppix = pmat.PmatMap(scan, area)
print "K"

# Direct transformation to angles
det_opoint = pang.transform(det_ipoint)
print "L"
print det_opoint

