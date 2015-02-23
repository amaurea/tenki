import numpy as np, argparse, os, sys
from enlib import enmap, pmat, config
from enact import filedb, data
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("id")
parser.add_argument("area")
parser.add_argument("--di", type=int, default=0, help="Index into array of accepted detectors to use.")
args = parser.parse_args()
dtype = np.float64

eqsys = config.get("map_eqsys")

area  = enmap.read_map(args.area).astype(dtype)
area  = enmap.zeros((3,)+area.shape[-2:], area.wcs, dtype)
entry = filedb.data[args.id]

# First get the raw samples
d        = data.read(entry, subdets=[args.di])
raw_tod  = d.tod[0,d.sample_offset:d.cutafter].copy()
raw_bore = d.boresight[:,d.sample_offset:d.cutafter].T
# Then some calibrated samples
d        = data.calibrate(d)
cal_tod  = d.tod[0]
cal_bore = d.boresight.T
# Apply fourier-truncation to raw data
raw_tod = raw_tod[:cal_tod.shape[0]]
raw_bore = raw_bore[:cal_bore.shape[0]]

# And a proper ACTScan
scan = data.ACTScan(entry, subdets=[args.di])
# Detector pointing
det_ipoint = scan.boresight + scan.offsets[0]

# Manual transformation
trf = pmat.pos2pix(scan, None, eqsys)
det_opoint_exact = trf(det_ipoint.T).T

psi_rot_exact = np.arctan2(det_opoint_exact[:,3],det_opoint_exact[:,2])
psi_det = np.arctan2(scan.comps[:,2],scan.comps[:,1])
psi_exact = (psi_rot_exact + psi_det)/2%np.pi

ppix = pmat.PmatMap(scan, area)
# Exact to pixels
det_opix_exact = ppix.transform(det_ipoint.T).T

# Interpolated to pixels
det_opix, det_ophase = ppix.translate(scan.boresight, scan.offsets, scan.comps)

# Get cut status of each sample also
cut = scan.cut.to_mask()[0]

# Ok, output all the information. First a header describing our information
print "# id %s det %d samp_off %d" % (args.id, scan.dets[0], d.sample_offset)
print "# det angle %10.3f" % (psi_det/2*180/np.pi)
print "# FITS header"
for card in area.wcs.to_header().cards:
	print "#  " + str(card)
print "%5s %5s | %s | %10s %10s | %15s %10s %10s | %8s %10s %10s | %10s %10s | %10s %10s | %10s | %9s %9s | %9s %9s | %10s" % ("#samp", "rsamp", "c", "rtod", "ctod", "rsec", "raz","rel", "csec", "caz", "cel", "daz", "del", "dec", "ra", "psi", "y", "x", "y_ip", "x_ip", "psi_ip")
sys.stdout.flush()
deg = 180/np.pi
# build total output array. Will output with numpy savetxt
out = np.array([
	np.arange(scan.nsamp),
	np.arange(scan.nsamp)+d.sample_offset,
	cut[:scan.nsamp],
	raw_tod, cal_tod,
	raw_bore[:,0], raw_bore[:,1]*deg, raw_bore[:,2]*deg,
	cal_bore[:,0]-cal_bore[0,0], cal_bore[:,1]*deg, cal_bore[:,2]*deg,
	det_ipoint[:,1]*deg, det_ipoint[:,2]*deg,
	det_opoint_exact[:,0]*deg, det_opoint_exact[:,1]*deg,
	psi_exact*deg,
	det_opix_exact[:,0], det_opix_exact[:,1],
	det_opix[0,:,0], det_opix[0,:,1],
	np.arctan2(det_ophase[0,:,2], det_ophase[0,:,1])*deg/2%180])
np.savetxt("/dev/stdout", out.T, fmt="%5d %5d | %d | %10.0f %10.2f | %15.3f %10.6f %10.6f | %8.4f %10.6f %10.6f | %10.6f %10.6f | %10.6f %10.6f | %10.6f | %9.3f %9.3f | %9.3f %9.3f | %10.6f")
