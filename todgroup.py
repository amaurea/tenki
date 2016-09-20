import numpy as np, os
from enlib import utils, coordinates, config
from enact import files, filedb
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("todinfo")
parser.add_argument("odir")
# A binwidth of 10 typically results in 6 groups with similar population levels,
# though one might get one or two small ones in addition. Splitting this
# way is almost the same as splitting by azimuth.
parser.add_argument("-w", "--binwidth", type=float, default=10)
args = parser.parse_args()

def calc_driftangle(hor, t, site):
	hor = np.atleast_2d(hor).T
	t   = np.atleast_1d(t)
	equ = coordinates.transform("hor", "equ", hor, time=utils.ctime2mjd(t), site=site)
	hor_drift = utils.rewind(coordinates.transform("equ","hor", equ, time=utils.ctime2mjd(t+1), site=site),hor,2*np.pi)
	vec_drift = hor_drift-hor
	# Compute angle between this vector and the az axis
	angle = np.arctan2(vec_drift[1],vec_drift[0]*np.cos(hor[1]))%np.pi
	return angle

filedb.init()
db = filedb.data
utils.mkdir(args.odir)

# Read in lines with format [id, az, el]
# Compute drift angle and assign each id to a bin in angle.
nbin = int(np.ceil(360/args.binwidth))
with open(args.odir + "/bins.txt","w") as f:
	for bi in range(nbin):
		f.write("%8.3f %8.3f\n" % (bi*args.binwidth, (bi+1)*args.binwidth))
bfiles = [None for i in range(nbin)]
site = None
for line in open(args.todinfo,"r"):
	toks = line.split()
	id = toks[0]
	entry = db[id]

	az = utils.rewind(float(toks[1]),0,360)
	el = float(toks[2])
	t  = float(id[:id.index(".")])
	if site is None: site = files.read_site(entry.site[0])

	angle = calc_driftangle([az*np.pi/180,el*np.pi/180],t,site)*180/np.pi
	if not np.isfinite(angle): continue
	# Assign bin
	bi = int(angle/args.binwidth)
	if bfiles[bi] is None:
		bfiles[bi] = open(args.odir + "/ang%03d.txt"%bi,"w")
	bfiles[bi].write("%s %8.2f\n" % (id, angle))
	print "%s %8.2f %8.2f %8.2f %3d" % (id, az, el, angle, bi)
