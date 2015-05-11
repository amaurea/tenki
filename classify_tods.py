import numpy as np, mpi4py.MPI, os, sys, zipfile
from enlib import config, utils, coordinates, targets
from enact import filedb, files

config.default("filedb", "filedb.txt", "File describing the location of the TOD and their metadata")
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("query")
parser.add_argument("-r", "--focalplane-radius", type=float, default=1.0)
args = parser.parse_args()

comm  = mpi4py.MPI.COMM_WORLD

filedb.init()
db       = filedb.data
filelist = filedb.scans[args.query].ids

for ind in range(comm.rank, len(filelist), comm.size):
	id    = filelist[ind]
	entry = db[id]

	# Get a few representative samples
	site  = files.read_site(entry.site)
	try:
		bore  = files.read_boresight(entry.tod)[0]
	except zipfile.BadZipfile:
		print "%s %9.3f %9.3f %9.3f %9.3f %s" % (id, np.nan, np.nan, np.nan, np.nan, "badzip")
		continue
	if bore.shape[0] < 3 or bore.shape[1] < 1:
		print "%s %9.3f %9.3f %9.3f %9.3f %s" % (id, np.nan, np.nan, np.nan, np.nan, "nobore")
		continue
	bsub  = bore[:,50::100].copy()
	bsub  = bsub[:,np.any(~np.isnan(bsub),0)]
	bsub[0] = utils.ctime2mjd(bsub[0])
	try:
		bsub[1:3] = coordinates.transform("hor","equ",bsub[1:3]*utils.degree,time=bsub[0],site=site)
		# Compute matching object
		targdb= targets.TargetDB(entry.targets)
		obj   = targdb.match(bsub.T, margin=args.focalplane_radius*np.pi/180)
		name  = obj.name if obj else "misc"
	except AttributeError:
		name = "error"

	# Compute mean pointing in hor and equ
	t   = np.median(bore[0,::10])
	hor = np.median(bore[1:,::10],1)
	try:
		equ = coordinates.transform("hor","equ",hor[None]*utils.degree,time=utils.ctime2mjd(t[None]),site=site)[0]/utils.degree
	except AttributeError:
		equ = np.array([np.nan,np.nan])

	hour = t/3600%24

	print "%s %9.3f %9.3f %9.3f %9.3f %5.2f %s" % (id, hor[0],hor[1],equ[0],equ[1], hour, name)
	sys.stdout.flush()
