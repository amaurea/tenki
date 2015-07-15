import numpy as np, mpi4py.MPI, os, sys, zipfile
from enlib import config, utils, coordinates, errors
from enact import filedb, data

parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("query",type=str, default="cmb",      nargs="?")
parser.add_argument("objs", type=str, default="Sun,Moon", nargs="?")
args = parser.parse_args()

comm  = mpi4py.MPI.COMM_WORLD
objs  = args.objs.split(",")
nobj  = len(objs)
dstep = 10
sstep = 100

filedb.init()
db       = filedb.data
info     = filedb.scans[args.query]
ids      = info.ids
hprint = False
for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	entry = db[id]

	# Get a few representative samples
	try:
		d = data.read(entry, fields=["gain","cut","point_offsets","boresight","site"])
		d = data.calibrate(d)
	except (zipfile.BadZipfile, errors.DataMissing) as e:
		print "#%s error: %s" % (id,e.message)
		#print "%s %8.3f %7.3f %8.3f %7.3f %s" % (id, np.nan, np.nan, np.nan, np.nan, "nodata")
		continue

	hour = info[ind].fields.hour
	tags = sorted(list(set(info[ind].tags)-set([id])))

	# Get input pointing
	bore = d.boresight[:,::sstep]
	offs = d.point_offset.T[:,::dstep]
	ipoint = np.zeros(bore.shape + offs.shape[1:])
	ipoint[0] = utils.ctime2mjd(bore[0,:,None])
	ipoint[1:]= bore[1:,:,None]+offs[:,None,:]
	ipoint = ipoint.reshape(3,-1)
	iref = np.mean(ipoint[1:],1)

	# Transform to equ
	opoint = coordinates.transform("hor","equ", ipoint[1:], time=ipoint[0], site=d.site)
	oref = np.mean(opoint,1)

	print "%s %4.1f %8.3f %7.3f %8.3f %7.3f" % (id, hour, iref[0]/utils.degree, iref[1]/utils.degree, oref[0]/utils.degree, oref[1]/utils.degree),

	# Compute position of each object, and distance to it
	orect = coordinates.ang2rect(opoint, zenith=False)
	for obj in objs:
		objpos  = coordinates.ephem_pos(obj, ipoint[0,0])
		objrect = coordinates.ang2rect(objpos, zenith=False)
		odist   = np.min(np.arccos(np.sum(orect*objrect[:,None],0)))
		print "| %8.3f %7.3f %7.3f" % (objpos[0]/utils.degree, objpos[1]/utils.degree, odist/utils.degree),

	print "| "+" ".join(tags)
	sys.stdout.flush()
