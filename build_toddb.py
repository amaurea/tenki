import numpy as np, os, h5py
from enlib import config, tagdb, mpi, coordinates, utils
from enact import filedb, actdata
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("tagfile")
parser.add_argument("sel", nargs="?", default="")
parser.add_argument("ofile")
args = parser.parse_args()

file_db = filedb.ACTFiles()
tag_db  = tagdb.read(args.tagfile)
comm    = mpi.COMM_WORLD
Nt      = 2
Naz     = 5

data = {
	"t":[], "az":[], "el":[], "ra":[], "dec":[], "nsamp":[], "ndet":[],
	"baz":[], "bel":[], "waz":[], "wel":[], "dur":[], "bounds":[], "cut":[],
	}

ids = tag_db.query(args.sel)
for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	entry = file_db[id]
	# We mostly care about pointing when selecting
	d = actdata.read(entry, ["boresight","point_offsets","cut","polangle","tconst","gain","site"])
	d = actdata.calibrate(d)
	# Get the array center and radius
	acenter = np.mean(d.point_offset,0)
	arad    = np.mean((d.point_offset-acenter)**2,0)**0.5

	data["ndet"].append(d.ndet)
	data["nsamp"].append(d.nsamp)
	data["cut"].append(d.cut.sum()/float(d.ndet*d.nsamp))

	t, baz, bel = np.mean(d.boresight,1)
	az  = baz + acenter[0]
	el  = bel + acenter[1]
	dur, waz, wel = np.max(d.boresight,1)-np.min(d.boresight,1)

	data["t"].append(t)
	data["dur"].append(dur)
	data["az"].append(az/utils.degree)
	data["el"].append(el/utils.degree)
	data["baz"].append(baz/utils.degree)
	data["bel"].append(bel/utils.degree)
	data["waz"].append(waz/utils.degree)
	data["wel"].append(wel/utils.degree)

	ra, dec = coordinates.transform("hor","cel",[az,el],[utils.ctime2mjd(t)])
	data["ra"].append(ra/utils.degree)
	data["dec"].append(dec/utils.degree)

	# Get the array center bounds on the sky, assuming constant elevation
	ts  = utils.ctime2mjd(t+dur/2*np.linspace(-1,1,Nt))
	azs = az + waz/2*np.linspace(-1,1,Naz)
	E1 = coordinates.transform("hor","cel",[azs,         [el]*Naz],time=[ts[0]]*Naz, site=d.site)
	E2 = coordinates.transform("hor","cel",[[azs[-1]]*Nt,[el]*Nt], time=ts,          site=d.site)
	E3 = coordinates.transform("hor","cel",[azs[::-1],   [el]*Naz],time=[ts[-1]]*Naz,site=d.site)
	E4 = coordinates.transform("hor","cel",[[azs[0]]*Nt, [el]*Nt], time=ts[::-1],    site=d.site)
	bounds = np.concatenate([E1,E2,E3,E4],1)
	bounds[0] = utils.rewind(bounds[0])
	# Grow bounds by array radius
	bmid = np.mean(bounds,1)
	for i in range(2):
		bounds[i,bounds[i]<bmid[i]] -= arad[i]
		bounds[i,bounds[i]>bmid[i]] += arad[i]
	data["bounds"].append(bounds)
