import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model", help="Text file with one model fit per line")
parser.add_argument("sel",   help="Selector for which observations to package model for")
parser.add_argument("depth1_dir", help="Directory that contains depth1 info files, which will be used to map from model fits to subids")
parser.add_argument("odir", help="Directory to write output sqlite and hdf data. Will be created if necessary")
parser.add_argument("-P", "--tpad",    type=float, default=100)
parser.add_argument("-C", "--context", type=str,   default="lat")
parser.add_argument("-T", "--maxdt",   type=float, default=10, help="Max time offset between observation and model in days")
args = parser.parse_args()
import numpy as np, re, os
from pixell import utils, bunch, sqlite
from sogma import loading, device

def get_wtube(name):
	if m := re.search(r"[\b_]([cio]\d)_(ws\d)[\b_]", name):
		return m.group(1), m.group(2)
	elif m := re.search(r"[\b_]([cio]\d)[\b_]", name):
		return m.group(1),
	raise ValueError("Error parsing wtube from '%s'" % str(name))

def parse_sid(sid):
	id, wafer, band = sid.split(":")
	toks = id.split("_")
	tube = toks[2][-2:]
	return tube, wafer, band

def read_models(modelfile):
	with open(modelfile, "r") as mfile:
		# The first line must be a comment describing the parameter order
		header = next(mfile)
		params = header.split(":")[1].split()
		nparam = len(params)
		dtype  = [("name","U30"),("t","d"),("t1","d"),("t2","d"),("az","d"),
			("el","d"),("roll","d"),("n","i"),("chisq","d"),("sep","U1")]+[(param,"d") for param in params]
	return np.loadtxt(modelfile, dtype=dtype).view(np.recarray)

# For each sid, we want to find the model that:
# * is for the same wtube
# * has the same (az,el,roll) within some tolerance
# * is the closest in time if fulfilling the above

class ModelLookup:
	def __init__(self, models, pat_tol=10*utils.degree):
		self.models = models
		self.wtubes = np.array([get_wtube(name) for name in models.name])
		# Wrap-independent representation of the scanning pattern.
		# Az is excluded since we don't have the central value available
		el   = models.el  *utils.degree
		roll = models.roll*utils.degree
		self.patid  = utils.nint(np.array([el, roll])/pat_tol)
		self.pat_tol  = pat_tol
	@property
	def nmodel(self): return len(self.models)
	def lookup(self, obsinfo):
		omodel = np.zeros(len(obsinfo), self.models.dtype)
		toff   = np.full (len(obsinfo), np.inf)
		# Find the wtube for each sid. Keep the parts we care about
		wtubes = np.array([parse_sid(sid) for sid in obsinfo.id])
		wtubes = wtubes[:,:self.wtubes.shape[1]]
		# Pattern id
		patid  = utils.nint(np.array([obsinfo.bel, obsinfo.roll])/self.pat_tol)
		# Group us and the models by wtube+patid. We will loop over
		# these together in python
		labels = utils.label_multi(
			list(np.concatenate([self.wtubes, wtubes], 0).T) +
			list(np.concatenate([self.patid,  patid ], 1))
		)
		uvals, order, edges = utils.find_equal_groups_fast(labels)
		for gi, uval in enumerate(uvals):
			inds  = order[edges[gi]:edges[gi+1]]
			# Split inds into the model part and our part
			ismod = inds < self.nmodel
			imod  = inds[ismod]
			gmod  = self.models[imod]
			ilook = inds[~ismod] - self.nmodel
			olook = obsinfo[ilook]
			if len(imod) == 0 or len(ilook) == 0: continue
			# Now gmod is the set of models for this wtube+patid, and
			# glook is the set of observations for the same wtube+patid
			# that we want to look up a model for.
			# Now find the closets entry in time in gmod for each entry in glook
			ttarg = olook.ctime+olook.dur/2
			ibest = utils.nearest_ind(gmod.t, ttarg)
			# Find time-offset
			toff  [ilook] = np.abs(gmod.t[ibest]-ttarg)
			omodel[ilook] = gmod[ibest]
		return omodel, toff

# Given a list of subids, I want to map them to the best entry
# in models. 

def rm(fname):
	try: os.remove(fname)
	except FileNotFoundError: pass

# Get our observation info
dev     = device.get_device("cpu")
loader  = loading.Loader(args.context, dev=dev)
obsinfo = loader.query(args.sel)
# Load all the models
imodels = read_models(args.model)
# Match models to observations
lookup  = ModelLookup(imodels)
omodels, toff = lookup.lookup(obsinfo)
# These are the observations where a usable model was found
good    = toff < args.maxdt*utils.day
inds    = np.where(good)[0]
print("Found a matching model for %d/%d subids" % (len(inds),len(obsinfo)))

# Now create the output database
odtype = [("subid","S40"),("version","S6")]+omodels.dtype.descr[10:]
table  = np.zeros(len(inds), dtype=odtype).view(np.recarray)
table.subid   = np.strings.encode(obsinfo.id[inds])
table.version = "lat_v2"
for field in table.dtype.names[2:]:
	table[field] = omodels[field][inds]

utils.mkdir(args.odir)
# Output as single hdf file with a dataset indicating the start of the time-range we cover,
# which will be t0 for now
bunch.write(args.odir + "/pointing_offsets.h5", bunch.Bunch(t0=table))
np.savetxt(args.odir + "/unmatched.txt", obsinfo.id[~good], fmt="%s")

# Output an sqlite file with a single time-range pointing to this file
rm(args.odir + "/db.sqlite")
with sqlite.open(args.odir + "/db.sqlite") as s:
	s.execute("create table files (id integer, name text)")
	s.execute("insert into files (id, name) values (?,?)", (0, "pointing_offsets.h5"))
	print(s)
	s.execute("create table map (id integer, file_id integer, [obs:timestamp__lo] float, [obs:timestamp__hi] float, dataset text)")
	s.execute("insert into map (id, file_id, [obs:timestamp__lo], [obs:timestamp__hi], dataset) values (?,?,?,?,?)", (0, 0, 0, 9999999999, "t0"))
	print(s)
	# Not sure what these do, but they're there in all the sotodlib sqlite files
	s.execute("create table input_scheme (id integer, name text, purpose text, match text, dtype text)")
	s.execute("insert into input_scheme (id, name, purpose, match, dtype) values (?,?,?,?,?)", (1, "obs:timestamp", "in", "range", "numeric"))
	s.execute("insert into input_scheme (id, name, purpose, match, dtype) values (?,?,?,?,?)", (2, "dataset", "out", "exact", "numeric"))
	s.execute("commit")
	print(s)
