import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model", help="Text file with one model fit per line")
parser.add_argument("depth1_dir", help="Directory that contains depth1 info files, which will be used to map from model fits to subids")
parser.add_argument("odir", help="Directory to write output sqlite and hdf data. Will be created if necessary")
parser.add_argument("-P", "--tpad",    type=float, default=100)
parser.add_argument("-C", "--context", type=str,   default="lat")
parser.add_argument(      "--missing", type=str,   default="nearest-same")
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

def get_trange(fname):
	t1s, durs = np.loadtxt(fname, usecols=(1,2)).T
	t2s = t1s+durs
	t1  = np.min(t1s)
	t2  = np.max(t2s)
	return t1, t2

def parse_sid(sid):
	id, wafer, band = sid.split(":")
	toks = id.split("_")
	tube = toks[2][-2:]
	return tube, wafer, band

def filter_wtube(sids, wtube):
	res = []
	for sid in sids:
		tube, wafer, band = parse_sid(sid)
		if tube != wtube[0]: continue
		if len(wtube) > 1 and wafer != wtube[1]: continue
		res.append(sid)
	return res

def lookup_sids(rinfo, trange, wtube=None):
	"""return all subids that overlap with trange"""
	t1, t2 = trange
	# overlap starts from first element whose end is after our start
	i1 = rinfo.order2[np.searchsorted(rinfo.t2s, t1)]
	# and ends with the first element whose start is after or at our end
	i2 = np.searchsorted(rinfo.t1, t2)
	sids = rinfo.subid[i1:i2]
	if wtube is not None:
		sids = filter_wtube(sids, wtube)
	return sids

def read_models(modelfile):
	with open(modelfile, "r") as mfile:
		# The first line must be a comment describing the parameter order
		header = next(mfile)
		params = header.split(":")[1].split()
		nparam = len(params)
		dtype  = [("name","U30"),("t","d"),("t1","d"),("t2","d"),("az","d"),
			("el","d"),("roll","d"),("sep";"U1")]+[(param,"d") for param in params]
	return np.loadtxt(modelfile, dtype=dtype).view(np.recarray)

def ndfind(arr, vals):
	"""Find the indices of vals[n,...] in arr[n,...]"""
	# Map the ... into simple 1d integers
	cat    = np.concatenate([arr,vals],0)
	cat    = cat.reshape(len(cat),-1).T
	labels = utils.label_multi(cat)
	n      = len(arr)
	# Then we can use the simple find
	return utils.find(labels[:n], labels[n:])

# For each sid, we want to find the model that:
# * is for the same wtube
# * has the same (az,el,roll) within some tolerance
# * is the closest in time if fulfilling the above

class ModelLookup:
	def __init__(self, models, pat_tol=5*utils.degree):
		self.models = models
		self.wtubes = np.array([get_wtube(name) for name in models.name])
		# Wrap-independent representation of the scanning pattern.
		# Az is excluded since we don't have the central value available
		self.patid  = utils.floor(np.array([
			np.cos(self.models.el  ), np.sin(self.models.el  ),
			np.cos(self.models.roll), np.sin(self.models.roll),
		])/pat_tol)
		self.pat_tol  = pat_tol
	@property
	def nmodel(self): return len(self.models)
	def lookup(self, obsinfo):
		# Find the wtube for each sid. Keep the parts we care about
		wtubes = np.array([parse_sid(sid) for sid in obsinfo.id])
		wtubes = wtubes[:,:self.wtubes.shape[1]]
		# Pattern id
		patid  = utils.floor(np.array([
			np.cos(obsinfo.el  ), np.sin(obsinfo.el  ),
			np.cos(obsinfo.roll), np.sin(obsinfo.roll),
		])/self.pat_tol)
		# Group us and the models by wtube+patid. We will loop over
		# these together in python
		labels = utils.label_multi(
			list(np.concatenate([self.wtubes, wtubes], 1).T) +
			list(np.concatenate([self.patid,  patid ], 1).T)
		)
		uvals, order, edges = utils.find_equal_groups_fast(labels)
		for gi, uval in enumerate(uvals):
			inds  = order[edges[gi]:edges[gi+1]]
			# Split inds into the model part and our part
			ismod = inds < self.nmodel
			imod  = inds[ismod]
			gmod  = self.models[imod]
			ilook = inds[~ismod]
			olook = obsinfo[ilook]
			# Now gmod is the set of models for this wtube+patid, and
			# glook is the set of observations for the same wtube+patid
			# that we want to look up a model for.
			# Now find the closets entry in time in gmod for each entry in glook
			ttarg = obsinfo.t+obsinfo.dur/2
			ibest = utils.nearest_ind(gmod.t, ttarg)








# Given a list of subids, I want to map them to the best entry
# in models. 

def rm(fname):
	try: os.remove(fname)
	except FileNotFoundError: pass

dev     = device.get_device("cpu")
loader  = loading.Loader(args.context, dev=dev)
obsinfo = loader.query(sel)

table  = []

with open(args.model, "r") as mfile:
	# The first line must be a comment describing the parameter order
	header = next(mfile)
	params = header.split(":")[1].split()
	nparam = len(params)
	dtype  = [("subid","S40"),("version","S6")]+[(param,"d") for param in params]
	# Process each norma line
	for line in mfile:
		if line.startswith("#"): continue
		toks = line.split()
		# Get the name of info file
		name  = toks[0]
		# t1, t2 give the ctime range between the first and last point
		# that went into the fit. That isn't the same as the first and last
		# sample that went into the fit, which is more like what we really want,
		# but it's hopefully good enough. If it isn't, we would need to do some
		# scanning pattern-aware padding.
		t1    = float(toks[2])
		t2    = float(toks[3])
		# Sadly, this file doesn't contain actual subids, but instead depth1 ids.
		# That's because of how the joint dataset stuff in the mapmaker works. However,
		# they do contain ctime/dur pairs which we can use to recover the time-range
		# spanned by the depth1 file. Since we also know the tube and wafer from toks[0],
		# we can do a database lookup for all observations of that tube and wafer that
		# fall into that time-range. This is clunky, but still probably the best I can do
		# with what I have.
		wtube  = get_wtube(name)
		sids   = lookup_sids(rinfo, [t1,t2], wtube)
		pvals  = [float(tok) for tok in toks[-nparam:]]
		rows   = np.zeros(len(sids), dtype=dtype).view(np.recarray)
		rows.subid = sids
		rows.version = "lat_v2"
		for pname, pval in zip(params, pvals):
			rows[pname] = pval * utils.degree
		table.append(rows)

utils.mkdir(args.odir)
# Output as single hdf file with a dataset indicating the start of the time-range we cover,
# which will be t0 for now
table = np.concatenate(table)
bunch.write(args.odir + "/pointing_offsets.h5", bunch.Bunch(t0=table))

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
