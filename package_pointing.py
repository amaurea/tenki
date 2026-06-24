import argparse
parser = argparse.ArgumentParser()
parser.add_argument("params", help="Text file with model parameters, as written by model_pointing.py")
parser.add_argument("sel",    help="Selector for which observations to package model for")
parser.add_argument("odir",   help="Directory to write output sqlite and hdf data. Will be created if necessary")
parser.add_argument("-C", "--context", type=str,   default="lat")
parser.add_argument("-T", "--maxdt",   type=float, default=10, help="Max time offset between observation and model in days")
args = parser.parse_args()
import numpy as np, re, os
from pixell import utils, bunch, sqlite
from sogma import loading, device

# This needs to be updated for my new model format.
# 1. Use the model directly
#  + Small, efficient, easy to read
#  - Harder read logic, incompatible with sotodlib approach
#  In particular, reading code would need to support each type
#  of dependency (currently static, wafer and time). Not too big
#  problem, really. Calibration happens at the wafer level, so
#  reading code can safely evaluate the dependency to return a
#  simple parameter object like it currently does.
#  More importantly, it seems hard to restrict to the same
#  scanning pattern...
# 2. Pre-evaluate the model for each observation.
#  + Same format as before, no need to change code
#
# Going with #2 here

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

def read_params(pfile):
	params = np.loadtxt(pfile, dtype=[("name", "U32"), ("type", "U32"), ("bin", "U100"), ("val", "d")])
	# get the model version too
	with open(pfile, "r") as ifile:
		line = next(ifile)
		m    = re.match(r"# *model *: *(\w+).*", line.strip())
		if not m: raise ValueError("Invalid params header '%s'" % line)
		model = m.group(1)
	return params, model

def parse_tpat(desc):
	ttok, atok, etok, rtok = [tok.split(":") for tok in desc.split(",")]
	if ttok[0] != "t" or atok[0] != "az" or etok[0] != "el" or rtok[0] != "roll" or ttok[2][0] != "+":
		raise ValueError("Invalid tpat format '%s'" % desc)
	t1 = float(ttok[1])
	t2 = t1+float(ttok[2])
	az = float(atok[1])*utils.degree
	el = float(etok[1])*utils.degree
	roll = float(rtok[1])*utils.degree
	return t1,t2,az,el,roll

def id2waf(subid):
	obs, ws, band = subid.split(":")
	fields = obs.split("_")
	return ":".join(fields[2][-2:], ws)

def close(a,b,tol): return np.abs(a-b) <= tol

def sorted_range_cands(ranges, vals):
	"""Given possibly overlapping ranges[n,{from,to}] sorted by
	from, return the index of the first and beyond-last element in ranges
	that could contain each entry in vals[m] as cand_ranges[m,{ifrom,ito}]"""
	# First possible match is after last that ends before us
	order2= np.searchsorted(ranges[:,1])
	i1s0  = np.searchsorted(ranges[order2,1], vals)+1
	bad   = i1s0 >= len(vals)
	i1s   = order2[np.minimum(i1s0, len(vals))]
	i1s[bad] = len(vals)
	# Last possible match is first that starts after us
	i2s   = np.searchsorted(ranges[:,0], ttarg)+1
	cand_ranges = np.array([i1s, i2s]).T
	return canc_ranges

def tpat_lookukp(obsinfo, keys, ttol=1*utils.day, stol=1*utils.degree):
	inds  = np.full(len(obsinfo), -1, int)
	ttargs= obsinfo.ctime + obsinfo.dur/2
	# Will end up looping in python, but let's make the start/end points smart at least.
	order = np.argsort(info.keys[:,0])
	keys  = info.keys[order].copy()
	# Apply time-padding
	keys[:,0] -= ttol
	keys[:,1] += ttol
	# Find sub-range we need to search through
	i1s, i2s = sorted_range_cands(keys[:,2], ttargs)
	# Then loop through all the obsinfo entries and find the best match, if any
	for ei, (entry, ttarg, i1, i2) in enumerate(zip(obsinfo, ttargs, i1s, i2s)):
		ibest, dbest = 0, np.inf
		for i in range(i1, i2):
			t1, t2, az, el, roll = keys[i]
			if t1-ttol <= ttarg and ttarg < t2+ttol and close(az, entry.baz, stol) and close(el, entry.bel, stol) and close(roll, entry.roll, stol):
				dist = np.abs(0.5*(t1+t2)-ttarg)
				if dist < dbest:
					ibest = i
					dbest = dist
		inds[ei] = order[ibest]
	return inds

def parse_params(params):
	res = {}
	for ind, (name, type, bin, val) in enumerate(params):
		if name not in res: res[name] = bunch.Bunch(type=type, keys=[], inds=[], vals=[])
		if   type == "tpat":   key = parse_tpat(bin)
		elif type == "static": key = bin
		elif type == "wafer":  key = bin
		else: raise ValueError("Unrecognized param type '%s'" % str(type))
		res[name].keys.append(key)
		res[name].inds.append(ind)
		res[name].vals.append(val)
	for key, val in res.items():
		val.keys = np.array(val.keys)
		val.inds = np.array(val.inds)
		val.vals = np.array(val.vals)
	return res

def eval_params(parsed, obsinfo, model="arc", ttol=1*utils.day, stol=1*utils.degree):
	# Initialize the table
	dtype = [("model", "U32")]
	for name, info in parsed.items():
		dtype.append((name, info.vals[0].dtype))
	nobs  = len(obsinfo)
	table = np.zeros(nobs, dtype).view(np.recarray)
	good  = np.full(nobs, True, bool)
	table["model"] = model
	# Get wafer id for each entry in obsinfo
	wafers = np.array([id2waf(id) for id in obsinfo.id])
	# Get the fields
	for name, info in parsed.items():
		if   info.type == "tpat": inds = tpat_lookup(obsinfo, info.keys)
		elif info.type == "wafer": inds = utils.find(info.keys, wafers, default=-1)
		elif info.type == "static": inds = np.zeros(nobs, int)
		else: raise ValueError("Unrecognized param type '%s'" % info.type)
		valid = inds >= 0
		table[name][valid] = info.val[inds[valid]]
		good &= valid
	return table, good

# Get our observation info
dev     = device.get_device("cpu")
loader  = loading.Loader(args.context, dev=dev)
obsinfo = loader.query(args.sel)
# Read the raw parametrs
params, model = read_params(args.params)
table, good = eval_params(params, obsinfo, model=model, ttol=args.maxdt)
# Restrict to cases with a valid match
print("Found a matching model for %d/%d subids" % (np.sum(good),len(obsinfo)))
# Output as single hdf file with a dataset indicating the start of the time-range we cover,
# which will be t0 for now
utils.mkdir(args.odir)
bunch.write(args.odir + "/pointing_offsets.h5", bunch.Bunch(t0=table[good]))
np.savetxt(args.odir + "/unmatched.txt", obsinfo.id[~good], fmt="%s")
# Output an sqlite file with a single time-range pointing to this file
utils.rm(args.odir + "/db.sqlite")
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
