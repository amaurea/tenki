# Warning: This function doesn't work correctly with databases that
# use detector-independent selectors in .map, such as relcal which only
# uses time-ranges. In that case, the lookup code won't know which of
# the resulting overlapping time-ranges to use. Fixing this would require
# changing the map logic and the sofast read-in code

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
parser.add_argument("odir")
args = parser.parse_args()
import yaml, os
from pixell import utils
from sogma import gutils

def read_yaml(fname):
	with open(fname, "r") as ifile:
		return yaml.safe_load(ifile)

def write_yaml(fname, data):
	with open(fname, "w") as ofile:
		return yaml.dump(data, ofile)

def _expand_context_helper(obj, tags):
	if isinstance(obj, dict):
		return {key:_expand_context_helper(obj[key], tags) for key in obj}
	elif isinstance(obj, list):
		return [_expand_context_helper(val, tags) for val in obj]
	elif isinstance(obj, str):
		return obj.format(**tags)
	else:
		return obj

def read_context(fname):
	context = read_yaml(fname)
	context = _expand_context_helper(context, context["tags"])
	return context

def write_context(fname, context): write_yaml(fname, context)

def merge_contexts(contexts, odir, verbose=False):
	"""Expand all paths in contexts and merge all the metadata databases into
	odir, returning a new context pointing to these."""
	utils.mkdir(odir)
	# start by copying over the first context
	ocontext = contexts[0].copy()
	# The metadata group is where the main action happens
	imetas = {}
	for context in contexts:
		for field in context["metadata"]:
			name = field["label"] if "label" in field else field["name"]
			if name not in imetas: imetas[name] = []
			imetas[name].append(field["db"])
	ocontext["metadata"] = []
	for name in imetas:
		dbs = imetas[name]
		if len(dbs) < len(contexts):
			print("Warning: %s not present in all contexts. Skipping" % name)
			continue
		ofname = os.path.abspath(os.path.join(odir, "%s.db" % name))
		gutils.merge_metadbs(dbs, ofname, verbose=verbose)
		ocontext["metadata"].append({"db":ofname, "label":name})
	# Tags is expected to be present, but not used here
	ocontext["tags"] = {}
	obsdb     = os.path.abspath(os.path.join(odir, "obsdb.db"))
	obsfiledb = os.path.abspath(os.path.join(odir, "obsfiledb.db"))
	gutils.merge_obsdbs    ([context["obsdb"]     for context in contexts], obsdb, verbose=verbose)
	gutils.merge_obsfiledbs([context["obsfiledb"] for context in contexts], obsfiledb, verbose=verbose)
	ocontext["obsdb"]     = obsdb
	ocontext["obsfiledb"] = obsfiledb
	return ocontext

contexts = [read_context(fname) for fname in args.ifiles]
ocontext = merge_contexts(contexts, args.odir, verbose=True)
write_context(os.path.join(args.odir, "context.yaml"), ocontext)
