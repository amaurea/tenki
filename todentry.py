import os
from enact import filedb
from enlib import config
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("query", nargs="?", default=None)
args = parser.parse_args()
filedb.init()
ids = filedb.scans[args.query].ids
if len(ids) == 0:
	print "No matching tods!"
else:
	entry = filedb.data.query(ids[0], multi=True)
	names = sorted(entry.keys())
	for name in names:
		vals = entry[name]
		if not isinstance(vals, list): vals = [vals]
		if len(vals) > 1:
			print name
			for val in vals:
				print "  " + val
		else:
			print name + "  " + vals[0]
	#print filedb.data.query(ids[0], multi=True)
