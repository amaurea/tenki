from __future__ import print_function, division
import os
from enact import filedb, todinfo
from enlib import config
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("query", nargs="?", default=None)
parser.add_argument("-f", "--dbfile", type=str, default=None)
args = parser.parse_args()
if args.dbfile:
	filedb.scans = todinfo.read(args.dbfile)
else:
	filedb.init()
print(repr(filedb.scans.select(filedb.scans[args.query])))
