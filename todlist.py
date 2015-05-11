import os
from enact import filedb
from enlib import config
parser = config.ArgumentParser(os.environ["HOME"] + "/.enkirc")
parser.add_argument("query", nargs="?", default=None)
args = parser.parse_args()
filedb.init()
print repr(filedb.scans[args.query])
