from enact import filedb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("query", nargs="?", default=None)
args = parser.parse_args()
filedb.init()
print repr(filedb.scans[args.query])
