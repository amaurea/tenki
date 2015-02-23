from enact import filedb
import argparse
parser = argparse.ArumentParser()
parser.add_argument("query", nargs="?", default=None)
args = parser.parse_args()
print repr(filedb.scans[args.query])
