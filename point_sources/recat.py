import argparse
parser = argparse.ArgumentParser()
parser.add_argument("icat")
parser.add_argument("ocat")
args = parser.parse_args()
from enlib import dory
cat = dory.read_catalog(args.icat)
dory.write_catalog(args.ocat, cat)
