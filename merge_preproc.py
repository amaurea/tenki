import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+", help="Input sqlite files")
parser.add_argument("ofile", help="Output sqlite file")
args = parser.parse_args()
from pixell import sqlite, utils

# Prepare to write a new file db from scratch
utils.mkdir(os.path.dirname(args.ofile))
utils.rm(args.ofile)

npfile = 0
with sqlite.open(args.ofile) as ofile:
	for fi, ifname in enumerate(args.ifiles):
		idir = os.path.dirname(ifname)
		with sqlite.open(ifname) as ifile:
			with ofile.attach(ifile):
				# Copy over table definitions from first
				if fi == 0:
					for name, sql in ofile.execute("select name, sql from other.sqlite_master where type = 'table' and name != 'sqlite_sequence'"):
						ofile.execute(sql)
					# Copy over the input scheme one, which should not append
					ofile.execute("insert into input_scheme select * from other.input_scheme")
				# Done with special treatment of first input.
				# Now append data from all the other tables
				# First the map. Here we just need to add to the file_id.
				# I thought I could avoid mentioning the columns by name here,
				# but that's not compatible with singling out an individual column
				cols = ['[%s]' % col for col in ifile.columns("map") if col != "id"]
				print("insert into map (%s) select %s from other.map" % (
					", ".join(cols),
					", ".join([("%s+%d"%(col,npfile) if col == "[file_id]" else col) for col in cols])))
				ofile.execute("insert into map (%s) select %s from other.map" % (
					", ".join(cols),
					", ".join([("%s+%d"%(col,npfile) if col == "[file_id]" else col) for col in cols])))
				# Then the 'files' table, which needs to be translated to
				# absolute paths
				for id, pfile in ofile.execute("select id, name from other.files"):
					ofile.execute("insert into files (name) values ('%s')" % os.path.join(idir, pfile))
					npfile += 1
				ofile.execute("commit")
