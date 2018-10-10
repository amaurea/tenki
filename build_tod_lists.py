import numpy as np, astropy.io.fits, argparse, os
from enlib import utils, config, execdb
parser = config.ArgumentParser()
parser.add_argument("catalog", nargs="?", default=None)
parser.add_argument("odir")
parser.add_argument("-m", "--mode", type=str, default="afreq")
args = parser.parse_args()

def get_subcat(catalog, name, val):
	return catalog[catalog[name] == val]

afreqs = {"pa1":("f150",), "pa2":("f150",), "pa3":("f150","f090"), "pa4":("f150","f220"),
		"pa5":("f150","f090"), "pa6":("f150","f090")}

root   = config.get("root")
levels = ["scode", "pa", "obs_detail"]
def get_tags(taglists):
	res = []
	for name, val in taglists:
		if name == "obs_detail":
			val = val.replace("boss_north","boss")
			if val.startswith("deep") or val.startswith("wide") or val.startswith("day_") or val.startswith("boss"):
				res.append("cmb")
			if val.startswith("boss_"):   res.append("boss")
			if val.startswith("deep56_"): res.append("deep56")
			if name in ["moon","mercury","venus","mars","jupiter","saturn","uranus","neptune"]:
				res.append("planet")
		res.append(val)
	return res

def process(catalog, levels, odir, taglists=[], infofile=None):
	if len(taglists) == 0:
		infofile = open(odir + "/todinfo.txt","w")
		infofile.write("p = %s\n\n" % args.odir)
	if len(levels) > 0:
		name  = levels[0]
		vals  = np.unique(catalog[name])
		for val in vals:
			if args.mode == "afreq" and name == "pa":
				for freq in afreqs[val]:
					process(get_subcat(catalog, name, val), levels[1:], odir, taglists + [(name,val),("freq",":"+freq)], infofile)
			else:
				process(get_subcat(catalog, name, val), levels[1:], odir, taglists + [(name,val)], infofile)
		infofile.write("\n")
	else:
		# We've gone through all the levels, so it's time to output
		tags  = get_tags(taglists)
		oname = "selectedTODs_%s.txt" % "_".join(tags).replace(":","")
		print oname
		odata = np.empty(len(catalog),[("tod_name","S30"),("hour_utc","f"),("alt","f"),("az","f"),("pwv","f"),("sel","i"),("field","S30")])
		for key in odata.dtype.names[:5]:
			odata[key] = catalog[key]
		odata["sel"] = 2
		odata["field"] = "_".join(tags)
		np.savetxt(odir + "/" + oname, odata, fmt="%s %5.2f %7.2f %7.2f %7.2f %d %s")
		# Add a correspnding line to our top level file
		#infofile.write("%-0s %s\n" % (os.path.relpath(ofile, root), " ".join(tags)))
		infofile.write("%-50s %s\n" % ("{p}/" + oname, " ".join(tags)))
	# And close our file if we are done
	if len(taglists) == 0:
		infofile.close()

catalog = astropy.io.fits.open(args.catalog)[1].data
utils.mkdir(args.odir)
process(catalog, levels, args.odir)
