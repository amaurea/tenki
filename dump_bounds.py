import numpy as np, argparse, h5py
parser = argparse.ArgumentParser()
parser.add_argument("ids")
parser.add_argument("todinfo")
parser.add_argument("ofile")
args = parser.parse_args()

def find(array, vals):
	"""Return the indices of each value of vals in the given array."""
	order = np.argsort(array)
	return order[np.searchsorted(array, vals, sorter=order)]

# Load the ids
ids = np.array([line.split()[0].split(":")[0] for line in open(args.ids,"r")])

with h5py.File(args.todinfo, "r") as hfile:
	# Find the ids in the file
	tids = hfile["id"].value
	# Find the index for each of our target ids
	inds = find(tids, ids)
	# And get those bounds
	bounds = hfile["bounds"][:,:,inds]

with open(args.ofile, "w") as f:
	for i, id in enumerate(ids):
		bflat = bounds[:,:,i].T.reshape(-1)
		f.write("%s" % id + " %10.4f"*len(bflat) % tuple(bflat) + "\n")
