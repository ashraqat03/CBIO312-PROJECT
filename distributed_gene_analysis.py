from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    digits = load_digits()
    data = digits.data
    target = digits.target
else:
    data = None
    target = None

data = comm.bcast(data, root=0)
target = comm.bcast(target, root=0)

chunk_size = len(data) // size
start = rank * chunk_size
end = start + chunk_size if rank != size - 1 else len(data)
local_data = data[start:end]
local_target = target[start:end]

clf = RandomForestClassifier(n_estimators=100)
clf.fit(local_data, local_target)

local_score = clf.score(local_data, local_target)
scores = comm.gather(local_score, root=0)

if rank == 0:
    print("Scores from all nodes:", scores)
    print("Average score:", np.mean(scores))
