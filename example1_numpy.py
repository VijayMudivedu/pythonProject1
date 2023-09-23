import numpy as np

alist = [1, 2, 3, 4, 5]   # Define a python list. It looks like an np array
narray = np.array([1, 2, 3, 4]) # Define a numpy array

print(alist, narray, type(alist), type(narray))

print(narray + narray)
print(alist + alist)

print(narray * 3)
print(alist * 3)

nmatrx = np.array([narray,narray,narray])
nmatrx = np.array([narray,[1]*4,narray])
print(nmatrx)

okmatrix = np.array([[1,2],[3,4]])
print(okmatrix+okmatrix)
okmatrix.T
narray.T
np.array([[1,2,3,4]]).T

nmatrx
nmatrxT = nmatrx.T

np.linalg.norm(nmatrx)
np.linalg.norm(nmatrxT)

np.linalg.norm(nmatrx,axis=0)
np.linalg.norm(nmatrxT,axis=0)

# dot product
g = np.random.default_rng(seed=12)
aa = g.integers(low=1,high=5,size=(4,3))
aa
p= np.random.randint(low=1,high=5,size=(4,3))

rng = np.random.default_rng(seed=2323)
aa = rng.integers(1,5,(4,3))
aa1 = rng.integers(1,5,(4,3))

np.dot(aa,aa.T)
np.dot(aa[0],aa[1])

aa @ aa.T
np.dot(aa,aa.T)
aa * aa
aa
np.sum(aa,axis=0)
np.sum(aa,axis=1)

aa.sum(axis=1)
aa.sum(axis=0)

# center columns
aa - aa.mean()
aa - aa.mean(axis=0)
aa.T - aa.mean(axis=1)

np.linalg.norm(aa-aa1,axis=1)

from sklearn.metrics import pairwise_distances
pairwise_distances(metric='cosine',X=np.array([[20,40],[30,20]]).reshape(1,-1))
x1 = np.array([20, 40])
x2 = np.array([30, 20])

np.dot(x1,x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))
from scipy.stats import cosine
from sklearn.metrics import pairwise

pairwise.cosine_similarity(X=np.array([x1,x2]),dense_output=True)



from pgmpy.models import BayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt

# Define the Bayesian Network structure
model = BayesianNetwork([
    ('Weather', 'Traffic'),
    ('Weather', 'Mood'),
    ('Traffic', 'MeetingDelay'),
    ('CarBreakdown', 'MeetingDelay'),
    ('CoffeeSpill', 'Mood'),
    ('Mood', 'Productivity'),
    ('Workload', 'Productivity'),
    ('ComputerCrash', 'Workload')
])
model.add_node("Train Delay")
# Visualize the network
pos = nx.spring_layout(model)
nx.draw(model, pos) #, with_labels=True, node_size=3000, node_color="skyblue", node_shape="o", alpha=0.5, linewidths=40)
plt.show()