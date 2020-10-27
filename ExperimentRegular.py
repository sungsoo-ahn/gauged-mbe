from copy import copy
from tqdm import tqdm
import numpy as np

from CompleteFactorGraph import ThreeRegularFactorGraph
from WMBE_Optimization import WMBE_Optimization

exp_num = 1
T = 0.5
max_iter = 150
graph_size = 180
WMBE_ibound = 4

gm = ThreeRegularFactorGraph(nfactors=graph_size, T=T)
WMBE = WMBE_Optimization(gm, WMBE_ibound)
WMBE.update_messages()
WMBE_logZs = []
WMBE_times = []
for iteration in tqdm(range(max_iter)):
    WMBE.update_parameters(weight=False, gauge=True, reparam=False)
    WMBE.compute_Z()
    WMBE_logZs.append(copy(WMBE.logZ))
    WMBE_times.append(copy(WMBE.elapsed_time))

print(WMBE_logZs)