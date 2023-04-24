import os
import swmmio as sw
import pyswmm as ps
import pandas as pd
import numpy as np
import pprint

from inp_manager.inp import SwmmModel
from inp_manager.test_inp import TEST_FILE, RPT_TEST_FILE
from data.feature_engineering import feature_engineering

desired_width = 500
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 30)


model = sw.Model(TEST_FILE, include_rpt=True)

# with ps.Simulation(model.inp.path) as sim:
#     for _ in sim:
#         pass

conduits_data, nodes_data, subcatchments_data = feature_engineering(model)

o = SwmmModel(model, conduits_data, nodes_data, subcatchments_data)
# print(o.find_all_traces())
pprint.pprint(o.find_all_traces())
