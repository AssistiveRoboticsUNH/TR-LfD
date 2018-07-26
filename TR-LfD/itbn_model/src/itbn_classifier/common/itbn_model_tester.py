from pgmpy.models import IntervalTemporalBayesianNetwork

import os
import networkx as nx
import pandas as pd

model_file = '../../dbn_arl/output/itbn.nx'

nx_model = nx.read_gpickle(model_file)
model = IntervalTemporalBayesianNetwork(nx_model.edges())
model.add_cpds(*nx_model.cpds)
model.learn_temporal_relationships_from_cpds()

model.draw_to_file("test_output/loaded_itbn.png", include_obs=True)
# os.system('gnome-open test_output/itbn.png')

for cpd in model.cpds:
    print(cpd)
print(sorted(model.nodes()))

obs_robot = 0
obs_human = 1

data = pd.DataFrame([('N', 'Y', 'N', 'N', 'N',
                      obs_robot, obs_robot, obs_robot, obs_human, obs_robot,
                      2, 0, 0, 0, 0)],
                    columns=['abort', 'command', 'prompt', 'response', 'reward',
                             'obs_abort', 'obs_command', 'obs_prompt', 'obs_response', 'obs_reward',
                             'tm_command_prompt', 'tm_command_response', 'tm_prompt_abort',
                             'tm_prompt_response', 'tm_response_reward'])

# data.drop('abort', axis=1, inplace=True)
# data.drop('command', axis=1, inplace=True)
# data.drop('prompt', axis=1, inplace=True)
data.drop('response', axis=1, inplace=True)
# data.drop('reward', axis=1, inplace=True)
predictions = model.predict_probability(data)
print(predictions)
