import numpy as np
import os
import torch
import pandas as pd

os.environ['PLANTF'] = '/home/sgwang/planTF'

def load_scenario_features(folder_path):
    # Load encoder features from a folder
    scenarios = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npz'):
            file_path = os.path.join(folder_path, file_name)
            scenario_name = file_name.split('.')[0]
            try:
                scenario = np.load(file_path)
                scenarios.append({scenario_name:scenario})
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
    return scenarios

def get_array_features(scenario):
    "return array features of one scenario"
    array_features = []
    for array_name in scenario.files:
        array_features.append(scenario[array_name])
    return array_features

def get_ego_features(features):
    ego_features = []
    for feature in features:
        feature = feature[:,0]
        ego_features.append(feature)
    return ego_features

def get_agent_features(features):
    agent_features = []
    for feature in features:
        feature = feature[:,1:]
        agent_features.append(feature)
    return agent_features
   
def main():
    plantf_path = os.getenv('PLANTF')
    scenario_path = os.path.join(plantf_path, 'encoder_features')
    scenarios =load_scenario_features(scenario_path)
    first_scenario = scenarios[0]
    scenario = list(first_scenario.values())[0]
    array_features = get_array_features(scenario)
    ego_features = get_ego_features(array_features)

if __name__ == '__main__':
    main()