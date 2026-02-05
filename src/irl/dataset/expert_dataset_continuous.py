import numpy as np
import json

class ExpertTrajectory:
    """
    Data structure for expert trajectories
    """
    def __init__(self, 
                 episodeID: int,
                 segment: str,
                 feature_expectation: list,
                 initial_values: dict,
                 features: list,
                 soc_history: list
                 ):
        self.episodeID = episodeID
        self.segment = segment
        self.feature_expectation = feature_expectation
        self.initial_values = initial_values
        self.features = features
        self.soc_history = soc_history

class ExpertDatasetContinuous:
    """
    Expert Dataset for Continuous V2G Environment
    """
    def __init__(self):
        self.trajectories = []
    
    def load_trajectories_from_json(self, json_path: str):

        with open(json_path, "r") as f:
            data = json.load(f)
        
        for traj in data:
            expert_traj = ExpertTrajectory(
                episodeID=traj['episodeID'],
                segment=traj['segment'],
                feature_expectation=traj['feature_expectation'],
                initial_values=traj['initial_values'],
                features=traj['features'],
                soc_history=traj['soc_history']
            )
            self.trajectories.append(expert_traj)
    
    def split_dataset(self, train_ratio: float = 0.8, segment = None):
        """
        Split the dataset into training and validation sets
        """
        if segment is not None:
            filtered_trajectories = [traj for traj in self.trajectories if segment in traj.segment]
        else:
            filtered_trajectories = self.trajectories

        n_train = int(len(filtered_trajectories) * train_ratio)
        train_set = filtered_trajectories[:n_train]
        val_set = filtered_trajectories[n_train:]
        return train_set, val_set