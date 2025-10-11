import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from omegaconf import DictConfig  # Assuming DictConfig is from `omegaconf`

class ComputePostScore:
    def __init__(self, df: pd.DataFrame, cfg: DictConfig):
        self.df = df
        self.post_score = cfg.post_score  # Config key for score column

    def get_energy_score(self) -> pd.DataFrame:
        ood_score = []
        for i in self.df[self.post_score]:
            # Compute raw ood_score as the log sum of exponentials for each array
            raw_ood_score = [np.log(np.sum(np.exp(j))) for j in i]
            ood_score.append(raw_ood_score)
        self.df['score'] = ood_score
        return self.df

    def get_msp_score(self) -> pd.DataFrame:
        ood_score = []
        for i in self.df[self.post_score]:
            # Compute the maximum softmax probability for each array
            raw_score = [np.max(F.softmax(torch.tensor(x), dim=0).numpy()) for x in i]
            ood_score.append(raw_score)
        self.df['score'] = ood_score
        return self.df
    
    def get_mean_score(self) -> pd.DataFrame:
        ood_score = []
        for i in self.df[self.post_score]:
            # Compute the mean for each array
            raw_ood_score = [np.mean(j) for j in i]
            ood_score.append(raw_ood_score)
        self.df['score'] = ood_score
        return self.df
    
    def get_low_score(self) -> pd.DataFrame:
        ood_score = []
        for i in self.df[self.post_score]:
            # Compute the mean for each array
            raw_ood_score = [np.min(j) for j in i]
            ood_score.append(raw_ood_score)
        self.df['score'] = ood_score
        return self.df
    
    def get_max_score(self) -> pd.DataFrame:
        ood_score = []
        for i in self.df[self.post_score]:
            raw_ood_score = [np.max(j) for j in i]
            ood_score.append(raw_ood_score)
        self.df['score'] = ood_score
        return self.df

    def get_exp_mean_score(self) -> pd.DataFrame:
        ood_score = []
        for i in self.df[self.post_score]:
            # Compute the exponential mean for each array
            raw_ood_score = [np.log(np.mean(np.exp(j))) for j in i]
            ood_score.append(raw_ood_score)
        self.df['score'] = ood_score
        return self.df

    def get_var_score(self) -> pd.DataFrame:
        ood_score = []
        for i in self.df[self.post_score]:
            # Compute the variance for each array
            raw_ood_score = [np.var(j) for j in i]
            ood_score.append(raw_ood_score)
        self.df['var'] = ood_score
        return self.df

    def get_plus_score(self) -> pd.DataFrame:
        ood_score = []
        for i in self.df[self.post_score]:
            # Compute the sum of variance and mean for each array
            raw_ood_score = [(np.var(j) + np.mean(j)) for j in i]
            ood_score.append(raw_ood_score)
        self.df['score'] = ood_score
        return self.df

    def get_min_score(self) -> pd.DataFrame:
        ood_score = []
        for i in self.df[self.post_score]:
            # Compute the difference between max and mean for each array
            raw_ood_score = [(np.max(j) - np.mean(j)) for j in i]
            ood_score.append(raw_ood_score)
        self.df['score'] = ood_score
        return self.df


    def get_entropy_score(self) -> pd.DataFrame:
        ood_score = []
        for i in self.df[self.post_score]:
            # Compute entropy for each array
            raw_ood_score = [entropy(j) for j in i]
            ood_score.append(raw_ood_score)
        self.df['score'] = ood_score
        return self.df

    def calculate_average_ood_score(self) -> pd.DataFrame:
        # Compute mean of scores
        self.df['ood_score_avg'] = self.df['score'].apply(lambda x: np.mean(x))
        return self.df

    def calculate_max_ood_score(self) -> pd.DataFrame:
        # Compute max of scores
        self.df['ood_score_max'] = self.df['score'].apply(lambda x: np.max(x))
        return self.df

    def calculate_min_ood_score(self) -> pd.DataFrame:
        # Compute min of scores
        self.df['ood_score_min'] = self.df['score'].apply(lambda x: np.min(x))
        return self.df

    def calculate_std_ood_score(self) -> pd.DataFrame:
        # Compute standard deviation of scores
        self.df['ood_score_std'] = self.df['score'].apply(lambda x: np.std(x))
        return self.df

    def calculate_var_ood_score(self) -> pd.DataFrame:
        # Compute variance of scores
        self.df['ood_score_var'] = self.df['score'].apply(lambda x: np.var(x))
        return self.df

    def ash_based_ood_detection(self, p=0.8):
        """
        ASH-based OOD detection for a trajectory.

        Args:
            p (float): Percentile threshold for pruning (default: 0.8).

        Returns:
            overall_ood_score (float): The final OOD score for the trajectory.
            scaled_scores (torch.Tensor): The ASH-processed frame-level OOD scores.
        """
        # Step 1: Convert Pandas column to PyTorch Tensor
        scores = torch.tensor(self.df['score'].values, dtype=torch.float32)

        # Step 2: Calculate the percentile threshold
        threshold = torch.quantile(scores, p)

        # Step 3: Pruning - Set scores below the threshold to 0
        pruned_scores = torch.where(scores > threshold, scores, torch.tensor(0.0, dtype=torch.float32))

        # Step 4: Scaling - Calculate scaling factor and scale the remaining scores
        total_score = torch.sum(scores)
        pruned_total_score = torch.sum(pruned_scores)
        scaling_factor = total_score / pruned_total_score if pruned_total_score > 0 else 1.0
        scaled_scores = pruned_scores * scaling_factor

        # Step 5: Aggregate scaled scores to get the overall OOD score
        overall_ood_score = torch.sum(scaled_scores).item()
        self.df['ash_score'] = overall_ood_score
        return self.df
