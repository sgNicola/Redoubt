import pandas as pd
import os
from omegaconf import DictConfig 
from read_report import ReportProcessor
from compute_post_scores import ComputePostScore
import hydra
from hydra.utils import instantiate
from visualization import DataVisualization
from performance_statistics import *

def run_computation(cfg:DictConfig):
    processor = ReportProcessor(
        cfg
    )
    result_df = processor.read_metric_reports()
    InD_scenarios= load_scenario_types_from_csv('scenario_type_counts.csv')
    labeled_df = label_scenarios(result_df, InD_scenarios) 
    compute_postscore=ComputePostScore(labeled_df, cfg)
    energy_score = compute_postscore.get_energy_score()
    average_energy_score = compute_postscore.calculate_average_ood_score()
    visualizer = DataVisualization(figsize=(12, 8), alpha=0.6, grid=True)
    visualizer.draw_distribution(average_energy_score, score='ood_score_avg')

if __name__ == "__main__":
    CONFIG_PATH = 'config'
    CONFIG_NAME = 'runner_report'
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)
    cfg = hydra.compose(config_name=CONFIG_NAME)
    run_computation(cfg)