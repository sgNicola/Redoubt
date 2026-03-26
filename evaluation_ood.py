import pandas as pd
import os
from omegaconf import DictConfig 
from utils.read_report import ReportProcessor
from utils.compute_post_scores import ComputePostScore
import hydra
from hydra.utils import instantiate
from utils.performance_statistics import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
from utils.visualization import DataVisualization
import argparse

os.environ["NUPLAN_EXP_ROOT"] = "/home/sgwang/nuplan/exp"
CONFIG_PATH = 'utils/config'
CONFIG_NAME = 'runner_report'
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path=CONFIG_PATH)

def min_max_normalize(scores):
    return (scores - scores.min()) / (scores.max() - scores.min())

def calculate_auroc(y_true, y_scores):
    """
    calculate AUROC (Area Under Receiver Operating Characteristic).
    parameters:
        y_true (list or ndarray): true， 0 or 1
        y_scores (list or ndarray): model score or possibility

    return:
        float: AUROC，range is [0.0, 1.0]。
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    
    try:
        auroc = roc_auc_score(y_true, y_scores)
        return auroc
    except ValueError as e:
        raise ValueError(f" AUROC error: {e}")
    
def calculate_fpr(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    fpr95 = fpr[optimal_idx ]
    print(f"FPR95: {fpr95}")
    return fpr95,optimal_threshold

def calculate_fpr95_best(y_true, y_score, min_tpr=0.95):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    valid_idxs = np.where(tpr >= min_tpr)[0]

    if len(valid_idxs) == 0:
        return 1.0, None  
 
    best_idx = valid_idxs[np.argmin(fpr[valid_idxs])]

    return fpr[best_idx], thresholds[best_idx]


def load_and_prepare_data(planner, benchmark, density_file):
    scenarios = ['scenario_group_0', 'scenario_group_1', 'scenario_group_2', 'scenario_group_3', 'scenario_group_4']
    all_dfs = []
    for scenario_filter in scenarios:
        cfg = hydra.compose(
            config_name=CONFIG_NAME,
            overrides=[
                f'split={scenario_filter}',
                f'planner=inference_{planner}_planner',
                f'job_name={benchmark}'
            ]
        )
        processor = ReportProcessor(cfg)
        result_df = processor.read_metric_reports()
        all_dfs.append(result_df)

    open_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    InD_scenarios = load_scenario_types('InD.yaml')
    labeled_df = label_scenarios(open_df, InD_scenarios)
    density = pd.read_parquet(density_file)
    density_df = pd.merge(density, labeled_df)
    return density_df, cfg

def preprocess_df(density_df, cfg):
    compute_postscore = ComputePostScore(density_df, cfg)
    energy_score = compute_postscore.get_energy_score()
    df = density_df.copy()
    df['label'] = df['scenario_distribution'].apply(lambda x: 1 if x == 'InD' else 0)
    df['dac_label']= df['drivable_area_compliance'].apply(lambda x: 0 if x == 1 else 1)
    df['nefc_label']=df['no_ego_at_fault_collisions'].apply(lambda x: 0 if x == 1 else 1)
    return df

def compute_density_energy(df, ts_range):
    df = df.copy()
    df['density'] = df['log_p'].apply(
        lambda scores: np.mean([scores[i] for i in ts_range]) if isinstance(scores, (list, np.ndarray)) and max(ts_range) < len(scores) else np.nan
    )
    df['energy'] = df['score'].apply(
        lambda scores: np.mean([scores[i] for i in ts_range]) if isinstance(scores, (list, np.ndarray)) and max(ts_range) < len(scores) else np.nan
    )
    
    df = df.dropna(subset=['density', 'energy'])
    df = df[~np.isinf(df['energy'])]
    return df

def compute_auroc(df, alpha):
    df = df.copy()
    normalized_density = min_max_normalize(df['density'])
    normalized_energy = min_max_normalize(df['energy'])
    df['ood_score'] =   normalized_density* alpha +normalized_energy* (1-alpha) 
    df['sigmoid_ood'] = 1 / (1 + np.exp(-df['ood_score']))

    y_true = df['label']
    y_pred = df['sigmoid_ood']
    if y_true.nunique() < 2:
        return 0
    return roc_auc_score(y_true, y_pred)

def compute_fpr(df, alpha):
    df = df.copy()
    normalized_density = min_max_normalize(df['density'])
    normalized_energy = min_max_normalize(df['energy'])
    df['ood_score'] =   normalized_density* alpha +normalized_energy* (1-alpha) 
    df['sigmoid_ood'] = 1 / (1 + np.exp(-df['ood_score']))

    y_true = df['label']
    y_pred = df['sigmoid_ood']
    if y_true.nunique() < 2:
        return 0
    fpr95, threshold = calculate_fpr95_best(y_true, y_pred)
    return fpr95, threshold
    
def search_best_auc_params(df):
    best_auroc = 0
    best_t = 0
    best_alpha = 0
    max_time = 149
    window_size =10
    for t_start in range(0, max_time - window_size + 1):
        ts_range = range(t_start, t_start + window_size)
        ts_df = compute_density_energy(df, ts_range)

        for alpha in np.linspace(0, 1, 10):  
            auc = compute_auroc(ts_df, alpha)
            if auc > best_auroc:
                best_auroc = auc
                best_t = t_start
                best_alpha = alpha

    print(f"\n Best AUROC: {best_auroc:.4f} at alpha={best_alpha:.2f}")
    return best_t, best_alpha, best_auroc


def search_best_fpr_params(df):
    best_fpr = float('inf')
    best_t = None
    best_alpha = None
    best_threshold = None
    max_time = 149
    window_size =10
    for t_start in range(0, max_time - window_size + 1):
        ts_range = range(t_start, t_start + window_size)
        ts_df = compute_density_energy(df, ts_range)

        for alpha in np.linspace(0, 1, 10):  # alpha from 0.0 to 1.0

            fpr95, threshold = compute_fpr(ts_df, alpha)

            if fpr95 < best_fpr:
                best_fpr = fpr95
                best_t = t_start
                best_alpha = alpha
                best_threshold = threshold

    print(f"\n Best FPR: {best_fpr:.4f} at ts={best_t}~{best_t+window_size}, alpha={best_alpha:.2f}, threshold={best_threshold:.4f}")
    return best_t, best_alpha, best_fpr, best_threshold


def evaluate_auroc(df,t_start,alpha):
    window_size =10
    ts_range = range(t_start, t_start + window_size)
    ts_df = compute_density_energy(df, ts_range)
    auc = compute_auroc(ts_df, alpha)
    print(f"t={t_start}~{t_start+window_size}, alpha={alpha:.2f}, AUROC={auc:.4f}")
    return auc

def evaluate_fpr(df, t_start, alpha):
    window_size =10
    ts_range = range(t_start, t_start + window_size)
    ts_df = compute_density_energy(df, ts_range)
    fpr95, threshold = compute_fpr(ts_df, alpha)
    print(f"t={t_start}-{t_start+window_size}, alpha={alpha:.2f}, FPR={fpr95:.4f}, threshold={threshold:.2f}")
    
           
if __name__ == "__main__":
    # ================== Argument Parser ==================
    parser=argparse.ArgumentParser(description="Evaluate OOD detection performance.")
    parser.add_argument("--planner", type=str, default="scope", help="Planner name")
    parser.add_argument("--benchmark", type=str, default="closed_loop_nonreactive_agents", help="Benchmark name")
    parser.add_argument("--density-file", type=str, default="planscope_dataset/prediction/cln_nopro_density.parquet", help="Path to density file")
    args = parser.parse_args()
    planner = args.planner
    benchmark = args.benchmark
    density_file = args.density_file
    density_df, cfg = load_and_prepare_data(planner, benchmark, density_file)    
    ignore_scenarios = ['unknown']
    density_df =  density_df[~ density_df['scenario_type'].isin(ignore_scenarios)]
    df = preprocess_df(density_df, cfg)
    best_t, best_alpha, best_auc = search_best_auc_params(df)
    fpr_t, fpr_alpha, best_fpr, best_threshold = search_best_fpr_params(df)