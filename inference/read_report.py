import os
import pandas as pd
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

class ReportProcessor:
    def __init__(self,cfg:DictConfig):
        """
        Initialize the ReportProcessor class.
        :param file_path: Path to the data files
        :param runner_columns: Columns to extract from runner_report
        :param metric_columns: Columns to extract from metric_report
        """
        self.file_path = cfg.runner_report_dir
        # Default columns for runner_report
        self.runner_columns = cfg.runner_columns or ['scenario_type', 'scenario_name', 'log_name', 'risk_score']
        # Default columns for metric_report
        self.metric_columns = cfg.metric_columns or ['scenario_name', 'metric_score']

    def read_runner_reports(self) -> pd.DataFrame:
        """
        Read the runner_report.parquet file and extract specified columns.
        """
        runner_report = os.path.join(self.file_path, 'runner_report.parquet')
        if not os.path.exists(runner_report):
            raise FileNotFoundError(f"File {runner_report} does not exist.")

        # Read the runner_report file and extract the specified columns
        df = pd.read_parquet(runner_report)
        if not all(col in df.columns for col in self.runner_columns):
            raise ValueError(f"One or more specified columns {self.runner_columns} do not exist in runner_report.")
        df = df[self.runner_columns]
        return df

    def read_metric_reports(self) -> pd.DataFrame:
        """
        Read all .parquet files from the metric folder and merge them with runner_report on scenario_name.
        """
        # Read runner_report data
        runner_df = self.read_runner_reports()

        # Path to the metric folder
        metric_file_path = os.path.join(self.file_path, 'metrics')
        if not os.path.exists(metric_file_path):
            raise FileNotFoundError(f"Directory {metric_file_path} does not exist.")

        # Traverse all .parquet files in the metric folder
        for root, _, files in os.walk(metric_file_path):
            for file in files:
                if file.endswith('.parquet'):
                    metric_file = os.path.join(root, file)

                    # Read the current metric_report file and extract columns
                    df_metric = pd.read_parquet(metric_file)
                    if not all(col in df_metric.columns for col in self.metric_columns):
                        raise ValueError(f"One or more specified columns {self.metric_columns} do not exist in {file}.")
                    df_metric = df_metric[self.metric_columns]

                    # Rename the metric_score column based on the file name (removing .parquet)
                    metric_name = file.replace('.parquet', '')
                    df_metric.rename(columns={'metric_score': f'{metric_name}'}, inplace=True)

                    # Merge with runner_df on scenario_name
                    runner_df = pd.merge(runner_df, df_metric, on='scenario_name', how='left')
        return runner_df