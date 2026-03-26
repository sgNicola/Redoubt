import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd


class DataVisualization:
    def __init__(self, figsize=(10, 6), alpha=0.8, grid=True):
        """
        Initialize the DataVisualization class with common parameters for plotting.

        Parameters:
        - figsize (tuple): The size of the figure for the plots.
        - alpha (float): The transparency level for the KDE plots.
        - grid (bool): Whether to display a grid on the plots.
        """
        self.figsize = figsize
        self.alpha = alpha
        self.grid = grid
        self.alert_color = '#e6bb90'
        self.normal_color = '#9dce83'

    def draw_distribution(self, df: pd.DataFrame, score: str) -> None:
        """
        Plot the distribution of scores for InD (In-Distribution) and OOD (Out-of-Distribution) scenarios.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the data.
        - score (str): The column name of the score to plot.
        """
        plt.figure(figsize=self.figsize)
        sns.kdeplot(
            data=df[df['scenario_distribution'] == 'InD'],
            x=score,
            fill=True,
            color=self.normal_color,
            label='InD',
            alpha=self.alpha
        )

        sns.kdeplot(
            data=df[df['scenario_distribution'] == 'OOD'],
            x=score,
            fill=True,
            color=self.alert_color,
            label='OOD',
            alpha=self.alpha
        )

        if self.grid:
            plt.grid(True)
        plt.title('Scenario Distribution')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend(fontsize=14)
        plt.show()

    def draw_risk(self, df: pd.DataFrame, score: str) -> None:
        """
        Plot the distribution of risk scores categorized into low risk and high risk.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the data.
        - score (str): The column name of the score to plot.
        """
        plt.figure(figsize=self.figsize)
        sns.kdeplot(
            data=df[df['risk_label'] == False],
            x=score,
            fill=True,
            color=self.normal_color,
            label='Low Risk',
            alpha=self.alpha
        )

        sns.kdeplot(
            data=df[df['risk_label'] == True],
            x=score,
            fill=self.alert_color,
            color=self.alert_color,
            label='High Risk',
            alpha=self.alpha
        )

        if self.grid:
            plt.grid(True)
        plt.title('Risk Score Distribution')
        plt.xlabel('Performance Score')
        plt.ylabel('Density')
        plt.legend(fontsize=14)
        plt.show()

    def draw_performance(self, df: pd.DataFrame, score: str) -> None:
        """
        Plot the performance score distribution for InD and OOD scenarios.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the data.
        - score (str): The column name of the performance score to plot.
        """
        plt.figure(figsize=self.figsize)
        sns.kdeplot(
            data=df[df['scenario_distribution'] == 'InD'],
            x=score,
            fill=True,
            color=self.normal_color,
            label='InD',
            alpha=self.alpha
        )

        sns.kdeplot(
            data=df[df['scenario_distribution'] == 'OOD'],
            x=score,
            fill=True,
            color=self.alert_color,
            label='OOD',
            alpha=self.alpha
        )

        if self.grid:
            plt.grid(True)
        plt.title('Performance Score vs. Distribution')
        plt.xlabel('Performance')
        plt.ylabel('Density')
        plt.legend(fontsize=14)
        plt.show()
