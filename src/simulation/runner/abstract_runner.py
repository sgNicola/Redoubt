from abc import ABCMeta, abstractmethod

from src.simulation.runner.runner_report import RunnerReport

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner



class AbstractRunner(metaclass=ABCMeta):
    """Interface for a generic runner."""

    @abstractmethod
    def run(self) -> RunnerReport:
        """
        Run through all runners with simulation history.
        :return A list of runner reports.
        """
        pass

    @property
    @abstractmethod
    def scenario(self) -> AbstractScenario:
        """
        :return: Get a list of scenarios.
        """
        pass

    @property
    @abstractmethod
    def planner(self) -> AbstractPlanner:
        """
        :return: Get a planner.
        """
        pass
