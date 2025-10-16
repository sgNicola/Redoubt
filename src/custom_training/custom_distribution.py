# Code adopted from https://github.com/jchengai/pluto
from typing import Dict, List, Set
from collections import Counter
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

class TypeDistribution():
    def __init__(self):
        pass

    def scenario_counts(self,scenarios: List[AbstractScenario]):
        scenario_types = [scenario.scenario_type for scenario in scenarios]
        scenario_type_counts = Counter(scenario_types)
        return dict(scenario_type_counts)