import os
import argparse
from tqdm import tqdm
from common_utils import *
from GameFormer.data_utils import *
import matplotlib.pyplot as plt
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.training.experiments.caching import cache_data
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.planning.script.builders.folder_builder import (
    build_training_experiment_folder,
)
from omegaconf import DictConfig
from hydra.utils import instantiate
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from scenario_manager.scenario_manager import OccupancyType, ScenarioManager
from scenario_manager.cost_map_manager import CostMapManager

# define data processor
class DataProcessor(object):
    def __init__(self, scenario):
        self._scenario = scenario

        self.past_time_horizon = 2 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon 
        self.future_time_horizon = 8 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon
        self.num_agents = 20

        self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30} # maximum number of points per feature to extract per feature layer.
        self._radius = 60 # [m] query radius scope relative to the current pose.
        self._interpolation_method = 'linear' # Interpolation method to apply when interpolating to maintain fixed size map elements.
        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        
    def get_ego_agent(self):
        self.anchor_ego_state = self.scenario.initial_ego_state
        
        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )

        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)

        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
        ) + [self.scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor
    
    def get_neighbor_agents(self):
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_tensor_list, past_tracked_objects_types = \
              sampled_tracked_objects_to_tensor_list(sampled_past_observations)

        return past_tracked_objects_tensor_list, past_tracked_objects_types

    def get_map(self):        
        ego_state = self.scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)

        coords, traffic_light_data = get_neighbor_vector_set_map(
            self.map_api, self._map_features, ego_coords, self._radius, route_roadblock_ids, traffic_light_data
        )

        vector_map = map_process(ego_state.rear_axle, coords, traffic_light_data, self._map_features, 
                                 self._max_elements, self._max_points, self._interpolation_method)

        return vector_map

    def get_ego_agent_future(self):
        current_absolute_state = self.scenario.initial_ego_state

        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
        )

        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )

        return trajectory_relative_poses
    
    def get_neighbor_agents_future(self, agent_index):
        current_ego_state = self.scenario.initial_ego_state
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(sampled_future_observations)
        agent_futures = agent_future_process(current_ego_state, future_tracked_objects_tensor_list, self.num_agents, agent_index)

        return agent_futures
    
    def _get_agent_features(
        self,
        present_idx,
        query_xy: Point2D,
    ):

        ego_cur_state = self.scenario.initial_ego_state

        # ego features
        past_ego_trajectory = self.scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )
        future_ego_trajectory = self.scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
        )
        ego_state_list = (
            list(past_ego_trajectory) + [ego_cur_state] + list(future_ego_trajectory)
        )
        
        if present_idx < 0:
            present_idx = len(ego_state_list) + present_idx
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
                # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]
        tracked_objects_list = [present_tracked_objects] + future_tracked_objects
        present_agents = present_tracked_objects.get_tracked_objects_of_types(
            self.interested_objects_types
        )
        N, T = min(len(present_agents), self.num_agents), len(tracked_objects_list)

        position = np.zeros((N, T, 2), dtype=np.float64)
        heading = np.zeros((N, T), dtype=np.float64)
        velocity = np.zeros((N, T, 2), dtype=np.float64)
        shape = np.zeros((N, T, 2), dtype=np.float64)
        category = np.zeros((N,), dtype=np.int8)
        valid_mask = np.zeros((N, T), dtype=bool)
        polygon = [None] * N

        if N == 0:
            return (
                {
                    "position": position,
                    "heading": heading,
                    "velocity": velocity,
                    "shape": shape,
                    "category": category,
                    "valid_mask": valid_mask,
                },
                [],
                [],
            )

        agent_ids = np.array([agent.track_token for agent in present_agents])
        agent_cur_pos = np.array([agent.center.array for agent in present_agents])
        distance = np.linalg.norm(agent_cur_pos - query_xy.array[None, :], axis=1)
        agent_ids_sorted = agent_ids[np.argsort(distance)[: self.num_agents]]
        agent_ids_dict = {agent_id: i for i, agent_id in enumerate(agent_ids_sorted)}

        for t, tracked_objects in enumerate(tracked_objects_list):
            for agent in tracked_objects.get_tracked_objects_of_types(
                self.interested_objects_types
            ):
                if agent.track_token not in agent_ids_dict:
                    continue

                idx = agent_ids_dict[agent.track_token]
                position[idx, t] = agent.center.array
                heading[idx, t] = agent.center.heading
                velocity[idx, t] = agent.velocity.array
                shape[idx, t] = np.array([agent.box.width, agent.box.length])
                valid_mask[idx, t] = True

                if t == present_idx:
                    category[idx] = self.interested_objects_types.index(
                        agent.tracked_object_type
                    )
                    polygon[idx] = agent.box.geometry

        agent_features = {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "shape": shape,
            "category": category,
            "valid_mask": valid_mask,
        }

        return agent_features, list(agent_ids_sorted), polygon
    
    def get_cost_map(self):
        static_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects.get_static_objects()
        current_ego_state = self.scenario.initial_ego_state
        query_xy = current_ego_state.center
        present_idx=self.num_past_poses
        agent_features, agent_tokens, agents_polygon = self._get_agent_features(
            query_xy=query_xy,
            present_idx=present_idx
        )
        route_roadblocks_ids=self.scenario.get_route_roadblock_ids()
        cost_map_manager = CostMapManager(
            origin=current_ego_state.rear_axle.array,
            angle=current_ego_state.rear_axle.heading,
                height=600,
                width=600,
                resolution=0.2,
                map_api=self.map_api,
            )
        cost_maps = cost_map_manager.build_cost_maps(
                static_objects=static_tracked_objects ,
                agents=agent_features,
                agents_polygon=agents_polygon,
                route_roadblock_ids=set(route_roadblocks_ids),
            )
        return cost_maps["cost_maps"]
    
    def plot_scenario(self, data):
        # Create map layers
        create_map_raster(data['lanes'], data['crosswalks'], data['route_lanes'])

        # Create agent layers
        create_ego_raster(data['ego_agent_past'][-1])
        create_agents_raster(data['neighbor_agents_past'][:, -1])

        # Draw past and future trajectories
        draw_trajectory(data['ego_agent_past'], data['neighbor_agents_past'])
        draw_trajectory(data['ego_agent_future'], data['neighbor_agents_future'])

        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['log_name']}_{data['token']}_{data['scenario_type']}.npz", **data)


def cache_features(arg_list, debug=False):
    for i, args in enumerate(arg_list):
        scenario = args['scenario']
        save_dir = args['params']
    # Initialize the DataProcessor with the provided scenarios
        data_processor = DataProcessor(scenario)
        map_name = scenario._map_name
        token = scenario.token
        data_processor.scenario = scenario
        data_processor.map_api = scenario.map_api
        scenario_type = scenario.scenario_type    
        log_name = scenario.log_name
        # Get agent past tracks
        ego_agent_past, time_stamps_past = data_processor.get_ego_agent()
        neighbor_agents_past, neighbor_agents_types = data_processor.get_neighbor_agents()
        ego_agent_past, neighbor_agents_past, neighbor_indices = \
        agent_past_process(ego_agent_past, time_stamps_past, neighbor_agents_past, neighbor_agents_types, data_processor.num_agents)

        # Get vector set map
        vector_map = data_processor.get_map()

        # Get agent future tracks
        ego_agent_future = data_processor.get_ego_agent_future()
        neighbor_agents_future = data_processor.get_neighbor_agents_future(neighbor_indices)
        cost_map = data_processor.get_cost_map()
            # Gather data
        data = {
            "map_name": map_name,
            "token": token,
            "ego_agent_past": ego_agent_past,
            "ego_agent_future": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past,
            "neighbor_agents_future": neighbor_agents_future,
            "scenario_type": scenario_type,
            "log_name": log_name,
            "cost_map": cost_map,
        }
        data.update(vector_map)

        # Visualization
        if debug:
            data_processor.plot_scenario(data)

        # Save to disk
        data_processor.save_to_disk(save_dir, data) 
           

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', type=str, help='path to raw data')
    parser.add_argument('--map_path', type=str, help='path to map data')
    parser.add_argument('--save_path', type=str, help='path to save processed data')
    parser.add_argument('--debug', action="store_true", help='if visualize the data output', default=False)
    parser.add_argument('--config_path', type=str, help='path to the scenario filter configuration file', default=4)
    args = parser.parse_args()

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)
 
    # get scenarios
    map_version = "nuplan-maps-v1.0"    
    sensor_root = None
    db_files = None
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version, scenario_mapping=scenario_mapping)
 
    filter_params = get_filter_parameters(config_path=args.config_path)
    scenario_filter = ScenarioFilter(**filter_params)
    print(f"[DEBUG] filter_params: {filter_params}")
    # scenario_filter = ScenarioFilter(*get_filter_parameters(config_path="/home/sgwang/GameFormer-Planner/config/scenario_filter/InD_scenarios_300k.yaml"))
    worker = RayDistributed()
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")
    del builder, scenario_filter, scenario_mapping
    worker_map(worker, cache_features, [{'scenario': s, 'params': args.save_path} for s in scenarios])