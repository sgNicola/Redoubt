import pandas as pd
import yaml
def load_scenario_types(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return set(data['scenario_types'])

def load_scenario_types_from_csv(file_path:str):
    scenario_types = pd.read_csv(file_path)['scenario_type'].values
    return set(scenario_types)

def label_scenarios(df: pd.DataFrame, scenario_types) -> pd.DataFrame:
    df['scenario_distribution'] = df['scenario_type'].apply(lambda x: 'InD' if x in scenario_types else 'OOD')
    return df

def calculate_average_metric_score(df: pd.DataFrame) -> pd.DataFrame:
    df['metric_score_avg'] = df[['planner_expert_average_heading_error_within_bound', 
    'planner_expert_average_l2_error_within_bound',
     'planner_expert_final_heading_error_within_bound', 
     'planner_expert_final_l2_error_within_bound',
     'planner_miss_rate_within_bound'
     ]].mean(axis=1)
    return df

def label_low_score(df):
    df['risk_label'] = df['metric_score_avg'] < df['metric_score_avg'].mean()
    return df

def count_high_label_scenario_type(df: pd.DataFrame) -> None:
    """
    Count the number of occurrences of each scenario_type for rows labeled as 'low risk'
    and print the top 20 scenario_types.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing 'risk_label' and 'scenario_type' columns.
    """
    if 'risk_label' not in df.columns or 'scenario_type' not in df.columns:
        raise ValueError("The DataFrame must contain 'risk_label' and 'scenario_type' columns.")
    
    # Filter rows where risk_label is True
    low_risk_df = df[df['risk_label'] == True]

    # Count occurrences of each scenario_type
    scenario_counts = low_risk_df['scenario_type'].value_counts()

    # Print the top 20 scenario_types with counts
    print("Top 20 scenario types with low risk counts:")
    print(scenario_counts.head(20))
    return scenario_counts

def add_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'risk_label' column to the DataFrame. If any boolean metric is less than 1, 
    set 'risk_label' to True; otherwise, set it to False.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - boolean_metrics (list): List of boolean metric column names.

    Returns:
    - pd.DataFrame: The input DataFrame with an additional 'risk_label' column.
    """
    boolean_metrics =[
    'no_ego_at_fault_collisions',  
    'drivable_area_compliance',
     'driving_direction_compliance',  
    'time_to_collision_within_bound',
    'ego_progress_along_expert_route',
    'ego_is_making_progress',
    'ego_is_comfortable',
    'speed_limit_compliance']

    # Ensure all metrics exist in the DataFrame
    missing_metrics = [metric for metric in boolean_metrics if metric not in df.columns]
    if missing_metrics:
        raise ValueError(f"The following metrics are missing from the DataFrame: {missing_metrics}")
    
    # Create the 'risk_label' column: True if any boolean metric < 1, False otherwise
    df['risk_label'] = (df[boolean_metrics] < 0.5).any(axis=1)

    return df

def count_high_risk_ind_types(df: pd.DataFrame) -> None:
    """
    Count the occurrences of 'scenario_type' for rows where:
    - 'risk_label' is True
    - 'scenario_distribution' is 'InD'

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing 'risk_label' and 'scenario_distribution' columns.
    """
    if 'risk_label' not in df.columns or 'scenario_distribution' not in df.columns:
        raise ValueError("The DataFrame must contain 'risk_label' and 'scenario_distribution' columns.")
    
    # Filter rows where risk_label is True and scenario_distribution is 'InD'
    high_risk_ind_df = df[(df['risk_label'] == True) & (df['scenario_distribution'] == 'InD')]

    # Count occurrences of each scenario_type
    scenario_counts = high_risk_ind_df['scenario_type'].value_counts()

    # Print the counts
    print("Scenario type counts for low-risk InD scenarios:")
    print(scenario_counts)
    return scenario_counts

def count_low_risk_ood_types(df: pd.DataFrame) -> None:
    """
    Count the occurrences of 'scenario_type' for rows where:
    - 'risk_label' is True
    - 'scenario_distribution' is 'InD'

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing 'risk_label' and 'scenario_distribution' columns.
    """
    if 'risk_label' not in df.columns or 'scenario_distribution' not in df.columns:
        raise ValueError("The DataFrame must contain 'risk_label' and 'scenario_distribution' columns.")
    
    # Filter rows where risk_label is True and scenario_distribution is 'InD'
    low_risk_ood_df = df[(df['risk_label'] == False) & (df['scenario_distribution'] == 'OOD')]

    # Count occurrences of each scenario_type
    scenario_counts = low_risk_ood_df['scenario_type'].value_counts()

    # Print the counts
    print("Scenario type counts for low-risk InD scenarios:")
    print(scenario_counts)
    return scenario_counts
     
def filter_df_by_ood_score(df):
    return df[(df['ood_score_avg'] >= 2.5) & (df['ood_score_avg'] <= 3.5)]


def relabel_scenarios(df: pd.DataFrame, label) -> pd.DataFrame:
    scenarios = [
        'stopping_at_stop_sign_without_lead', 'starting_unprotected_noncross_turn',
        'starting_protected_cross_turn', 'on_carpark', 'on_pickup_dropoff', 
        'on_intersection', 'on_stopline_traffic_light', 'stopping_at_crosswalk', 
        'high_lateral_acceleration', 'traversing_pickup_dropoff', 
        'starting_protected_noncross_turn', 'on_traffic_light_intersection', 
        'following_lane_without_lead', 
        'starting_straight_traffic_light_intersection_traversal'
    ]
    
 
    df['scenario_distribution'] = np.where(
        df['scenario_type'].isin(scenarios),   
        label,                                
        df['scenario_distribution']            
    )
    return df

def find_common_scenario_types(plantf_ind, gameformer_ind):
    """
    Find and count the common scenario types between plantf_ind and gameformer_ind.

    Parameters:
    - plantf_ind (pd.Series): A Pandas Series containing scenario types and their counts for plantf_ind.
    - gameformer_ind (pd.Series): A Pandas Series containing scenario types and their counts for gameformer_ind.

    Returns:
    - pd.DataFrame: A DataFrame with common scenario types and their counts in both plantf_ind and gameformer_ind.
    """
    # Convert the Series to DataFrames for easy manipulation
    plantf_df = plantf_ind.reset_index()
    gameformer_df = gameformer_ind.reset_index()

    # Rename columns for clarity
    plantf_df.columns = ['scenario_type', 'plantf_count']
    gameformer_df.columns = ['scenario_type', 'gameformer_count']

    # Merge the two DataFrames on 'scenario_type' to find common types
    common_df = pd.merge(plantf_df, gameformer_df, on='scenario_type', how='inner')

    # Sort the result by scenario_type or counts (optional)
    common_df = common_df.sort_values(by=['plantf_count', 'gameformer_count'], ascending=False)
    print(common_df)
    return common_df

