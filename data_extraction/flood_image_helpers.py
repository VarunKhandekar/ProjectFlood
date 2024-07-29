import pandas as pd
import json

def generate_overlapping_events(file_path: str) -> pd.DataFrame:
    """
    Identify overlapping flood events in a dataset and annotate each event with the indices of overlapping events.

    Args:
        file_path (str): Path to the CSV or Excel file containing flood event data.

    Returns:
        pd.DataFrame: A DataFrame with an additional column 'overlapping_events' indicating overlapping event indices.

    Raises:
        ValueError: If the provided file is not a CSV or Excel file.
        AssertionError: If the required columns are not present in the dataset.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError("Please provide a CSV or Excel file.")
    
    # Data file integrity checks
    required_columns = ['index', 'dfo_began_uk', 'dfo_ended_uk']
    assert all(column in df.columns for column in required_columns), f"Missing columns: {[column for column in required_columns if column not in df.columns]}"

    df['dfo_began_uk'] = pd.to_datetime(df['dfo_began_uk'], dayfirst=True)
    df['dfo_ended_uk'] = pd.to_datetime(df['dfo_ended_uk'], dayfirst=True)

    df['overlapping_events'] = ''

    for i, row in df.iterrows():
        overlaps = []
        for j, compare_row in df.iterrows():
            if i != j:
                if not (row['dfo_ended_uk'] < compare_row['dfo_began_uk'] or row['dfo_began_uk'] > compare_row['dfo_ended_uk']):
                    overlaps.append(j)
        df.at[i, 'overlapping_events'] = pd.NA
        if len(overlaps) != 0:
            df.at[i, 'overlapping_events'] = ','.join(map(str, overlaps))

    return df

def generate_flood_events(file_path: str) -> dict:
    """
    Generate a dictionary of flood events with overlapping events merged under the earliest start date.

    Args:
        file_path (str): Path to the CSV or Excel file containing flood event data.

    Returns:
        dict: A dictionary where keys are the earliest event dates and values are lists of event indices that overlap.

    Raises:
        ValueError: If the provided file is not a CSV or Excel file.
        AssertionError: If the required columns are not present in the dataset.
    """
    # Preprocess raw flood data
    df = generate_overlapping_events(file_path)
    
    overlapping_dict = {}

    for idx, row in df.iterrows():
        dfo_began_uk = row['dfo_began_uk']
        overlapping_events = row['overlapping_events']
        
        # Create list for overlapping events (list includes the current row itself)
        overlapping_events_list = [idx] # add current row to list
        if pd.notna(overlapping_events):
            overlapping_events_list = [i for i in list(map(int, overlapping_events.split(',')))] # get rows of overlapping events
        
        overlapping_events_list = [df['index'][i] for i in overlapping_events_list]

        # Find earliest event date; 
        # REMOVE THIS LOGIC IF YOU WANT MORE DATAPOINTS, I.E. NOT MERGING TO THE START DATE OF THE EARLIEST OVERLAPPING EVENT
        earliest_event_date = dfo_began_uk
        for event in overlapping_events_list:
            event_date = df[df['index'] == event]['dfo_began_uk'].values[0]
            event_date = pd.Timestamp(event_date)
            if event_date < earliest_event_date:
                earliest_event_date = event_date
        
        # Add the events to the dictionary with the earliest event date as key
        if earliest_event_date not in overlapping_dict:
            overlapping_dict[earliest_event_date] = overlapping_events_list
        else:
            overlapping_dict[earliest_event_date].extend(overlapping_events_list)
            overlapping_dict[earliest_event_date] = list(set(overlapping_dict[earliest_event_date]))
    
    return overlapping_dict

def store_flood_dates(events_dict: dict, config_file: str):
    """
    Store flood event dates into an Excel file.

    Args:
        events_dict (dict): A dictionary where keys are flood event dates.
        config_file (str): Path to the configuration file containing the target path for storing the flood dates.

    This function performs the following steps:
        1. Extracts the flood event dates from the keys of the `events_dict` dictionary.
        2. Creates a pandas DataFrame with the extracted dates.
        3. Reads the target path for storing the flood dates from the configuration file.
        4. Saves the DataFrame to an Excel file at the specified target path.

    Note:
        The configuration file should contain a key "final_flood_dates" specifying the target path for the Excel file.
    """
    dates = [i for i in events_dict.keys()]
    df = pd.DataFrame(dates, columns=["Dates"])
    with open(config_file) as config_file:
        config = json.load(config_file)
    target_path = config["final_flood_dates"]
    df.to_excel(target_path, index=False)
