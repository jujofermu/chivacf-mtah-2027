import numpy as np
import pandas as pd
import warnings
from kloppy import metrica, sportscode
from games import asset_paths

# Hide harmless NumPy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
paths = asset_paths()

def load_tracking_data():
    print("Loading tracking data...")
    dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
    df = dataset.to_df()
    
    home_cols, away_cols = [], []
    for team in dataset.metadata.teams:
        is_home = "barcelona" in team.name.lower() or "home" in str(team.ground).lower()
        for player in team.players:
            col_name = f"{player.player_id}_x" if f"{player.player_id}_x" in df.columns else f"{player.jersey_no}_x"
            if col_name in df.columns:
                (home_cols if is_home else away_cols).append(col_name)
    return df, home_cols, away_cols

def load_smart_tagging():
    print("Loading tactical events...")
    events = sportscode.load(paths["smart_tagging"]).to_df()
    return events[events['code'] == 'SET PIECES']

def calculate_closest_team(df, home_cols, away_cols):
    print("Calculating spatial analysis...")
    labels = np.full(len(df), 'X', dtype=object)
    
    ball_x, ball_y = df['ball_x'].values, df['ball_y'].values
    
    h_x = df[home_cols].values
    h_y = df[[c.replace('_x', '_y') for c in home_cols]].values
    a_x = df[away_cols].values
    a_y = df[[c.replace('_x', '_y') for c in away_cols]].values
    
    for i in range(len(df)):
        if np.isnan(ball_x[i]):
            continue
            
        dist_h = np.nanmin(np.hypot(h_x[i] - ball_x[i], h_y[i] - ball_y[i]))
        dist_a = np.nanmin(np.hypot(a_x[i] - ball_x[i], a_y[i] - ball_y[i]))
        
        # Ampliamos ligeramente el radio de control a 3.5 metros
        if dist_h > 0.035 and dist_a > 0.035:
            labels[i] = 'X' 
        elif dist_h < dist_a:
            labels[i] = 'H'
        elif dist_a < dist_h:
            labels[i] = 'A'
            
    return labels

def smart_fill_gaps(labels_series, max_gap_frames=125):
    """
    If there is a short block of X (<= 5 seconds) surrounded by the same team (e.g., H -> X -> H),
    we fill in the X, assuming that the team never lost possession of the ball. 
    This increases the robustness of the possessions for spellF1.
    """
    print(f"Filling tracking gaps smaller than {max_gap_frames} frames...")
    labels = labels_series.copy().values
    
    start_x = -1
    for i in range(len(labels)):
        if labels[i] == 'X':
            if start_x == -1:
                start_x = i
        else:
            if start_x != -1:
                gap_len = i - start_x
                if gap_len <= max_gap_frames and start_x > 0:
                    prev_label = labels[start_x - 1]
                    next_label = labels[i]
                    
                    if prev_label == next_label and prev_label in ['H', 'A']:
                        labels[start_x:i] = prev_label
                start_x = -1
                
    return pd.Series(labels)

def clean_micro_possessions(labels_series, min_length=50):
    print(f"Cleaning up micro-poses shorter than {min_length} frames...")
    labels = labels_series.copy()
    current_label = labels[0]
    count = 0
    start_idx = 0
    
    for i in range(1, len(labels)):
        if labels[i] == current_label:
            count += 1
        else:
            if count < min_length and current_label in ['H', 'A']:
                labels.iloc[start_idx:i] = 'X'
            current_label = labels[i]
            count = 1
            start_idx = i
            
    return labels

def main():
    df, home_cols, away_cols = load_tracking_data()
    set_pieces = load_smart_tagging()
    
    raw_labels = pd.Series(calculate_closest_team(df, home_cols, away_cols))
    
    print("Applying maximum temporal inertia smoothing (125 frames)...")
    mapping = {'H': 1, 'A': -1, 'X': 0, 'D': 2}
    reverse_mapping = {1: 'H', -1: 'A', 0: 'X', 2: 'D'}
    num_series = raw_labels.map(mapping)
    
    def get_mode(x):
        valid = x[x != 0]
        if len(valid) == 0: return 0
        v, c = np.unique(valid, return_counts=True)
        return v[np.argmax(c)]
        
    smooth = num_series.rolling(125, center=True).apply(get_mode, raw=True).fillna(num_series)
    smooth_labels = smooth.map(reverse_mapping)
    
    # 1. Intelligent micro-cut infill
    filled_labels = smart_fill_gaps(smooth_labels, max_gap_frames=125)
    
    # 2. Aggressive injection of X during severe tracking losses
    ball_missing = df['ball_x'].isna()
    filled_labels[ball_missing] = 'X'
    
    # 3. Cleaning up rebounds (increasing to 50 frames = 2 seconds of actual minimum possession)
    final_labels = clean_micro_possessions(filled_labels, min_length=50)
    
    print("Injecting absolute 'D' state from a dead-ball situation...")
    for _, row in set_pieces.iterrows():
        start_frame = max(0, int(round(row['timestamp'].total_seconds() * 25)))
        end_frame = min(len(final_labels), int(round(row['end_timestamp'].total_seconds() * 25)))
        final_labels.iloc[start_frame:end_frame] = 'D'
    
    print("Saving final submission.csv...")
    submission_df = pd.DataFrame({'frame': range(len(final_labels)), 'label': final_labels.values})
    submission_df.to_csv('evaluation/results/chivacf.csv', index=False)
    print("Ready! Launch the final assessment.")

if __name__ == "__main__":
    main()