import os
import re
import argparse
import pandas as pd

# Sort numerically by the leading number before the underscore (1_, 2_, ..., 56_)
def get_prefix_number(filename):
    match = re.match(r'^(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')

def add_xy_diff(df):
    df = df.sort_values(by=['track_no', 'tframe']).reset_index(drop=True)

    distances_x = []
    distances_y = []

    for i in sorted(df['track_no'].unique()):
        track_data = df[df['track_no'] == i]
        for j in range(len(track_data) - 1):
            current_row = track_data.iloc[[j]]
            next_row = track_data.iloc[[j + 1]]
            dist_x = next_row['x'].values[0] - current_row['x'].values[0]
            dist_y = next_row['y'].values[0] - current_row['y'].values[0]
            distances_x.append(dist_x)
            distances_y.append(dist_y)

        distances_x.append(0)
        distances_y.append(0)

    df['displ_x'] = distances_x
    df['displ_y'] = distances_y

    return df

def load_and_combine_tracks(raw_files_path):
    # Get all csv files, skipping junk like .DS_Store
    files = [f for f in os.listdir(raw_files_path) if f.endswith('.csv')]

    # sort by prefix
    files.sort(key=get_prefix_number)

    dfs = []
    max_track_no = 0

    for f in files:
        print(f)
        full_path = os.path.join(raw_files_path, f)
        temp = pd.read_csv(full_path)
        temp = temp[['tframe', 'y', 'x', 'track_no']]
        temp['video_id'] = f
        
        # Remap track_no to consecutive integers starting at 1,
        # preserving the order tracks first appear in the file
        codes, uniques = pd.factorize(temp['track_no'], sort=True)
        temp['track_no'] = codes + 1  # start at 1 instead of 0

        # Now offset by the running max from previous files
        temp['track_no'] = temp['track_no'] + max_track_no
        max_track_no = temp['track_no'].max()

        dfs.append(temp)

    df = pd.concat(dfs, ignore_index=True)

    # Sort by track number, then by tframe within each track
    #df = df.sort_values(by=['track_no', 'tframe'], ignore_index=True)    
    df = add_xy_diff(df)

    print('The displ_x and displ_y were recalculated from the real data.')
    print(df.head(10))

    return df

def main():
    parser = argparse.ArgumentParser(description="Combine and renumber track CSV files.")
    parser.add_argument('raw_files_path', type=str, help="Path to the folder containing the CSV files")
    parser.add_argument('-o', '--output', type=str, default='all_data_combined.csv',
                         help="Output CSV filename (default: all_data_combined.csv)")
    args = parser.parse_args()

    df = load_and_combine_tracks(args.raw_files_path)
    df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()