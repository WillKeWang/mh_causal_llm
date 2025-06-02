import os
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import matplotlib.pyplot as plt

# Function to get a list of participants who meet the condition for BDI2 scores
def get_bdi2_filtered_participants(filtered_pdf, condition):
    # Get the list of participants that meet the condition
    filtered_participants = filtered_pdf.loc[condition, 'pid'].tolist()
    return filtered_participants

# Function to extract selected sleep features for a list of participants from the rapids.csv
def get_sleep_features_for_participants(participant_list, globem_root, selected_sleep_features):
    all_sleep_data = []

    # Iterate through the years (INS-W_1, INS-W_2, INS-W_3, INS-W_4)
    for year in [ "INS-W_2", "INS-W_3", "INS-W_4"]:
        participant_folder = os.path.join(globem_root, year, "FeatureData")

        # Check if rapids.csv exists for the participants in the given year folder
        rapids_file = os.path.join(participant_folder, "rapids.csv")
        if os.path.isfile(rapids_file):
            rapids_df = pd.read_csv(rapids_file)

            # Filter for the data corresponding to the participants in the list
            participants_data_for_year = rapids_df[rapids_df["pid"].isin(participant_list)]

            # Only select the relevant columns (including 'pid', 'date', and selected sleep features)
            columns_to_keep = ["pid", "date"] + [feature for feature in selected_sleep_features if feature in rapids_df.columns]
            participants_data_for_year = participants_data_for_year[columns_to_keep]

            # Append the data for the year
            all_sleep_data.append(participants_data_for_year)

    # Concatenate the data from all years
    sleep_data_df = pd.concat(all_sleep_data, ignore_index=True)

    return sleep_data_df

# Function to plot sleep data for each participant
# Function to plot sleep data for each participant
def plot_sleep_data_for_participants(sleep_data_df, combined_info_df, save_folder):
    # Loop through each participant
    for pid in sleep_data_df['pid'].unique():
        # Filter data for the current participant
        participant_data = sleep_data_df[sleep_data_df['pid'] == pid]

        # Get the corresponding BDI2_PRE and BDI2_POST from the combined_info_df
        participant_info = combined_info_df[combined_info_df['pid'] == pid]

        # Debugging - Check if pid exists in combined_info_df
        if participant_info.empty:
            print(f"Warning: {pid} not found in combined_info_df.")

        if not participant_info.empty:
            # Check BDI2 values
            print(f"BDI2_PRE for {pid}: {participant_info['BDI2_PRE'].iloc[0]}")
            print(f"BDI2_POST for {pid}: {participant_info['BDI2_POST'].iloc[0]}")
            
            bdi2_pre = participant_info['BDI2_PRE'].iloc[0]
            bdi2_post = participant_info['BDI2_POST'].iloc[0]

            # Ensure values are not NaN
            if pd.isna(bdi2_pre) or pd.isna(bdi2_post):
                print(f"Warning: NaN values for BDI2 scores for participant {pid}.")
            
            bdi2_change = int(bdi2_post - bdi2_pre)
            print(f"BDI2 Change for {pid}: {bdi2_change}")

            # Create a subplot with two rows
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Plot morning and night sumdurations in the top subplot
            ax1.plot(participant_data['date'], 
                     participant_data['f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:morning'], 
                     label='Morning Duration', color='blue', marker='o', linestyle='-', markersize=5)
            ax1.plot(participant_data['date'], 
                     participant_data['f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:night'], 
                     label='Night Duration', color='orange', marker='o', linestyle='-', markersize=5)
            ax1.set_title(f"{pid} - Morning + Night Sumdurations\nBDI2 Change: {bdi2_change}")
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Duration (minutes)')
            ax1.legend()

            # Plot afternoon and evening sumdurations in the bottom subplot
            ax2.plot(participant_data['date'], 
                     participant_data['f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:afternoon'], 
                     label='Afternoon Duration', color='green', marker='o', linestyle='-', markersize=5)
            ax2.plot(participant_data['date'], 
                     participant_data['f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:evening'], 
                     label='Evening Duration', color='red', marker='o', linestyle='-', markersize=5)
            ax2.set_title(f"{pid} - Afternoon + Evening Sumdurations\nBDI2 Change: {bdi2_change}")
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Duration (minutes)')
            ax2.legend()

            # Save the plot with the participant's id and BDI2 change as the filename
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, f"{pid}_BDI2_change_{bdi2_change}_sleep_plot.png"))
            plt.close()

    print(f"Plots saved to {save_folder}")

# Main Execution

# Step 1: Filter out the participants based on BDI2 condition
filtered_pdf = pd.read_csv("combined_participants_info.csv")  # Replace with the actual path to your filtered PDF

# Apply the condition to count participants with normal BDI2 pre and abnormal BDI2 post
condition_abnormal = (filtered_pdf["BDI2_PRE"] <= 13) & (filtered_pdf["BDI2_POST"] >= 20)
abnormal_participants = get_bdi2_filtered_participants(filtered_pdf, condition_abnormal)

# Apply the condition to count participants with normal BDI2 pre and post
condition_normal = (filtered_pdf["BDI2_PRE"] <= 13) & (filtered_pdf["BDI2_POST"] <= 13)
normal_participants = get_bdi2_filtered_participants(filtered_pdf, condition_normal)

# Sample 22 participants from the normal group
sampled_normal_participants = pd.Series(normal_participants).sample(n=22, random_state=42).tolist()

# Combine the abnormal and sampled normal participants
all_participants = abnormal_participants + sampled_normal_participants

# Step 2: Extract the sleep features for these participants from rapids.csv files across the years
globem_root = "data/globem_raw"  # Update with the actual globem root directory
selected_sleep_features = [
    "f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:morning",
    "f_slp:fitbit_sleep_intraday_rapids_sumdurationawakeunifiedmain:morning",
    "f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:afternoon",
    "f_slp:fitbit_sleep_intraday_rapids_sumdurationawakeunifiedmain:afternoon",
    "f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:evening",
    "f_slp:fitbit_sleep_intraday_rapids_sumdurationawakeunifiedmain:evening",
    "f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:night",
    "f_slp:fitbit_sleep_intraday_rapids_sumdurationawakeunifiedmain:night",
]

sleep_data_df = get_sleep_features_for_participants(all_participants, globem_root, selected_sleep_features)

# Load the combined_participants_info.csv file for BDI2 scores
combined_info_df = pd.read_csv("combined_participants_info.csv")  # Replace with actual path

# Step 3: Plot and save the sleep data for each participant
save_folder = "figures"  # Update with the actual folder path to save plots
plot_sleep_data_for_participants(sleep_data_df, combined_info_df, save_folder)
