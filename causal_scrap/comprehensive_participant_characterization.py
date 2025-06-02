import os
import pandas as pd

# List of years to process
years = ['INS-W_1', 'INS-W_2', 'INS-W_3', 'INS-W_4']

# Merge pre and post-survey data
def merge_surveys(pre_file, post_file):
    pre = pd.read_csv(pre_file)
    post = pd.read_csv(post_file)
    
    # Remove "Unnamed" columns before merging
    pre = pre.loc[:, ~pre.columns.str.contains('^Unnamed')]
    post = post.loc[:, ~post.columns.str.contains('^Unnamed')]
    
    # Merge pre and post data on pid
    merged = pd.merge(pre, post, on="pid", suffixes=('_PRE', '_POST'))
    return merged

# Function to load data and stack all years by rows (with NaNs for missing columns)
def process_year_data(year):
    # Paths to data files
    dep_endterm_file = f"data/globem_raw/{year}/SurveyData/dep_endterm.csv"
    dep_weekly_file = f"data/globem_raw/{year}/SurveyData/dep_weekly.csv"
    pre_file = f"data/globem_raw/{year}/SurveyData/pre.csv"
    post_file = f"data/globem_raw/{year}/SurveyData/post.csv"
    missing_file = f"missing_by_pid.csv"

    # Load the data for the current year
    dep_endterm = pd.read_csv(dep_endterm_file)
    dep_weekly = pd.read_csv(dep_weekly_file)
    pre = pd.read_csv(pre_file)
    post = pd.read_csv(post_file)
    missing_data = pd.read_csv(missing_file)
    
    # Merge pre and post data
    merged_survey = merge_surveys(pre_file, post_file)
    
    # Merge the data (missing data, BDI scores, depression status, and survey data)
    final_data = pd.merge(missing_data, dep_endterm[['pid', 'BDI2', 'dep']], on="pid", how="left")
    final_data = final_data.merge(merged_survey, on="pid", how="left")

    return final_data

# Initialize an empty dataframe to store all years' data
all_years_data = pd.DataFrame()

# Process data for all years and combine them (stack them vertically)
for year in years:
    year_data = process_year_data(year)
    
    # Concatenate data from the current year to the accumulated data
    all_years_data = pd.concat([all_years_data, year_data], ignore_index=True, sort=False)

# Remove rows where BDI2_PRE or BDI2_POST are NaN
all_years_data_unique = all_years_data.dropna(subset=['BDI2', 'BDI2_PRE', 'BDI2_POST'])

# Remove duplicates based on 'pid' (ensure only one row per participant)
all_years_data_unique = all_years_data_unique.drop_duplicates(subset=['pid'])

# Save the combined dataframe to the current working folder
all_years_data_unique.to_csv("combined_participants_info.csv", index=False)

# Print the final dataframe shape
print(f"Final dataframe shape: {all_years_data_unique.shape}")
