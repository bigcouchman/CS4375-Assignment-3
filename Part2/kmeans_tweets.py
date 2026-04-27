# CS 4375 Assignment 3 by Casey Nguyen and Nguyen Do
# Tweet Clustering using Jaccard Distance Calculation
# We use 1 free day to tidy up our code and let it run without having the dataset
# downloaded on local computer. Instructions to run in README.md

# Import libraries
import io
import re
import random
import string
import csv
import sys
import os
import time
import zipfile
from datetime import datetime
import pandas as pd
import requests

# Constants to run the program
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
DATASET_URL = 'https://archive.ics.uci.edu/static/public/438/health+news+in+twitter.zip'
DEFAULT_FILE = 'Health-Tweets/bbchealth.txt'

# Preprocessing function
def preprocess(text):
    """Return a set of cleaned tokens from raw tweet text."""
    text = URL_RE.sub('', text.lower())
    tokens = []
    for word in text.split():
        if word.startswith('@'):          # remove @mentions
            continue
        word = word.lstrip('#')          # strip # from hashtags
        word = word.strip(string.punctuation)
        if word:                        # Add non empty words into list
            tokens.append(word)
    return frozenset(tokens)

# Let tweet file paths to be consistent when loading them
def check_path(tweet_file):
    member = tweet_file.replace('\\', '/').strip().lstrip('/')
    if not member:
        member = DEFAULT_FILE
    if '/' not in member:
        member = f'Health-Tweets/{member}'
    return member

# Data loading function, download from the UCI url and look through all txt files
def df_loading():
    try:
        response = requests.get(DATASET_URL, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(f'Failed to load dataset: {DATASET_URL}') from exc

    # Open the zip file from the url
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as a:
            dfs = []
            for name in a.namelist():
                if not name.lower().endswith('.txt'):
                    continue
                
                # For every file, separate the 3 parts per tweet
                with a.open(name) as file_obj:
                    try:
                        df = pd.read_csv(
                            file_obj,
                            sep='|',
                            header=None,
                            encoding='latin-1',
                            on_bad_lines='skip',
                            engine='python'
                        )
                    except (pd.errors.EmptyDataError, pd.errors.ParserError):
                        continue

                    if not isinstance(df, pd.DataFrame):
                        continue

                    # Handles ny extra columns or missing columns
                    df = df.copy()
                    if df.shape[1] < 3:
                        continue
                    if df.shape[1] > 3:
                        df[2] = df.loc[:, 2:].fillna('').astype(str).agg('|'.join, axis=1)

                    # Store the 3 parts, and add to the dataframe
                    df = df.iloc[:, :3]
                    df.columns = ['id', 'datetime', 'tweet']
                    df['source'] = name.replace('\\', '/')
                    dfs.append(df)

            if not dfs:
                raise ValueError('No files were valid.')

            return pd.concat(dfs, ignore_index=True)
    # Error handling for bad zip
    except zipfile.BadZipFile as exc:
        raise ValueError('Cannot download dataset') from exc

# When running the code, user can run on one file of their choice from the dataframe
def src_df(full_df, member_path):

    # Get the parts of the relative path when running the code 
    src = full_df['source'].astype(str).str.replace('\\', '/', regex=False)
    if '/' in member_path:
        mask = src.eq(member_path) | src.str.endswith('/' + member_path)
    else:
        mask = src.eq(member_path) | src.str.endswith('/' + member_path)
    
    # Error handling for non available file
    selected = full_df[mask]
    if selected.empty:
        available = sorted(src.unique())
        raise FileNotFoundError(
            f'{member_path} was not found.'
        )
    return selected

# Final loading function and preprocessing
def final_loading(tweet_file=DEFAULT_FILE):
    """Load and preprocess tweets from a specific .txt"""
    member_path = check_path(tweet_file)
    full_df = df_loading()
    source_df = src_df(full_df, member_path)

    # Store the tweets using the preprocessing function
    tweets = []
    for value in source_df['tweet'].fillna(''):
        words = preprocess(str(value))
        if words:
            tweets.append(words)

    # If tweets are invalid, return error
    if not tweets:
        raise ValueError(f'Invalid tweets from {member_path}.')

    # Return the tweets and paths
    return tweets, member_path

# Jaccard Distance function
def jaccard_distance(a, b):
    union = a | b
    return 1.0 - len(a & b) / len(union) if union else 0.0

# K-Means Clustering
def kmeans(tweets, k, max_iter=100, seed=None):
    """K-means clustering using Jaccard distance and medoid centroid update."""
    rng = random.Random(seed)
    n = len(tweets)
    centroids = rng.sample(range(n), k)

    # Assignment a centroid per cluster
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for i in range(n):
            best = min(range(k), key=lambda c: jaccard_distance(tweets[i], tweets[centroids[c]]))
            clusters[best].append(i)
        
        # Update the centroid, find the closest tweet to centroidper cluster
        new_centroids = []
        for ci, members in enumerate(clusters):
            if not members:
                new_centroids.append(rng.randrange(n))
                continue
            medoid = min(
                members,
                key=lambda m: sum(jaccard_distance(tweets[m], tweets[o]) for o in members)
            )
            new_centroids.append(medoid)

        # If the center doesn't change, break
        if new_centroids == centroids:
            break
        centroids = new_centroids

    # Find SSE per k value
    sse = sum(
        jaccard_distance(tweets[i], tweets[centroids[c]]) ** 2
        for c, members in enumerate(clusters)
        for i in members
    )
    return clusters, sse

# Main function
if __name__ == '__main__':
    tweet_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    
    # Get the tweets and path 
    try:
        tweets, member_path = final_loading(tweet_file)
    except (ConnectionError, FileNotFoundError, ValueError) as exc:
        print(f'Error loading tweets: {exc}')
        sys.exit(1)

    print(f"Loaded {len(tweets)} tweets from {member_path}\n")

    # Get dataset name
    dataset = os.path.basename(member_path)   # bbchealth.txt
    dataset = os.path.splitext(dataset)[0]    # bbchealth

    # Run the clustering for 5 k values
    k_values = [5, 10, 15, 20, 25]
    rows = []

    print(f"{'K':<5} {'SSE':<12} {'Runtime(s)':<12}  Cluster sizes")
    print('-' * 90)

    for k in k_values:
        # Call the k means clustering with a set seed, track time of clusters forming
        start_time = time.time()
        clusters, sse = kmeans(tweets, k, seed=42)
        end_time = time.time()
        runtime = round(end_time - start_time, 4)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Record these values in a table to store in the csv
        sizes = '; '.join(f'{i+1}: {len(c)} tweets' for i, c in enumerate(clusters))
        print(f"{k:<5} {sse:<12.4f} {runtime:<12}  {sizes}")
        rows.append({
            'Value of K': k,
            'SSE': round(sse, 4),
            'Runtime': runtime,
            'Timestamp': timestamp,
            'Cluster Size': sizes
        })

    # Save results
    res_folder = "results"
    os.makedirs(res_folder, exist_ok=True)

    # Store the results as run number for respective txt files
    run_num = 1
    while os.path.exists(os.path.join(res_folder, f"run{run_num}_{dataset}.csv")):
        run_num += 1

    file_path = os.path.join(res_folder, f"run{run_num}_{dataset}.csv")

    # Write in csv file and store in results
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'Value of K',
                'SSE',
                'Runtime',
                'Timestamp',
                'Cluster Size'
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nResults saved to {file_path}')