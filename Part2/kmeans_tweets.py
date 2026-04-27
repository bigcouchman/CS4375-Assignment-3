# CS 4375 Assignment 3 by Casey Nguyen and Nguyen Do
# Tweet Clustering using Jaccard Distance Calculation

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

# Preprocessing function
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
DATASET_438_ZIP_URL = 'https://archive.ics.uci.edu/static/public/438/health+news+in+twitter.zip'
DEFAULT_TWEET_FILE = 'Health-Tweets/bbchealth.txt'

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

# Loading function
def _normalize_member_path(tweet_file):
    member = tweet_file.replace('\\', '/').strip().lstrip('/')
    if not member:
        member = DEFAULT_TWEET_FILE
    if '/' not in member:
        member = f'Health-Tweets/{member}'
    return member


def _load_full_dataframe_from_dataset_438():
    try:
        response = requests.get(DATASET_438_ZIP_URL, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(f'Could not download dataset 438 from {DATASET_438_ZIP_URL}') from exc

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            dfs = []
            for name in archive.namelist():
                if not name.lower().endswith('.txt'):
                    continue

                with archive.open(name) as file_obj:
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

                    df = df.copy()

                    if df.shape[1] < 3:
                        continue

                    if df.shape[1] > 3:
                        df[2] = df.loc[:, 2:].fillna('').astype(str).agg('|'.join, axis=1)

                    df = df.iloc[:, :3]
                    df.columns = ['id', 'datetime', 'tweet']
                    df['source'] = name.replace('\\', '/')
                    dfs.append(df)

            if not dfs:
                raise ValueError('No .txt files were found.')

            return pd.concat(dfs, ignore_index=True)
    except zipfile.BadZipFile as exc:
        raise ValueError('Error processing downloaded zipfile.') from exc


def _select_source_dataframe(full_df, member_path):
    sources = full_df['source'].astype(str).str.replace('\\', '/', regex=False)
    if '/' in member_path:
        mask = sources.eq(member_path) | sources.str.endswith('/' + member_path)
    else:
        mask = sources.eq(member_path) | sources.str.endswith('/' + member_path)

    selected = full_df[mask]
    if selected.empty:
        available = sorted(sources.unique())
        sample = ', '.join(available[:6])
        raise FileNotFoundError(
            f'{member_path} was not found.'
        )
    return selected


def load_tweets_from_dataset_438(tweet_file=DEFAULT_TWEET_FILE):
    """Load and preprocess tweets from a specific .txt."""
    member_path = _normalize_member_path(tweet_file)
    full_df = _load_full_dataframe_from_dataset_438()
    source_df = _select_source_dataframe(full_df, member_path)

    tweets = []
    for value in source_df['tweet'].fillna(''):
        words = preprocess(str(value))
        if words:
            tweets.append(words)

    if not tweets:
        raise ValueError(f'No tweets were parsed from {member_path}.')

    return tweets, member_path

# Jaccard Distance function
def jaccard(a, b):
    union = a | b
    return 1.0 - len(a & b) / len(union) if union else 0.0

# K-Means Clustering
def kmeans(tweets, k, max_iter=100, seed=None):
    """K-means clustering using Jaccard distance and medoid centroid update."""
    rng = random.Random(seed)
    n = len(tweets)
    centroids = rng.sample(range(n), k)

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for i in range(n):
            best = min(range(k), key=lambda c: jaccard(tweets[i], tweets[centroids[c]]))
            clusters[best].append(i)

        new_centroids = []
        for ci, members in enumerate(clusters):
            if not members:
                new_centroids.append(rng.randrange(n))
                continue
            medoid = min(
                members,
                key=lambda m: sum(jaccard(tweets[m], tweets[o]) for o in members)
            )
            new_centroids.append(medoid)

        if new_centroids == centroids:
            break
        centroids = new_centroids

    sse = sum(
        jaccard(tweets[i], tweets[centroids[c]]) ** 2
        for c, members in enumerate(clusters)
        for i in members
    )
    return clusters, sse

# Main
if __name__ == '__main__':
    tweet_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TWEET_FILE

    try:
        tweets, member_path = load_tweets_from_dataset_438(tweet_file)
    except (ConnectionError, FileNotFoundError, ValueError) as exc:
        print(f'Error loading tweets from dataset 438: {exc}')
        sys.exit(1)

    print(f"Loaded {len(tweets)} tweets from dataset 438 file {member_path}\n")

    # 🔥 Extract dataset name for filename
    dataset_name = os.path.basename(member_path)        # bbchealth.txt
    dataset_name = os.path.splitext(dataset_name)[0]    # bbchealth

    k_values = [5, 10, 15, 20, 25]
    rows = []

    print(f"{'K':<5} {'SSE':<12} {'Runtime(s)':<12}  Cluster sizes")
    print('-' * 90)

    for k in k_values:
        start_time = time.time()

        clusters, sse = kmeans(tweets, k, seed=42)

        end_time = time.time()
        runtime = round(end_time - start_time, 4)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sizes = '; '.join(f'{i+1}: {len(c)} tweets' for i, c in enumerate(clusters))
        print(f"{k:<5} {sse:<12.4f} {runtime:<12}  {sizes}")

        rows.append({
            'Value of K': k,
            'SSE': round(sse, 4),
            'Runtime': runtime,
            'Timestamp': timestamp,
            'Size of each cluster': sizes
        })

    # Save results
    base_folder = "results"
    os.makedirs(base_folder, exist_ok=True)

    run_number = 1
    while os.path.exists(os.path.join(base_folder, f"run{run_number}_{dataset_name}.csv")):
        run_number += 1

    file_path = os.path.join(base_folder, f"run{run_number}_{dataset_name}.csv")

    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'Value of K',
                'SSE',
                'Runtime',
                'Timestamp',
                'Size of each cluster'
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nResults saved to {file_path}')