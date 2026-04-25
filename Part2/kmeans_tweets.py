# CS 4375 Assignment 2 by Casey Nguyen and Nguyen Do
# Tweet Clustering using Jaccard Distance Calculation

# Import libraries
import re
import random
import string
import csv
import sys
import os
import time
from datetime import datetime

# Preprocessing function
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)

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
def load_tweets(path):
    """Load and preprocess a tweet file."""

    # Every tweet is divided to 3 parts, take the 3rd part (text)
    tweets = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('|', 2)
            if len(parts) == 3:
                words = preprocess(parts[2])
                if words:
                    tweets.append(words)
    return tweets

# Jaccard Distance function
def jaccard(a, b):
    union = a | b
    return 1.0 - len(a & b) / len(union) if union else 0.0

# K-Means Clustering
def kmeans(tweets, k, max_iter=100, seed=None):
    """K-means clustering using Jaccard distance and medoid centroid update."""
    rng = random.Random(seed)
    n = len(tweets)
    centroids = rng.sample(range(n), k)         # Select k random unique tweets to be initial centroids

    for _ in range(max_iter):
        # Assignment; each centroid contains the closest tweets
        clusters = [[] for _ in range(k)]
        for i in range(n):
            best = min(range(k), key=lambda c: jaccard(tweets[i], tweets[centroids[c]]))
            clusters[best].append(i)

        # Centroid Update
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
        
        # Check if centroids update
        if new_centroids == centroids:
            break
        centroids = new_centroids

    # Find SSE
    sse = sum(
        jaccard(tweets[i], tweets[centroids[c]]) ** 2
        for c, members in enumerate(clusters)
        for i in members
    )
    return clusters, sse

# Main

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'Health-Tweets/bbchealth.txt'
    tweets = load_tweets(path)
    print(f"Loaded {len(tweets)} tweets from {path}\n")

    k_values = [5, 10, 15, 20, 25]
    rows = []

    print(f"{'K':<5} {'SSE':<12} {'Runtime(s)':<12}  Cluster sizes")
    print('-' * 90)

    for k in k_values:
        start_time = time.time()

        # Clustering with fixed seed, and csv file format
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

    # Stores model runs in a folder called results and names .csv by number
    base_folder = "results"
    os.makedirs(base_folder, exist_ok=True)

    run_number = 1
    while os.path.exists(os.path.join(base_folder, f"run{run_number}.csv")):
        run_number += 1

    file_path = os.path.join(base_folder, f"run{run_number}.csv")

    # Every run record these values to be put in a table (needed for submission)
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