# Assignment 3 Tweet Clustering by Nguyen Do and Casey Nguyen
# Cluster tweets using Jaccard Distance calculation

# Import libraries
import re
import random
import string
import csv
import time
import sys
from datetime import datetime
from dataclasses import dataclass

# Data preprocessing
def preprocess_data(raw_data):
    # Strip down the tweets to just words, add into a list of words
    text = re.sub(r'https?://\S+', '', raw_data.lower())
    words = []
    for i in text.split():
        if i.startswith('@'):
            continue

        word = i.strip(string.punctuation).replace('#', '')
        if word:
            words.append(word)
    
    return set(words)

# For a specific file, preprocess every tweet in this file and add into a tweet list.
def loading_data(file_path):
    tweet_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|', 2)
            if len(parts) >= 3:
                preproc_tweet = preprocess_data(parts[2])
                if preproc_tweet:
                    tweet_list.append(preproc_tweet)
    
    return tweet_list

# A distance cache for jaccard distance
distance_cache = {}

# Calculate jaccard distance between 2 sets
def calc_jaccard_distance(a, b):
    intersection = len(a & b)
    union = len(a | b)
    return 1.0 - (intersection / union) if union else 0.0

# Perform k mean clustering
def kmeans_cluster(tweet_list, k, seed=None):
    random_seed = random.Random(seed)
    centroid = random_seed.sample(range(len(tweet_list)), k)
    
    iters = 0
    for i in range(100):
        iters = iters + 1
        cluster_list = [[] for _ in range(k)]
        
        # Assign the tweet to a cluster based on the distance from the centroid
        for tweet_ind, tweet_text in enumerate(tweet_list):
            distance = [calc_jaccard_distance(tweet_text, tweet_list[c]) for c in centroid]
            nearest_ind = distance.index(min(distance))
            cluster_list[nearest_ind].append(tweet_ind)
        
        prev_centroid = list(centroid)
        new_centroid = []

        for c, members in enumerate(cluster_list):
            if not members:
                new_centroid.append(random_seed.randrange(len(tweet_list)))
                continue
            
            best_member = min(members, key=lambda m: sum(calc_jaccard_distance(tweet_list[m], tweet_list[o]) for o in members))
            new_centroid.append(best_member)

        centroid = new_centroid
        
        if prev_centroid == centroid:
            break
    
    final_cluster_list = [[tweet_list[i] for i in c] for c in cluster_list]
    final_centroid_list = [tweet_list[i] for i in centroid]

    return final_cluster_list, final_centroid_list, iters

if __name__ == "__main__":
    
    path = "Health-Tweets/bbchealth.txt"
    tweets = loading_data(path)
    k_list = [5, 10, 15, 20, 25]
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_file = f"k_values_table_log.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["K", "SSE", "SSE_per_Tweet", "Iterations", "Max_Size", "Min_Size", "Ratio"])
        print(f"{'K':>4} | {'SSE':>10} | {'Iteration':>5} | {'Ratio':>6}")
    
        for k in k_list:
            clusters, centroids, iterations = kmeans_cluster(tweets, k, seed=42)
            sse = sum(sum(calc_jaccard_distance(c, centroids[i])**2 for c in clusters[i]) for i in range(k))

            sizes = [len(c) for c in clusters]
            size_output = ", ".join([f"{i+1}: {size} tweets" for i, size in enumerate(sizes)])

            print(f"{k:>4} | {sse:>10.2f} | {size_output}")
            writer.writerow([k, round(sse, 4), size_output])