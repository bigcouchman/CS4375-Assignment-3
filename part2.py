# Assignment 3 Tweet Clustering by Nguyen Do and Casey Nguyen
# Cluster tweets using Jaccard Distance calculation

# Import libraries
import re
import random
import string
import csv
import time
from datetime import datetime
from dataclasses import dataclass

# A tweet class 
@dataclass
class Tweet:
    id: str
    words: set

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
            parts = line.strip().split('|')
            if len(parts) >= 3:
                preproc_tweet = preprocess_data(parts[2])
                if preproc_tweet:
                    tweet_list.append(preproc_tweet)
    
    return tweet_list

# A distance cache for jaccard distance
distance_cache = {}

# Calculate jaccard distance between 2 sets
def calc_jaccard_distance(a, b):
    cache_id = tuple(sorted((id(a), id(b))))
    if cache_id in distance_cache:
        return distance_cache[cache_id]
    
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    distance = 1 - (intersection / union) if union > 0 else 1.0
    distance_cache[cache_id] = distance

    return distance

# Perform k mean clustering
def kmeans_cluster(tweet_list, k):
    distance_cache.clear()
    centroid = random.sample(tweet_list, k)
    
    iters = 0
    for i in range(100):
        iters = iters + 1
        cluster_list = [[] for _ in range(k)]
        
        # Assign the tweet to a cluster based on the distance from the centroid
        for j in tweet_list:
            distance = [calc_jaccard_distance(j, l) for l in centroid]
            nearest_ind = distance.index(min(distance))
            cluster_list[nearest_ind].append(j)
        
        prev_centroid = list(centroid)

        for i in range(k):
            if not cluster_list[i]:
                centroid[i] = random.choice(tweet_list)
                continue

            new_centroid = None
            min_distance = float('inf')
            for j in cluster_list[i]:
                total_distance = sum(calc_jaccard_distance(j, l) for l in cluster_list[i])
                if total_distance < min_distance:
                    min_distance = total_distance
                    new_centroid = j

            centroid[i] = new_centroid
        
        if prev_centroid == centroid:
            break

    return cluster_list, centroid, iters

if __name__ == "__main__":
    
    path = "Health-Tweets/goodhealth.txt"
    tweets = loading_data(path)
    k_list = [5, 10, 15, 20, 25]
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_file = f"k_values_table_{timestamp}.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["K", "SSE", "SSE_per_Tweet", "Iterations", "Max_Size", "Min_Size", "Ratio"])
        print(f"{'K':>4} | {'SSE':>10} | {'Iteration':>5} | {'Ratio':>6}")
    
        for k in k_list:
            clusters, centroids, iterations = kmeans_cluster(tweets, k)
            sse = sum(sum(calc_jaccard_distance(c, centroids[i])**2 for c in clusters[i]) for i in range(k))

            sizes = [len(c) for c in clusters]
            size_output = ", ".join([f"{i+1}: {size} tweets" for i, size in enumerate(sizes)])

            print(f"{k:>4} | {sse:>10.2f} | {size_output}")
            writer.writerow([k, round(sse, 4), size_output])