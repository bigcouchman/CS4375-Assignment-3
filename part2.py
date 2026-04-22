import re
import random
import string
import csv
import time
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Tweet:
    id: str
    words: set

def preprocess_data(raw_data):
    text = re.sub(r'https?://\S+', '', raw_data.lower())
    
    words = []
    for i in text.split():
        if i.startswith('@'):
            continue

        word = i.strip(string.punctuation)
        if word:
            words.append(word)
    
    return set(words)

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


distance_cache = {}

def calc_jaccard_distance(a, b):
    cache_id = tuple(sorted((id(a), id(b))))
    if cache_id in distance_cache:
        return distance_cache[cache_id]
    
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    distance = 1 - (intersection / union) if union > 0 else 1.0
    distance_cache[cache_id] = distance

    return distance

def kmeans_cluster(tweet_list, k):
    distance_cache.clear()
    centroid = random.sample(tweet_list, k)
    
    iters = 0
    for i in range(100):
        iters = iters + 1
        cluster_list = [[] for _ in range(k)]
        
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
    path = "Health-Tweets/bbchealth.txt"
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
            mx, mn = max(sizes), min(sizes)
            ratio = mx / mn if mn > 0 else float('inf')
        
            writer.writerow([k, round(sse, 4), round(sse/len(tweets), 4), iterations, mx, mn, round(ratio, 2)])
            print(f"{k:>4} | {sse:>10.2f} | {iterations:>5} | {ratio:>6.1f}")