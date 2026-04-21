import re
import random

def preprocess_data(raw_data):
    text = raw_data.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = text.replace('#', '')
    words = set(text.split())
    return words

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

def calc_jaccard_distance(a, b):
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return 1 - (intersection / union)

def kmeans_cluster(tweet_list, k):
    centroid = random.sample(tweet_list, k)
    for i in range(100):
        print(f"Iteration {i+1}...")
        cluster_list = [[] for _ in range(k)]
        for j in tweet_list:
            distance = [calc_jaccard_distance(j, l) for l in centroid]
            nearest_ind = distance.index(min(distance))
            cluster_list[nearest_ind].append(j)
        
        prev_centroid = list(centroid)

        for i in range(k):
            if not cluster_list[i]:
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

    return cluster_list, centroid
            


if __name__ == "__main__":
    path = "Health-Tweets/usnewshealth.txt"
    tweets = loading_data(path)
    k_list = [5, 10, 15, 20, 25]

    for k in k_list:
        clusters, centroids = kmeans_cluster(tweets, k)
        sse = 0

        for i in range(len(clusters)):
            for j in clusters[i]:
                sse += calc_jaccard_distance(j, centroids[i])**2
        
        print(f"Value of K: {k}")
        print(f"SSE: {sse}")
        for i in range(k):
            print(f"{i+1}: {len(clusters[i])} tweets")
        print("-" * 30)