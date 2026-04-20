import re

def preprocess_data(raw_data):
    text = raw_data.lower()
    text = re.sub('https?://\S+', '', text)
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

def calc_jaccard_distance(set1, set2):
    pass

def kmeans_cluster(tweet_list, k):
    pass

if __name__ == "__main__":
    path = "Health-Tweets/bbchealth.txt"