# CS 4375 – Assignment 3, Part II: Tweet Clustering

# Casey Nguyen, CXN220034 
# Nguyen Phuc Do, NPD220001

## Requirements

- Python 3.8+
- No third-party libraries — uses the Python standard library

## Dataset

Download from UCI:
https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter

Go to the link and download the datafile. Then unzip the file and add to the part 2 
folder. Afterwards pick a .txt to run the model. 

Note the dataset is already in Part2 for you, the TA

## How to Run

``` bash
cd Part2
```

```bash
python kmeans_tweets.py Health-Tweets/bbchealth.txt
```

Or use the default path (runs bbchealth.txt):

```bash
python kmeans_tweets.py
```

## Output

The commands prints a results table in the termnial and saves a copy to `/results/runX.csv`

EX:

```
K     SSE         Runetime    Cluster sizes
----------------------------------------------------------------------
5     3342.1200   13.3737     1: 1900 tweets; 2: 540 tweets; ...
10    3241.5800   9.2982      1: 1600 tweets; 2: 100 tweets; ...
...
```

## Algorithm Notes

- Preprocessing: removes tweet ID and timestamp, strips `@mentions`, converts `#hashtag` → `hashtag`, removes URLs, lowercases all words.
- Similarity: Jaccard distance — `1 − |A ∩ B| / |A ∪ B|` on word sets.
- Centroid: medoid — the tweet in each cluster with minimum total Jaccard distance to all other members.
- K values tested: 5, 10, 15, 20, 25
