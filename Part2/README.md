# CS 4375 – Assignment 3, Part II: Tweet Clustering

# Casey Nguyen, CXN220034 
# Nguyen Phuc Do, NPD220001

## Requirements

- Python 3.8+
- `requests`
- `pandas`

## Dataset

The script fetches tweets online from UCI dataset `438`:
https://archive.ics.uci.edu/static/public/438/health+news+in+twitter.zip

## How to Run

```
.\.venv\Scripts\Activate
```

``` bash
cd Part2
```

```bash
pip install requests pandas
```

runs bbchealth.txt by default 
```bash
python kmeans_tweets.py
```

can pass a .txt file as well 

```bash
python kmeans_tweets.py bbchealth.txt
```

## Output

The commands prints a results table in the terminal and saves a copy to `/results/run#_datasetname.csv`

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
