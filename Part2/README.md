# CS 4375 – Assignment 3, Part II: Tweet Clustering

# Casey Nguyen, CXN220034 
# Nguyen Phuc Do, NPD220001

## Free days used

1 free day was used for this assignment.

## Requirements 

- Python 3.8+
- `requests`
- `pandas`
- Note: We used `requests` and `panda` for data loading and processing only. The K-means logic is all written by us.

## Dataset

The script fetches tweets online from UCI dataset `438`:
https://archive.ics.uci.edu/static/public/438/health+news+in+twitter.zip
- We fetched the UCI dataset archive using the `requests` library so that there is no need for manual downloading
- With the `zipfile` module, the script goes through the archivewithout having to extract the files
- We used `pandas` to separate columns per tweet and identify them into 3 parts

The dataset contains these txt files for running:
- goodhealth.txt
- nytimeshealth.txt
- cbchealth.txt
- cnnhealth.txt
- reuters_health.txt
- latimeshealth.txt
- nprhealth.txt
- wsjhealth.txt
- NBChealth.txt
- KaiserHealthNews.txt
- gdnhealthcare.txt
- everydayhealth.txt
- bbchealth.txt
- msnhealthnews.txt
- foxnewshealth.txt
- usnewshealth.txt

## How to Run

Enter to the right directory, and install required libraries

``` bash
cd Part2
```

```
python -m venv venv
.\venv\Scripts\Activate
```

```bash
pip install requests pandas
```

You can run bbchealth.txt by default: 
```bash
python kmeans_tweets.py
```

Or you can pass any .txt file as well: 
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
