from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)
tweeteval = load_dataset("cardiffnlp/tweet_eval", "sentiment")
tweet_test = pd.DataFrame(tweeteval["test"])

# Save and verify 12,284 rows
tweet_test.to_csv("data/processed/tweet_test.csv", index=False)
print("TweetEval test rows:", len(tweet_test))