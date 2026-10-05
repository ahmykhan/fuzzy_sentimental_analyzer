import os
import pandas as pd
import pytest

def test_file_existence():
    assert os.path.exists("data/processed/sst_train.csv")
    assert os.path.exists("data/processed/sst_dev.csv")
    assert os.path.exists("data/processed/sst_test.csv")
    assert os.path.exists("data/processed/tweet_test.csv")

def test_sst_train_count():
    df = pd.read_csv("data/processed/sst_train.csv")
    assert len(df) == 8544

def test_sst_dev_count():
    df = pd.read_csv("data/processed/sst_dev.csv")
    assert len(df) == 1101

def test_sst_test_count():
    df = pd.read_csv("data/processed/sst_test.csv")
    assert len(df) == 2210

def test_tweet_test_count():
    df = pd.read_csv("data/processed/tweet_test.csv")
    assert len(df) == 12284

def test_no_null_sentences():
    df = pd.read_csv("data/processed/sst_train.csv")
    assert df["sentence"].isnull().sum() == 0