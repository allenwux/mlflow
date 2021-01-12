
from sklearn.linear_model import LogisticRegression
import argparse
import pandas as pd
import json
import os
import pickle
import mlflow

parser = argparse.ArgumentParser("sklearn")
parser.add_argument("--data_file", type=str, help="data file")
args = parser.parse_args()
