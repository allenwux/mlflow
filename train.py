
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

train_data = pd.read_csv(args.data_file)

X = train_data.drop([train_data.columns[-1]], axis=1)

Y = train_data[train_data.columns[-1]]

log_reg = LogisticRegression()
log_reg.fit(X, Y)

model_path = 'models'
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_file = os.path.join(model_path, "model.pkl")
pickle.dump(log_reg, open(model_file, 'wb'))

output_file = os.path.join(model_path, "output.json")

with open(output_file, 'w+') as file:
    json.dump({"message": "Hello World!"}, file)

with mlflow.start_run():
    active_run = mlflow.active_run()

    if active_run:
        print (active_run.info.run_id)
        mlflow.log_artifact(model_file, artifact_path='output')
        mlflow.log_artifact(output_file, artifact_path='output/output.json')
    else:
        print ("can not find the active run")
