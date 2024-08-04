import pandas as pd
import numpy as np
import time
import requests

from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=10_000, n_features=3, cluster_std=6, centers=[[40, 40, 40]], shuffle=True, random_state=15,
                  center_box=(30, 70))
x_outlier, y_outlier = make_blobs(n_samples=1_250, n_features=3, centers=5, shuffle=True, random_state=15,
                                  center_box=(5, 105))

for row in x_outlier:
    x = np.vstack([x, row])

x = pd.DataFrame(x, columns=['HUMIDITY', 'TEMPERATURE', 'SOUND_VOLUME'])

url = f'http://localhost:5001/invocations'
headers = {'Content-Type': 'application/json'}

for index, row in x.iterrows():
    payload = f'{{"inputs": [{{ "HUMIDITY": {row["HUMIDITY"]}, "TEMPERATURE": {row["TEMPERATURE"]}, "SOUND_VOLUME": {row["SOUND_VOLUME"]}}}]}}'
    r = requests.post(url, data=payload, headers=headers)
    print(r.text)
    time.sleep(1)
