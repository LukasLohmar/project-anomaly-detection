import mlflow.sklearn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# set environment variables for mlflow and s3 bucket
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio_password'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = f'http://localhost:9000'

# set tracking
mlflow.set_tracking_uri(f'http://localhost:5000')

# set th experiment name
mlflow.set_experiment('anomaly-detection')

# enable autologger
mlflow.sklearn.autolog()

with mlflow.start_run() as run:
    x, y = make_blobs(n_samples=10_000, n_features=3, cluster_std=6, centers=[[40, 40, 40]], shuffle=True,
                      random_state=15, center_box=(30, 70))
    x_outlier, y_outlier = make_blobs(n_samples=1_250, n_features=3, centers=5, shuffle=True, random_state=15,
                                      center_box=(5, 105))

    for row in x_outlier:
        x = np.vstack([x, row])

    for row in y_outlier:
        y = np.append(y, 1)

    x = pd.DataFrame(x, columns=['HUMIDITY', 'TEMPERATURE', 'SOUND_VOLUME'])

    # setup plotting of dataset
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=x.get('HUMIDITY'), ys=x.get('TEMPERATURE'), zs=x.get('SOUND_VOLUME'))

    ax.set_xlabel('Humidity')
    ax.set_ylabel('Temperature')
    ax.set_zlabel('Sound')

    mlflow.log_figure(fig, 'dataset_plot.png')

    x_training_set, x_testing_set, y_training_set, y_testing_set = train_test_split(x, y,
                                                                                    test_size=0.3, random_state=15)

    # calculate fraction
    contamination = 1_250 / 10_000

    # actual training
    clf = IsolationForest(n_jobs=1, random_state=15, contamination=contamination)

    clf.fit(x_training_set)

    # evaluate performance
    scores_prediction = clf.decision_function(x)
    y_pred = clf.predict(x)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    # get metrics
    confusion_matrix = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot()

    mlflow.log_figure(disp.figure_, 'confusion_matrix.png')

    # accuracy
    print('accuracy:', accuracy_score(y, y_pred))
    print('confusion matrix:\n', confusion_matrix)
    print('classification report:\n', classification_report(y, y_pred))

    mlflow.log_metric('accuracy', accuracy_score(y, y_pred))
    mlflow.log_metric('true_positive', confusion_matrix[0][0])
    mlflow.log_metric('true_negative', confusion_matrix[1][1])
    mlflow.log_metric('false_positive', confusion_matrix[0][1])
    mlflow.log_metric('false_negative', confusion_matrix[1][0])

    # build docker image
    mlflow.models.build_docker(model_uri=f'runs:/{run.info.run_id}/model', name='anomaly-detection-model',
                               enable_mlserver=True)
