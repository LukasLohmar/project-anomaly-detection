# project-anomaly-detection

## build and run

1. create a `.env` inside this projects root folder with following content:
```.env
AWS_ACCESS_KEY_ID=minio
AWS_ACCESS_KEY=minio_password
MYSQL_DATABASE=mlflow_database
MYSQL_USER=mlflow
MYSQL_PASSWORD=mlflow_password
MYSQL_ROOT_PASSWORD=mysql_root_password
```
2. run ``docker compose up -d`` in root folder to start the project as a new docker container
    - if ``docker compose`` errors out with ``error getting credentials - err: exec: "docker-credential-desktop": executable file not found in %PATH%, out: ''`` -> change **credsStore** to **credStore** in ``%USERPROFILE%/.docker/config.json`` on windows or ``$HOME/.docker/config.json`` on linux
    - depending on OS or distribution ``docker compose`` is not a known command and/or aliased to ``docker-compose``
3. services available
    - MLflow on port [5000](http://localhost:5000)
    - minIO on port [9000/9001](http://localhost:9000) -> user=minio, password=minio_password
    - MYSQL database on port 3306 -> user=mysql, password=mysql_password, db=mlflow_database, root_password=mysql_root_password
4. log model into MLflow and generate Docker image for inference server
    - cd into `./src/model`
    - install dependencies with: `pip install -r requirements.txt`
    - log model into MLflow and build inference server: `python ./model.py`
    - run `docker run -p 5001:8080 -e DISABLE_NGINX=true "anomaly-detection-model"` to create a service running on port 5001 with the built inference server
5. test inference server
    - cd into `./src/model`
    - run the test-service: `python ./simulate.py`
