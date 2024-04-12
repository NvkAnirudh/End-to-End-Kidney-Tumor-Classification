# Kidney Tumor Classification using Deep Learning, MLFlow, and DVC

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. Update app.py

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/NvkAnirudh/End-to-End-Kidney-Tumor-Classification
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -p venv python=3.9 -y
```

```bash
conda activate ./venv
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```

## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI='your_mlflow_uri'

export MLFLOW_TRACKING_USERNAME='your_username'

export MLFLOW_TRACKING_PASSWORD='your_password'

```
