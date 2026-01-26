# Datasets

## bank-full

https://www.kaggle.com/datasets/hariharanpavan/bank-marketing-dataset-analysis-classification?resource=download

## Running in Google Colab

1. Open `notebooks/colab_runner.ipynb` in Colab (File → Open Notebook → GitHub or upload the repo).
2. If you're starting from a blank runtime, set `REPO_URL` in the first code cell so the notebook can clone this repository into `/content/clustering`.
3. Run the dependency-install cell, choose a dataset metadata file plus the pipelines you want to execute, and run the execution cell. Results land under `results/<pipeline>/<config>/` just like they do locally.
