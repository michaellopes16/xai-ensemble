# XAI Ensemble for Electronic-Nose Time Series

Research code accompanying *Towards Transparent VOCs Detection: An XAI Ensemble Approach for E-Nose Time-Series*. It combines Grad-CAM, LIME, and SHAP to explain predictions from a trained classifier by identifying electronic-nose sensor channels supported by multiple explainers.

## Repository contents

- **xaiensembleapi/xaiensemble.py** contains the explanation API, consensus aggregation, and plotting utilities.
- **xaiensembleapi/xaiensemble_ex.py** implements explainer agreement, ablation, and sensitivity analyses. It computes Jaccard similarity, the overlap coefficient, and Kendall rank correlation.
- **xaiensembleapi/app.py** is a separate Streamlit demonstration of block perturbation with a simulated model.

## Installation

Use Python 3.11 with the TensorFlow 2.15.1 dependency pin. From the repository root:

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m pip install --no-deps -e .

On Linux or macOS, activate the environment with 'source .venv/bin/activate'.

## Running the evaluation

The evaluation requires training and test CSVs and a compatible trained Keras model. CSV files have no header. The final column is the class label and the preceding columns contain the flattened sensor readings. Use the same feature ordering and preprocessing as in model training.

The default configuration expects the following local files under 'xaiensembleapi/src/BloodData/':

- AllCandidas_TRAIN.csv
- AllCandidas_TEST.csv
- best_model.keras

In the main block of 'xaiensembleapi/xaiensemble_ex.py', check the sensor order, class order, sample index, and target convolutional layer against your model. To switch to culture recordings, update those settings and the CSV and model paths together. Supply the inputs before running:

    python -m xaiensembleapi.xaiensemble_ex

The research datasets, trained models, SQLite database, and generated results are excluded from this source snapshot. Full reproduction requires access to the corresponding inputs.

## Scope

The explanations describe the behavior of the evaluated classifiers. The software is a research prototype.
