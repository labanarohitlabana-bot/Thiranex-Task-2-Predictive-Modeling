# Thiranex Task 2 — Predictive Modeling Using Machine Learning

## Project Overview
This project predicts a student's final score using study-related features. It demonstrates data preparation, missing-value handling, supervised model training, model comparison, and performance evaluation.

**Dataset notice:** The included dataset is synthetic and generated for practice. It does not represent real student data. If an official dataset was provided by the mentor, use that dataset instead.

## Features Used
- Study hours per day
- Attendance percentage
- Previous score
- Assignments completed
- Sleep hours per night

**Target:** Final score

## Machine Learning Algorithms
- Linear Regression
- Random Forest Regressor

## Model Evaluation
The models are evaluated using:
- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of determination

The project also generates actual-versus-predicted, residual, and feature-importance visualizations.

## How to Run on Windows
1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Install dependencies:

   `py -m pip install -r requirements.txt`

4. Run the model:

   `py train_model.py`

5. Check the `outputs` folder for metrics, predictions, and charts.

## Project Files
- `train_model.py` — training and evaluation script
- `data/student_performance_synthetic.csv` — practice dataset
- `outputs/model_metrics.csv` — model comparison metrics
- `outputs/test_predictions.csv` — test-set predictions
- `outputs/*.png` — evaluation charts
- `Thiranex_Task2_Report.docx` — project report

## Important
This project is for supervised-learning practice. Results from the synthetic dataset should not be presented as evidence of real-world student performance.