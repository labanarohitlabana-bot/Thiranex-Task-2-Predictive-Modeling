# Thiranex Task 2 — Predictive Modeling Using Machine Learning

## Project overview
This project predicts a student's final score from study-related features. It demonstrates data preparation, missing-value handling, model training, model comparison, and visual evaluation.

**Dataset notice:** `data/student_performance_synthetic.csv` is a simulated practice dataset generated for this project. It is not real student data and must not be described as real-world evidence. If your mentor provides an official dataset, use that instead and update the report.

## Features
- `study_hours_per_day`
- `attendance_percent`
- `previous_score`
- `assignments_completed_out_of_10`
- `sleep_hours_per_night`
- Target: `final_score`

## Models
1. Linear Regression
2. Random Forest Regressor

## Setup (Windows laptop)
1. Install Python 3.10 or later from https://www.python.org/downloads/ (during setup, tick **Add Python to PATH**).
2. Extract this ZIP to a folder.
3. Open the folder in VS Code.
4. Open Terminal → New Terminal.
5. Run:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   py -m pip install -r requirements.txt
   py train_model.py
   ```
   If `py` is not recognized, use `python` instead.
6. Open the `outputs` folder to see metrics, predictions, and charts.

## Evaluation
- MAE: average absolute prediction error; lower is better.
- RMSE: penalizes larger errors; lower is better.
- R²: proportion of variance explained; closer to 1 is generally better.

Because this is a regression project, regression metrics and actual-vs-predicted/residual plots are appropriate. Confusion matrices and ROC curves apply to classification problems, not score prediction.

## Files
- `train_model.py`: full reproducible training/evaluation script
- `data/student_performance_synthetic.csv`: practice dataset
- `outputs/model_metrics.csv`: model comparison
- `outputs/test_predictions.csv`: held-out test predictions
- `outputs/*.png`: charts
- `Thiranex_Task2_Report.docx`: report with results and charts

## Before submitting
- Check the portal's accepted file format and upload instructions.
- Review the report and understand the workflow/results.
- If the portal/mentor supplied a dataset or template, follow that over this sample project.
- Do not claim this synthetic dataset represents actual students.
