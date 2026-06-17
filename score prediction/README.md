

---

## Project Overview
An ML-powered web application that predicts student academic performance (Grade A–F and Pass/Fail probability) using factors like attendance, study hours, previous marks, and assignment completion.

## ML Algorithms Used
- Random Forest (default, best accuracy)
- Logistic Regression
- Decision Tree
- K-Nearest Neighbours (KNN)

## Setup Instructions

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the models
```bash
python train_model.py
```
This generates `student_data.csv` and `models/model_bundle.pkl`.

### Step 3 — Run the app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## Project Structure
```
student_prediction/
├── app.py              ← Flask backend (routes + prediction logic)
├── train_model.py      ← Dataset generation + model training
├── requirements.txt    ← Python dependencies
├── student_data.csv    ← Synthetic dataset (generated)
├── models/
│   └── model_bundle.pkl ← Trained models (generated)
└── templates/
    └── index.html      ← Frontend UI
```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Frontend UI |
| POST | `/predict` | Single student prediction |
| POST | `/add_student` | Predict a student and save the record to CSV |
| GET | `/student_records` | Search and list saved student records |
| PATCH | `/student_records/<record_id>` | Update teacher feedback, marks, and action status |
| DELETE | `/student_records/<record_id>` | Delete a saved student record |
| GET | `/download_student_records` | Download the full saved student list with predictions as CSV |
| POST | `/batch_predict` | Multiple student prediction |

Saved student submissions are appended to `student_predictions.csv` with the student details, teacher marks, teacher feedback, action status, next action, input values, predicted grade, pass probability, risk score, recommendation, and improvement phase breakdown.

## Input Features
| Feature | Range | Description |
|---------|-------|-------------|
| attendance | 0–100% | Class attendance percentage |
| study_hours | 0–10 | Daily study hours |
| prev_marks | 0–100 | Previous semester marks |
| assignments_done | 0–10 | Assignments submitted |
| class_participation | 1–5 | Participation rating |
| sleep_hours | 4–10 | Daily sleep hours |
| family_income | 1–5 | Family income level |
| extra_activities | 0/1 | Extracurricular participation |
## Tech Stack
- **Backend**: Python, Flask
- **ML**: Scikit-learn (RandomForest, LogisticRegression, DecisionTree, KNN)
- **Data**: Pandas, NumPy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
