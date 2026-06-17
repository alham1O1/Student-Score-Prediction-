
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)
N = 1000

attendance       = np.random.randint(50, 101, N)
study_hours      = np.random.uniform(0, 10, N)
prev_marks       = np.random.randint(30, 101, N)
assignments_done = np.random.randint(0, 11, N)
class_participation = np.random.randint(1, 6, N)
sleep_hours      = np.random.uniform(4, 10, N)
family_income    = np.random.randint(1, 6, N)
extra_activities = np.random.randint(0, 2, N)

score = (
    0.24 * (attendance / 100)
    + 0.20 * (study_hours / 10)
    + 0.18 * (prev_marks / 100)
    + 0.15 * (assignments_done / 10)
    + 0.09 * (class_participation / 5)
    + 0.06 * (sleep_hours / 10)
    + 0.04 * (family_income / 5)
    + 0.04 * extra_activities
)
noise = np.random.normal(0, 0.05, N)
score = np.clip(score + noise, 0, 1)

# Grade labels
def score_to_grade(s):
    if s >= 0.80: return 'A'
    elif s >= 0.65: return 'B'
    elif s >= 0.50: return 'C'
    elif s >= 0.40: return 'D'
    else: return 'F'

grades = np.array([score_to_grade(s) for s in score])
pass_fail = (score >= 0.40).astype(int)

df = pd.DataFrame({
    'attendance': attendance,
    'study_hours': np.round(study_hours, 1),
    'prev_marks': prev_marks,
    'assignments_done': assignments_done,
    'class_participation': class_participation,
    'sleep_hours': np.round(sleep_hours, 1),
    'family_income': family_income,
    'extra_activities': extra_activities,
    'grade': grades,
    'pass_fail': pass_fail
})

df.to_csv('student_data.csv', index=False)
print(f"Dataset created: {len(df)} records")
print(f"Grade distribution:\n{df['grade'].value_counts().sort_index()}")
print(f"Pass rate: {df['pass_fail'].mean()*100:.1f}%")

FEATURES = [
    'attendance',
    'study_hours',
    'prev_marks',
    'assignments_done',
    'class_participation',
    'sleep_hours',
    'family_income',
    'extra_activities',
]
TARGET_GRADE = 'grade'
TARGET_PASS  = 'pass_fail'

X = df[FEATURES]
y_grade = df[TARGET_GRADE]
y_pass  = df[TARGET_PASS]

X_train, X_test, yg_train, yg_test, yp_train, yp_test = train_test_split(
    X, y_grade, y_pass, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    'Logistic Regression':  LogisticRegression(max_iter=500, random_state=42),
    'Decision Tree':        DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN':                  KNeighborsClassifier(n_neighbors=7),
}

print("\n--- Grade Prediction (A/B/C/D/F) ---")
best_acc   = 0
best_name  = ''
best_model = None

for name, mdl in models.items():
    if name == 'Logistic Regression' or name == 'KNN':
        mdl.fit(X_train_s, yg_train)
        preds = mdl.predict(X_test_s)
    else:
        mdl.fit(X_train, yg_train)
        preds = mdl.predict(X_test)
    acc = accuracy_score(yg_test, preds)
    print(f"  {name}: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc   = acc
        best_name  = name
        best_model = mdl

print(f"\nBest model: {best_name} ({best_acc*100:.2f}%)")

rf_grade = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
rf_grade.fit(X_train, yg_train)
rf_pass  = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
rf_pass.fit(X_train, yp_train)

lr_grade = LogisticRegression(max_iter=500, random_state=42)
lr_grade.fit(X_train_s, yg_train)
lr_pass  = LogisticRegression(max_iter=500, random_state=42)
lr_pass.fit(X_train_s, yp_train)

dt_grade = DecisionTreeClassifier(max_depth=8, random_state=42)
dt_grade.fit(X_train, yg_train)
knn_grade = KNeighborsClassifier(n_neighbors=7)
knn_grade.fit(X_train_s, yg_train)

os.makedirs('models', exist_ok=True)

model_bundle = {
    'scaler': scaler,
    'features': FEATURES,
    'rf_grade':  rf_grade,
    'rf_pass':   rf_pass,
    'lr_grade':  lr_grade,
    'lr_pass':   lr_pass,
    'dt_grade':  dt_grade,
    'knn_grade': knn_grade,
}

with open('models/model_bundle.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)

print("\nAll models saved to models/model_bundle.pkl")

rf_preds = rf_grade.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(yg_test, rf_preds))

importances = rf_grade.feature_importances_
fi = sorted(zip(FEATURES, importances), key=lambda x: -x[1])
print("\nFeature Importances (Random Forest):")
for feat, imp in fi:
    print(f"  {feat}: {imp:.4f}")
