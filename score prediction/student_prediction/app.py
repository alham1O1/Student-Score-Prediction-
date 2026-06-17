"""
Student Performance Prediction System - Flask Backend
Run: python app.py
Then visit: http://127.0.0.1:5000
"""

import csv
import io
import json
import pickle
import numpy as np
import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model_bundle.pkl')
STUDENT_RECORDS_PATH = os.path.join(os.path.dirname(__file__), 'student_predictions.csv')

LEGACY_SUBJECT_MARK_FIELDS = {
    'english_marks',
    'math_marks',
    'science_marks',
    'social_science_marks',
    'computer_marks',
}

IMPROVEMENT_FACTORS = [
    {
        'key': 'attendance',
        'label': 'Attendance',
        'target': 90,
        'default': 75,
        'minimum': 0,
        'maximum': 100,
        'weight': 1.15,
        'color': '#ef4444',
    },
    {
        'key': 'study_hours',
        'label': 'Study Time',
        'target': 6,
        'default': 4,
        'minimum': 0,
        'maximum': 10,
        'weight': 1.1,
        'color': '#f97316',
    },
    {
        'key': 'prev_marks',
        'label': 'Previous Marks',
        'target': 75,
        'default': 60,
        'minimum': 0,
        'maximum': 100,
        'weight': 1.0,
        'color': '#f59e0b',
    },
    {
        'key': 'assignments_done',
        'label': 'Assignments',
        'target': 9,
        'default': 7,
        'minimum': 0,
        'maximum': 10,
        'weight': 0.95,
        'color': '#3b82f6',
    },
    {
        'key': 'class_participation',
        'label': 'Participation',
        'target': 4,
        'default': 3,
        'minimum': 1,
        'maximum': 5,
        'weight': 0.85,
        'color': '#14b8a6',
    },
    {
        'key': 'sleep_hours',
        'label': 'Sleep Routine',
        'target': 7.5,
        'default': 7,
        'minimum': 4,
        'maximum': 10,
        'weight': 0.65,
        'color': '#a78bfa',
    },
]

STUDENT_RECORD_FIELDS = [
    'record_id',
    'created_at',
    'updated_at',
    'student_name',
    'roll_no',
    'student_class',
    'teacher_marks',
    'teacher_feedback',
    'action_status',
    'teacher_action',
    'action_note',
    'attendance',
    'study_hours',
    'prev_marks',
    'assignments_done',
    'class_participation',
    'sleep_hours',
    'family_income',
    'extra_activities',
    'algorithm',
    'predicted_grade',
    'grade_label',
    'pass_probability',
    'risk_score',
    'pass_fail',
    'advice',
    'improvement_phase',
    'improvement_score',
    'improvement_focus',
    'improvement_breakdown',
    'improvement_advice',
]

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

scaler    = bundle['scaler']
FEATURES  = bundle['features']
rf_grade  = bundle['rf_grade']
rf_pass   = bundle['rf_pass']
lr_grade  = bundle['lr_grade']
lr_pass   = bundle['lr_pass']
dt_grade  = bundle['dt_grade']
knn_grade = bundle['knn_grade']

GRADE_MESSAGES = {
    'A': {'label': 'Excellent',    'color': '#16a34a', 'advice': 'Outstanding! Keep up the great work. Consider mentoring peers.'},
    'B': {'label': 'Good',         'color': '#2563eb', 'advice': 'Good performance. A bit more focus can push you to excellence.'},
    'C': {'label': 'Average',      'color': '#d97706', 'advice': 'Average performance. Review priority areas and increase steady study time.'},
    'D': {'label': 'Below Average','color': '#dc2626', 'advice': 'Needs improvement. Attend extra classes and seek teacher guidance.'},
    'F': {'label': 'Fail Risk',    'color': '#7c2d12', 'advice': 'High failure risk! Immediate intervention needed. Contact your teacher now.'},
}

GRADE_SCORE_CENTERS = {
    'A': 0.86,
    'B': 0.72,
    'C': 0.56,
    'D': 0.44,
    'F': 0.28,
}


def score_to_grade(score: float) -> str:
    if score >= 0.80:
        return 'A'
    if score >= 0.65:
        return 'B'
    if score >= 0.50:
        return 'C'
    if score >= 0.40:
        return 'D'
    return 'F'


def estimate_grade_probabilities(score: float) -> dict:
    raw_scores = {
        grade: float(np.exp(-abs(score - center) * 10))
        for grade, center in GRADE_SCORE_CENTERS.items()
    }
    total = sum(raw_scores.values()) or 1
    return {grade: round((value / total) * 100, 1) for grade, value in raw_scores.items()}


def current_timestamp() -> str:
    return datetime.now().isoformat(timespec='seconds')


def create_record_id() -> str:
    return uuid.uuid4().hex[:12]


def parse_number(value, default, minimum=None, maximum=None) -> float:
    number = float(default) if value in (None, '') else float(value)

    if minimum is not None:
        number = max(float(minimum), number)
    if maximum is not None:
        number = min(float(maximum), number)

    return number


def parse_student_payload(data: dict) -> dict:
    prev_marks = parse_number(data.get('prev_marks', 60), 60, 0, 100)
    payload = {
        'attendance':          parse_number(data.get('attendance', 75), 75, 0, 100),
        'study_hours':         parse_number(data.get('study_hours', 4), 4, 0, 10),
        'prev_marks':          prev_marks,
        'assignments_done':    parse_number(data.get('assignments_done', 7), 7, 0, 10),
        'class_participation': parse_number(data.get('class_participation', 3), 3, 1, 5),
        'sleep_hours':         parse_number(data.get('sleep_hours', 7), 7, 4, 10),
        'family_income':       parse_number(data.get('family_income', 3), 3, 1, 5),
        'extra_activities':    parse_number(data.get('extra_activities', 0), 0, 0, 1),
    }

    return payload


def parse_teacher_profile(data: dict) -> dict:
    teacher_marks = data.get('teacher_marks', '')
    if teacher_marks not in (None, ''):
        teacher_marks = parse_number(teacher_marks, 0, 0, 100)
    else:
        teacher_marks = ''

    return {
        'student_name': str(data.get('student_name', '')).strip() or 'Unnamed Student',
        'roll_no': str(data.get('roll_no', '')).strip(),
        'student_class': str(data.get('student_class', '')).strip(),
        'teacher_marks': teacher_marks,
        'teacher_feedback': str(data.get('teacher_feedback', '')).strip(),
    }


def get_model_feature_value(data: dict, feature: str) -> float:
    if feature in LEGACY_SUBJECT_MARK_FIELDS:
        return parse_number(data.get('prev_marks', 60), 60, 0, 100)

    if feature in data:
        return data[feature]

    return 0


def build_model_input(data: dict) -> np.ndarray:
    return np.array([[get_model_feature_value(data, feature) for feature in FEATURES]])


def calculate_improvement_summary(data: dict, risk_score: float = 0) -> dict:
    factor_needs = []
    weighted_need = 0
    total_weight = 0

    for factor in IMPROVEMENT_FACTORS:
        value = parse_number(
            data.get(factor['key'], factor['default']),
            factor['default'],
            factor['minimum'],
            factor['maximum'],
        )
        target = factor['target']
        gap_ratio = max(target - value, 0) / target
        need_score = round(gap_ratio * 100, 1)
        weighted_need += need_score * factor['weight']
        total_weight += factor['weight']

        if need_score > 0:
            factor_needs.append({
                'label': factor['label'],
                'need': need_score,
                'color': factor['color'],
            })

    behavior_need = weighted_need / total_weight if total_weight else 0
    combined_need = round(min(100, (0.65 * behavior_need) + (0.35 * risk_score)), 1)

    if combined_need >= 65:
        phase = 'Critical Improvement'
    elif combined_need >= 45:
        phase = 'High Improvement'
    elif combined_need >= 25:
        phase = 'Moderate Improvement'
    elif combined_need >= 10:
        phase = 'Light Improvement'
    else:
        phase = 'On Track'

    factor_needs.sort(key=lambda item: item['need'], reverse=True)

    if factor_needs:
        total_need = sum(item['need'] for item in factor_needs) or 1
        breakdown = [
            {
                'label': item['label'],
                'value': round((item['need'] / total_need) * 100, 1),
                'need': item['need'],
                'color': item['color'],
            }
            for item in factor_needs
        ]
        focus = factor_needs[0]['label']
        top_labels = [item['label'] for item in factor_needs[:2]]
        if len(top_labels) == 1:
            improvement_advice = f"{phase}: prioritize {top_labels[0]} in the next improvement cycle."
        else:
            improvement_advice = f"{phase}: prioritize {top_labels[0]} and {top_labels[1]} in the next improvement cycle."
    elif combined_need >= 10:
        breakdown = [{'label': 'Prediction Risk', 'value': 100, 'need': combined_need, 'color': '#ef4444'}]
        focus = 'Prediction Risk'
        improvement_advice = f"{phase}: review the prediction risk with the student and monitor progress weekly."
    else:
        breakdown = [{'label': 'Maintain Current Level', 'value': 100, 'need': 0, 'color': '#22c55e'}]
        focus = 'Maintain Current Level'
        improvement_advice = 'On Track: maintain the current routine and keep monitoring progress weekly.'

    return {
        'improvement_phase': phase,
        'improvement_score': combined_need,
        'improvement_focus': focus,
        'improvement_breakdown': breakdown,
        'improvement_advice': improvement_advice,
    }


def predict_student(data: dict, algorithm: str = 'random_forest') -> dict:
    features_input = build_model_input(data)

    scaled_input = scaler.transform(features_input)

    if algorithm == 'logistic_regression':
        grade = lr_grade.predict(scaled_input)[0]
        proba = lr_grade.predict_proba(scaled_input)[0]
        classes = lr_grade.classes_
        pass_prob = float(lr_pass.predict_proba(scaled_input)[0][1])
    elif algorithm == 'decision_tree':
        grade = dt_grade.predict(features_input)[0]
        proba = dt_grade.predict_proba(features_input)[0]
        classes = dt_grade.classes_
        pass_prob = float(rf_pass.predict_proba(features_input)[0][1])
    elif algorithm == 'knn':
        grade = knn_grade.predict(scaled_input)[0]
        proba = knn_grade.predict_proba(scaled_input)[0]
        classes = knn_grade.classes_
        pass_prob = float(lr_pass.predict_proba(scaled_input)[0][1])
    else:  # random_forest (default)
        grade = rf_grade.predict(features_input)[0]
        proba = rf_grade.predict_proba(features_input)[0]
        classes = rf_grade.classes_
        pass_prob = float(rf_pass.predict_proba(features_input)[0][1])

    grade_proba = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}
    risk_score = round((1 - pass_prob) * 100, 1)
    info = GRADE_MESSAGES.get(grade, GRADE_MESSAGES['C'])
    improvement_summary = calculate_improvement_summary(data, risk_score)

    return {
        'grade': grade,
        'label': info['label'],
        'color': info['color'],
        'advice': f"{info['advice']} {improvement_summary['improvement_advice']}",
        'pass_probability': round(pass_prob * 100, 1),
        'risk_score': risk_score,
        'grade_probabilities': grade_proba,
        'algorithm_used': algorithm,
        **improvement_summary,
    }


def normalize_student_record(row: dict) -> dict:
    created_at = row.get('created_at') or current_timestamp()
    normalized = {field: row.get(field, '') for field in STUDENT_RECORD_FIELDS}
    normalized['record_id'] = normalized.get('record_id') or create_record_id()
    normalized['created_at'] = created_at
    normalized['updated_at'] = normalized.get('updated_at') or created_at
    normalized['action_status'] = normalized.get('action_status') or 'Needs Review'

    if any(not normalized.get(field) for field in (
        'improvement_phase',
        'improvement_score',
        'improvement_focus',
        'improvement_breakdown',
        'improvement_advice',
    )):
        risk_score = parse_number(normalized.get('risk_score', 0), 0, 0, 100)
        improvement_summary = calculate_improvement_summary(normalized, risk_score)
        normalized['improvement_phase'] = improvement_summary['improvement_phase']
        normalized['improvement_score'] = improvement_summary['improvement_score']
        normalized['improvement_focus'] = improvement_summary['improvement_focus']
        normalized['improvement_breakdown'] = json.dumps(improvement_summary['improvement_breakdown'])
        normalized['improvement_advice'] = improvement_summary['improvement_advice']

    return normalized


def ensure_student_records_file() -> None:
    if not os.path.exists(STUDENT_RECORDS_PATH) or os.path.getsize(STUDENT_RECORDS_PATH) == 0:
        with open(STUDENT_RECORDS_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=STUDENT_RECORD_FIELDS)
            writer.writeheader()
        return

    with open(STUDENT_RECORDS_PATH, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        needs_rewrite = (
            reader.fieldnames != STUDENT_RECORD_FIELDS
            or any(not row.get('record_id') for row in rows)
        )

    if not needs_rewrite:
        return

    with open(STUDENT_RECORDS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=STUDENT_RECORD_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_student_record(row))


def read_student_records() -> list:
    ensure_student_records_file()
    with open(STUDENT_RECORDS_PATH, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = [normalize_student_record(row) for row in reader]

    return sorted(records, key=lambda row: row.get('created_at', ''), reverse=True)


def write_student_records(records: list) -> None:
    with open(STUDENT_RECORDS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=STUDENT_RECORD_FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow(normalize_student_record(row))


def summarize_student_records(records: list) -> dict:
    total = len(records)
    pass_values = []
    at_risk = 0

    for record in records:
        try:
            pass_values.append(float(record.get('pass_probability') or 0))
            if float(record.get('risk_score') or 0) > 50:
                at_risk += 1
        except ValueError:
            continue

    average_pass = round(sum(pass_values) / len(pass_values), 1) if pass_values else 0
    return {'total': total, 'at_risk': at_risk, 'average_pass': average_pass}


def save_student_record(profile: dict, student_data: dict, algorithm: str, result: dict) -> dict:
    created_at = current_timestamp()
    row = {
        'record_id': create_record_id(),
        'created_at': created_at,
        'updated_at': created_at,
        'student_name': profile.get('student_name') or 'Unnamed Student',
        'roll_no': profile.get('roll_no', ''),
        'student_class': profile.get('student_class', ''),
        'teacher_marks': profile.get('teacher_marks', ''),
        'teacher_feedback': profile.get('teacher_feedback', ''),
        **student_data,
        'action_status': 'Needs Review',
        'teacher_action': '',
        'action_note': '',
        'algorithm': algorithm,
        'predicted_grade': result['grade'],
        'grade_label': result['label'],
        'pass_probability': result['pass_probability'],
        'risk_score': result['risk_score'],
        'pass_fail': 1 if result['pass_probability'] >= 50 else 0,
        'advice': result['advice'],
        'improvement_phase': result.get('improvement_phase', ''),
        'improvement_score': result.get('improvement_score', ''),
        'improvement_focus': result.get('improvement_focus', ''),
        'improvement_breakdown': json.dumps(result.get('improvement_breakdown', [])),
        'improvement_advice': result.get('improvement_advice', ''),
    }

    ensure_student_records_file()

    with open(STUDENT_RECORDS_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=STUDENT_RECORD_FIELDS)
        writer.writerow(row)

    return row


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        student_data = parse_student_payload(data)
        algorithm = data.get('algorithm', 'random_forest')
        result = predict_student(student_data, algorithm)

        return jsonify({'success': True, 'result': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/add_student', methods=['POST'])
def add_student():
    """Predict for a student and store the submitted record in CSV."""
    try:
        data = request.get_json() or {}
        student_data = parse_student_payload(data)
        algorithm = data.get('algorithm', 'random_forest')
        result = predict_student(student_data, algorithm)

        profile = parse_teacher_profile(data)
        saved_row = save_student_record(profile, student_data, algorithm, result)

        return jsonify({
            'success': True,
            'result': result,
            'saved': True,
            'csv_file': os.path.basename(STUDENT_RECORDS_PATH),
            'record': {
                'record_id': saved_row['record_id'],
                'student_name': saved_row['student_name'],
                'roll_no': saved_row['roll_no'],
                'student_class': saved_row['student_class'],
                'teacher_marks': saved_row['teacher_marks'],
                'teacher_feedback': saved_row['teacher_feedback'],
                'action_status': saved_row['action_status'],
                'improvement_phase': saved_row['improvement_phase'],
                'improvement_score': saved_row['improvement_score'],
                'improvement_focus': saved_row['improvement_focus'],
                'improvement_breakdown': saved_row['improvement_breakdown'],
                'improvement_advice': saved_row['improvement_advice'],
                'created_at': saved_row['created_at'],
            },
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/student_records', methods=['GET'])
def student_records():
    """Return saved student performance records, optionally filtered by a search term."""
    try:
        query = request.args.get('query', '').strip().lower()
        records = read_student_records()

        if query:
            records = [
                record for record in records
                if query in record.get('student_name', '').lower()
                or query in record.get('roll_no', '').lower()
                or query in record.get('student_class', '').lower()
            ]

        return jsonify({
            'success': True,
            'records': records,
            'summary': summarize_student_records(records),
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/student_records/<record_id>', methods=['PATCH'])
def update_student_record(record_id):
    """Update teacher-owned fields for a saved student record."""
    try:
        data = request.get_json() or {}
        records = read_student_records()
        record = next((row for row in records if row.get('record_id') == record_id), None)

        if record is None:
            return jsonify({'success': False, 'error': 'Record not found'}), 404

        if 'teacher_marks' in data:
            teacher_marks = data.get('teacher_marks', '')
            record['teacher_marks'] = float(teacher_marks) if teacher_marks not in (None, '') else ''

        for field in ('teacher_feedback', 'action_status', 'teacher_action', 'action_note'):
            if field in data:
                record[field] = str(data.get(field, '')).strip()

        record['updated_at'] = current_timestamp()
        write_student_records(records)

        return jsonify({'success': True, 'record': normalize_student_record(record)})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/student_records/<record_id>', methods=['DELETE'])
def delete_student_record(record_id):
    """Delete a saved student performance record."""
    try:
        records = read_student_records()
        kept_records = [row for row in records if row.get('record_id') != record_id]

        if len(kept_records) == len(records):
            return jsonify({'success': False, 'error': 'Record not found'}), 404

        write_student_records(kept_records)
        return jsonify({'success': True, 'deleted': record_id})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/download_student_records', methods=['GET'])
def download_student_records():
    """Download the full saved student prediction list as CSV."""
    try:
        records = read_student_records()
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=STUDENT_RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(records)

        filename = f"student_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict for multiple students at once (CSV upload use-case)."""
    try:
        data = request.get_json() or {}
        students = data.get('students', [])
        algorithm = data.get('algorithm', 'random_forest')

        results = []
        for student in students:
            student_data = parse_student_payload(student)
            result = predict_student(student_data, algorithm)
            result['name'] = student.get('name', 'Unknown')
            results.append(result)

        summary = {
            'total': len(results),
            'at_risk': sum(1 for r in results if r['risk_score'] > 50),
            'grade_A': sum(1 for r in results if r['grade'] == 'A'),
            'grade_B': sum(1 for r in results if r['grade'] == 'B'),
            'grade_C': sum(1 for r in results if r['grade'] == 'C'),
            'grade_D': sum(1 for r in results if r['grade'] == 'D'),
            'grade_F': sum(1 for r in results if r['grade'] == 'F'),
        }

        return jsonify({'success': True, 'results': results, 'summary': summary})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print("Student Performance Prediction System")
    print("Visit: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
