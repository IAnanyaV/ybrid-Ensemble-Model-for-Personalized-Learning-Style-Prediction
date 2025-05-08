import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# -------- Step 1: Load & Prepare Data --------
df = pd.read_csv('scaled_X_features.csv')  # Ensure the file exists

X = df.drop('learning_style', axis=1)
y = df['learning_style']

# Handle class imbalance using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset into training & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

# -------- Step 2: Scale Features --------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------- Step 3: Define Hybrid Model --------
rf_classifier = RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=0)
svm_classifier = SVC(C=100, gamma=1, kernel='linear', probability=True)  # Enable probability output
dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=0)  # J48 Equivalent
nb_classifier = GaussianNB()  # Naïve Bayes Classifier

# Ensemble Hybrid Model using Voting
hybrid_classifier = VotingClassifier(
    estimators=[
        ('RF', rf_classifier),
        ('SVM', svm_classifier),
        ('DT', dt_classifier),
        ('NB', nb_classifier)
    ],
    voting='soft'  # Use probability-based voting for better decision making
)

# -------- Step 4: Train Hybrid Model --------
hybrid_classifier.fit(X_train_scaled, y_train)

# -------- Step 5: Make Predictions & Evaluate --------
y_pred = hybrid_classifier.predict(X_test_scaled)

# Accuracy & F1 Score
accuracy = accuracy_score(y_test, y_pred)
f1_en = f1_score(y_test, y_pred, average="weighted")

print("Hybrid Model Accuracy:", accuracy)
print("F1 Score:", f1_en)

# Confusion Matrix Display
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=hybrid_classifier.classes_)
disp.plot()

# Cross-validation
cv_scores = cross_val_score(hybrid_classifier, X_train_scaled, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean accuracy: {:.2f} %".format(cv_scores.mean() * 100))
print("Standard Deviation: {:.2f} %".format(cv_scores.std() * 100))

# Classification report
print(classification_report(y_test, y_pred, zero_division=0))

# -------- Step 6: Save Model & Scaler --------
joblib.dump(hybrid_classifier, 'hybrid_model.pkl')
joblib.dump(scaler, 'hybrid_scaler.pkl')
print("Hybrid Model & Scaler saved successfully.")

# -------- Step 7: Load Model for User Input --------
def predict_learning_style():
    """Predict learning style from user input"""
    try:
        classifier = joblib.load('hybrid_model.pkl')
        scaler = joblib.load('hybrid_scaler.pkl')

        feature_names = [
            'Amount of time spent by learners interacting with images',
            'Amount of time spent on video related materials',
            'Amount of time spent on text-based material',
            'Amount of time spent on audio-related materials',
            'Complexity or depth of the learning material',
            'Frequency of PowerPoint usage',
            'Concrete contents',
            'Performance or achievement of the learner',
            'Number of correctly answered standard questions',
            'Number of messages or posts posted by the learner',
            'Time or duration spent by the learner in solving exercises',
            'Number of group discussions',
            'Number of lessons of learning objectives skipped',
            'Number of times the learner utilized the Next button',
            'Amount of Time Spent in sessions',
            'Number of questions on topics',
            'Number of questions or queries posed by the learner'
        ]

        # Take user input for features
        input_features = [float(input(f"Enter value for {feature}: ")) for feature in feature_names]

        # Convert & scale input features
        input_features = np.array(input_features).reshape(1, -1)
        scaled_input_features = scaler.transform(input_features)

        # Predict learning style
        target_label = classifier.predict(scaled_input_features)

        label_mapping = {0: 'Processing', 1: 'Understanding', 2: 'Input', 3: 'Perception'}
        predicted_label = label_mapping[target_label[0]]

        print("Predicted Learning Style:", predicted_label)
    
    except Exception as e:
        print("Error:", str(e))

# Call function if needed: predict_learning_style()