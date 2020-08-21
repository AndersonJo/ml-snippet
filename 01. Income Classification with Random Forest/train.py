import pandas as pd
import json
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

COLUMNS = (
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income-level')

CATEGORICAL_COLUMNS = (
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country')

with open('./train.csv', 'r') as f:
    train_data = pd.read_csv(f, header=None, names=COLUMNS)

with open('./test.csv', 'r') as f:
    test_data = pd.read_csv(f, names=COLUMNS, skiprows=1)

x_train = train_data.drop('income-level', axis=1).values
y_train = (train_data['income-level'] == ' >50K').values

x_test = test_data.drop('income-level', axis=1).values
y_test = (test_data['income-level'] == ' >50K.').values

categorical_pipelines = []
for i, col in enumerate(COLUMNS[:-1]):
    if col in CATEGORICAL_COLUMNS:
        # Build the scores array
        scores = [0] * len(COLUMNS[:-1])
        # This column is the categorical column you want to extract.
        scores[i] = 1
        skb = SelectKBest(k=1)
        skb.scores_ = scores
        # Convert the categorical column to a numerical value
        lbn = LabelBinarizer()
        r = skb.transform(x_train)
        lbn.fit(r)
        # Create the pipeline to extract the categorical feature
        categorical_pipelines.append(
            ('categorical-{}'.format(i), Pipeline([
                ('SKB-{}'.format(i), skb),
                ('LBN-{}'.format(i), lbn)])))

# Create pipeline to extract the numerical features
skb = SelectKBest(k=6)
skb.scores_ = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
categorical_pipelines.append(('numerical', skb))
preprocess = FeatureUnion(categorical_pipelines)

clf = RandomForestClassifier()
clf.fit(preprocess.transform(x_train), y_train)

# Create the overall model as a single pipeline
pipeline = Pipeline([
    ('union', preprocess),
    ('classifier', clf)
])
joblib.dump(pipeline, 'model.joblib')

y_pred = pipeline.predict(x_test)
print(classification_report(y_test, y_pred))
