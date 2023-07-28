from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score

X = np.loadtxt("Dataset X", dtype=np.float64)
y = np.loadtxt("Dataset Y", dtype=np.float64)
X = np.log10(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

scaler = StandardScaler()
scaler.fit(x_train)
xt = scaler.transform(x_train)
xte = scaler.transform(x_test)

from collections import Counter
from imblearn.over_sampling import SMOTE
print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(xt, y_train)
print('Resampled dataset shape %s' % Counter(y_res))

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=91,class_weight='balanced',random_state=42,max_features =14,max_depth=11)

classifier.fit(X_res, y_res )

logreg = RandomForestClassifier()
score = cross_val_score(logreg,xt, y_train,cv=5)
print(score)
print (score.mean())

y_pred = classifier.predict(xte)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:", )
print(result1)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:", result2)

分类 = confusion_matrix(y_test, y_pred)

x1= np.loadtxt("Pred X", dtype=np.float64)
y1 = np.loadtxt("Pred Y", dtype=np.float64)
x1= np.log10(x1)

x1 = scaler.transform(x1)
y_pred1 = classifier.predict(x1)
独立验证 = confusion_matrix(y1, y_pred1)
print(classification_report(y1, y_pred1))

score1  = accuracy_score(y1, y_pred1)
print("Accuracy:", score1)

print(y_pred1)