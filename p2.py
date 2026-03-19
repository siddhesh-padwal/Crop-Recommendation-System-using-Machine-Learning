import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df=pd.read_csv("crop_recommendation.csv")
le=LabelEncoder()
df['crop']=le.fit_transform(df['crop'])
features=['N','P','K','temperature','humidity','ph','rainfall']
scalar=StandardScaler()
df_scaled=df.copy()
df_scaled[features]=scalar.fit_transform(df_scaled[features])

x=df_scaled[features]
y=df_scaled['crop']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('classification report for random forest:')
print(classification_report(y_test,y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', mean_squared_error(y_test,y_pred))
print('R2 Score:', r2_score(y_test,y_pred))

conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues",xticklabels=le.classes_,yticklabels=(le.classes_))
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("confusion matrix for random forest")
plt.tight_layout()
plt.show()

#features importance checking
importances=model.feature_importances_
print("feature importances for random forest:")
print(importances*100)
plt.figure(figsize=(8,6))
sns.barplot(x=importances,y=features)
plt.title("feature importance for random forest")
plt.xlabel("importance")
plt.ylabel("features")
plt.tight_layout()
plt.show()

