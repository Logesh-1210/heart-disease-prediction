import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
# %matplotlib inline

# Filter the uneccesary warnings
import warnings
warnings.filterwarnings("ignore")

df1 = pd.read_csv('heart_disease.csv')
df = pd.DataFrame()
df['age'] = df1['age']  # Assuming 'url' is the name of the column in the original dataset
df['sex'] = df1['sex']
df['cp'] = df1['cp']
df['trestbps'] = df1['trestbps']
df['chol'] = df1['chol']
df['fbs'] = df1['fbs']
df['restecg'] = df1['restecg']
df['thalach'] = df1['thalach']
df['exang'] = df1['exang']
df['oldpeak'] = df1['oldpeak']
df['slope'] = df1['slope']
df['ca'] = df1['ca']
df['thal'] = df1['thal']
df['target']=df1['target']

df['target'] = df['target'].map({0:0, 1:1})
df['target'].unique()
#to check null values in the dataframe
df.isnull()

from sklearn.model_selection import train_test_split
X=df.drop("target",axis=1).values
y=df["target"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.ensemble import RandomForestClassifier
error= []
# Will take some time
for i in range(550,600):
    rfc = RandomForestClassifier(n_estimators=i)
    rfc.fit(X_train,y_train)
    pred_i = rfc.predict(X_test)
    error.append(np.mean(pred_i != y_test))

rfc = RandomForestClassifier(n_estimators=571)
rfc.fit(X_train,y_train)


pickle.dump(rfc,open('heart_model.pkl','wb'))
model=pickle.load(open('heart_model.pkl','rb'))
print("Heart model created")