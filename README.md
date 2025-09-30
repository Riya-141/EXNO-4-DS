# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
df1=pd.read_csv("C:\\Users\\admin\\Downloads\\bmi.csv")
df1
```

<img width="1394" height="531" alt="Screenshot 2025-09-30 101642" src="https://github.com/user-attachments/assets/0dfb70dc-cd35-477f-9da4-2748d925c251" />

```
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,Normalizer,RobustScaler
df2=df1.copy()
enc=StandardScaler()
df2[['new_height','new_height']]=enc.fit_transform(df2[['Height','Weight']])
df2
```

<img width="1392" height="598" alt="Screenshot 2025-09-30 101813" src="https://github.com/user-attachments/assets/97adf4b3-6648-44b4-8b4e-c0559e273a98" />

```
df3=df1.copy()
enc=MinMaxScaler()
df3[['new_height','new_height']]=enc.fit_transform(df3[['Height','Weight']])
df3
```

<img width="1345" height="547" alt="Screenshot 2025-09-30 102217" src="https://github.com/user-attachments/assets/e209c8ea-c801-4386-98ca-2b1a747048c4" />

```
df4=df1.copy()
enc=MaxAbsScaler()
df4[['new_height','new_height']]=enc.fit_transform(df4[['Height','Weight']])
df4
```

<img width="1392" height="571" alt="Screenshot 2025-09-30 102308" src="https://github.com/user-attachments/assets/b2b172e0-e617-4997-a795-557dd3745b5c" />

```
df5=df1.copy()
enc=Normalizer()
df5[['new_height','new_height']]=enc.fit_transform(df5[['Height','Weight']])
df5
```

<img width="1401" height="563" alt="Screenshot 2025-09-30 102405" src="https://github.com/user-attachments/assets/03d28369-a843-4132-b94d-c59bf3e159c5" />

```
df6=df1.copy()
enc=RobustScaler()
df6[['new_height','new_height']]=enc.fit_transform(df6[['Height','Weight']])
df6
```

<img width="1400" height="564" alt="Screenshot 2025-09-30 102439" src="https://github.com/user-attachments/assets/2b01c335-46a0-4543-a085-d1e60c3e1c81" />

```
import pandas as pd
df=pd.read_csv("C:\\Users\\admin\\Downloads\\income(1) (1).csv")
df
```

<img width="1261" height="783" alt="Screenshot 2025-09-30 102549" src="https://github.com/user-attachments/assets/9d9b4757-493b-4025-a8f4-3c1397de053a" />

```
from sklearn.preprocessing import LabelEncoder
df_encoded=df.copy()
le=LabelEncoder()

for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col]=le.fit_transform(df_encoded[col])
    
X=df_encoded.drop("SalStat",axis=1)
Y=df_encoded["SalStat"]
```

<img width="1239" height="637" alt="Screenshot 2025-09-30 102900" src="https://github.com/user-attachments/assets/fb7772f0-7933-4188-a78a-1bc4316dceac" />
<img width="1240" height="293" alt="Screenshot 2025-09-30 102935" src="https://github.com/user-attachments/assets/af0f330e-192b-4ba5-aab6-778830ca0beb" />

```
from sklearn.feature_selection import SelectKBest, chi2

chi2_selector = SelectKBest(chi2, k=5)
chi2_selector.fit(X, Y)

selected_features_chi2 = X.columns[chi2_selector.get_support()]
print("Selected features (Chi-Square):", list(selected_features_chi2))

mi_scores = pd.Series(chi2_selector.scores_, index=X.columns)
print(mi_scores.sort_values(ascending=False))
```

<img width="1267" height="513" alt="Screenshot 2025-09-30 103011" src="https://github.com/user-attachments/assets/a4a14da9-641b-47fb-b75d-a114d351a3b7" />

```
from sklearn.feature_selection import f_classif

anova_selector = SelectKBest(f_classif, k=5)
anova_selector.fit(X, Y)

selected_features_anova = X.columns[anova_selector.get_support()]
print("Selected features (ANOVA F-test):", list(selected_features_anova))

mi_scores = pd.Series(anova_selector.scores_, index=X.columns)
print(mi_scores.sort_values(ascending=False))
```

<img width="1263" height="508" alt="Screenshot 2025-09-30 103046" src="https://github.com/user-attachments/assets/a2056b3a-6a45-4660-9f7b-107bf0aec21f" />

```
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import pandas as pd
mi_selector = SelectKBest(mutual_info_classif, k=5)
mi_selector.fit(X, Y)
selected_features_mi = X.columns[mi_selector.get_support()]
print("Selected features (Mutual Info):", list(selected_features_mi))
mi_scores = pd.Series(mi_selector.scores_, index=X.columns)
print(mi_scores.sort_values(ascending=False))
```

<img width="1269" height="668" alt="Screenshot 2025-09-30 103138" src="https://github.com/user-attachments/assets/494305f5-42ea-4c22-8b10-2af8ca89c18e" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression(max_iter=100)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, Y)

selected_features_rfe = X.columns[rfe.support_]
print("Selected features (RFE):", list(selected_features_rfe))
```

<img width="701" height="782" alt="Screenshot 2025-09-30 103234" src="https://github.com/user-attachments/assets/7859c240-aaba-4e86-ac53-5f00511ceedc" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

model = LogisticRegression(max_iter=100)
rfe = SequentialFeatureSelector(model, n_features_to_select=5)
rfe.fit(X, Y)
selected_features_rfe = X.columns[rfe.get_support()]
print("Selected features (SF):", list(selected_features_rfe))
```

<img width="1415" height="625" alt="Screenshot 2025-09-30 103327" src="https://github.com/user-attachments/assets/79a67179-6a88-4143-9040-41a4ce9fd3de" />

```
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X, Y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
selected_features_rf = importances.sort_values(ascending=False).head(5).index
print("Top 5 features (Random Forest Importance):", list(selected_features_rf))
```

<img width="1405" height="245" alt="Screenshot 2025-09-30 103406" src="https://github.com/user-attachments/assets/d099ea96-b8f8-493a-ac35-f0616d9ad6dc" />

```
from sklearn.linear_model import LassoCV
import numpy as np

lasso = LassoCV(cv=5).fit(X, Y)
importance = np.abs(lasso.coef_)

selected_features_lasso = X.columns[importance > 0]
print("Selected features (Lasso):", list(selected_features_lasso))
```

<img width="1395" height="246" alt="Screenshot 2025-09-30 103443" src="https://github.com/user-attachments/assets/e21012bf-07ba-46c1-881f-6f12ad2bf912" />

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("C:\\Users\\admin\\Downloads\\Into to ds\\income(1) (1).csv")
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])
X = df_encoded.drop("SalStat", axis=1)
y = df_encoded["SalStat"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)  # you can tune k
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

<img width="1045" height="883" alt="Screenshot 2025-09-30 103608" src="https://github.com/user-attachments/assets/d0a8f64a-5e3a-4bd2-b65a-b6d00277f252" />

# RESULT:
   Thus, Feature selection and Feature scaling has been used on the given dataset and implemented successfully.
