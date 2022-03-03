# Bài tập Naive Bayes

## Sử dụng GaussianNB để phân nhóm cho tập Iris dataset
- dataset: 'Iris.csv'



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline 
```


```python
dat = pd.read_csv('Iris.csv')
dat.head()
```




<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



## Distribution types of specie plot


```python
count_Class=pd.value_counts(dat["Species"], sort= True)
print(count_Class)
count_Class.plot(kind= 'bar', color= ["blue", "orange", "green"])
# plt.title('Bar chart')
plt.show()
```

    Iris-setosa        50
    Iris-versicolor    50
    Iris-virginica     50
    Name: Species, dtype: int64
    


<p align="center">
<img src="/.github/output_5.png">
</p>



```python
count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()
```


<p align="center">
<img src="/.github/output_6_0.png">
</p>



```python
## Missing Data Checking:
pd.isnull(dat).any()
```




    Id               False
    SepalLengthCm    False
    SepalWidthCm     False
    PetalLengthCm    False
    PetalWidthCm     False
    Species          False
    dtype: bool



The result show that has no missing values and the dataset is balaced.


```python
#Chuẩn hóa dữ liệu cột Amount, thêm cột scaled_Amount
from sklearn.preprocessing import StandardScaler
fetures = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
dat_scale = pd.DataFrame(StandardScaler().fit_transform(dat[fetures].values))
```


```python
dat_scale.rename(columns={0: "SepalLengthCm", 1: "SepalWidthCm", 2: "PetalLengthCm", 3: "PetalWidthCm"})
```




<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.900681</td>
      <td>1.032057</td>
      <td>-1.341272</td>
      <td>-1.312977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.143017</td>
      <td>-0.124958</td>
      <td>-1.341272</td>
      <td>-1.312977</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.385353</td>
      <td>0.337848</td>
      <td>-1.398138</td>
      <td>-1.312977</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.506521</td>
      <td>0.106445</td>
      <td>-1.284407</td>
      <td>-1.312977</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.021849</td>
      <td>1.263460</td>
      <td>-1.341272</td>
      <td>-1.312977</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>1.038005</td>
      <td>-0.124958</td>
      <td>0.819624</td>
      <td>1.447956</td>
    </tr>
    <tr>
      <th>146</th>
      <td>0.553333</td>
      <td>-1.281972</td>
      <td>0.705893</td>
      <td>0.922064</td>
    </tr>
    <tr>
      <th>147</th>
      <td>0.795669</td>
      <td>-0.124958</td>
      <td>0.819624</td>
      <td>1.053537</td>
    </tr>
    <tr>
      <th>148</th>
      <td>0.432165</td>
      <td>0.800654</td>
      <td>0.933356</td>
      <td>1.447956</td>
    </tr>
    <tr>
      <th>149</th>
      <td>0.068662</td>
      <td>-0.124958</td>
      <td>0.762759</td>
      <td>0.790591</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 4 columns</p>
</div>




```python
#test train split time
from sklearn.model_selection import train_test_split
labels = dat['Species'].values # target
features = dat_scale.values # features
X_train, X_test, y_train, y_test = train_test_split(dat_scale, labels, test_size=0.2,
                                            random_state=1, stratify=labels)

print("Train Dataset: ", len(y_train), "\nTest Dataset :  ", len(y_test))
```

    Train Dataset:  120 
    Test Dataset :   30
    

###  Classification with GaussianNB Model


```python
## 
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
# fit it to training data
model.fit(X_train,y_train)
model
```




    GaussianNB()



#### Prediction


```python
# predict using test data
y_pred = model.predict(X_test)
print(y_pred)
```

    ['Iris-virginica' 'Iris-setosa' 'Iris-versicolor' 'Iris-setosa'
     'Iris-setosa' 'Iris-setosa' 'Iris-virginica' 'Iris-virginica'
     'Iris-virginica' 'Iris-versicolor' 'Iris-setosa' 'Iris-versicolor'
     'Iris-virginica' 'Iris-versicolor' 'Iris-virginica' 'Iris-setosa'
     'Iris-virginica' 'Iris-versicolor' 'Iris-versicolor' 'Iris-virginica'
     'Iris-versicolor' 'Iris-versicolor' 'Iris-setosa' 'Iris-setosa'
     'Iris-virginica' 'Iris-versicolor' 'Iris-setosa' 'Iris-setosa'
     'Iris-versicolor' 'Iris-versicolor']
    


```python
# Compute predicted probabilities: y_pred_prob
y_pred_prob = model.predict_proba(X_test)
print(y_pred_prob)
```

    [[3.15923295e-233 2.88633615e-007 9.99999711e-001]
     [1.00000000e+000 6.36418205e-020 1.46133651e-025]
     [1.10742309e-035 9.99999815e-001 1.84533301e-007]
     [1.00000000e+000 2.85960633e-019 5.08807131e-025]
     [1.00000000e+000 4.08487522e-011 1.37744907e-017]
     [1.00000000e+000 1.69003592e-020 6.31398590e-026]
     [3.56794911e-228 1.96023993e-007 9.99999804e-001]
     [3.44998577e-181 3.03181699e-003 9.96968183e-001]
     [1.93865592e-168 2.91504094e-002 9.70849591e-001]
     [5.87125061e-118 9.14394811e-001 8.56051886e-002]
     [1.00000000e+000 1.38559777e-018 8.43339205e-024]
     [5.36119723e-077 9.99982471e-001 1.75289169e-005]
     [2.48549577e-214 3.44922766e-006 9.99996551e-001]
     [7.40347328e-062 9.99992692e-001 7.30762478e-006]
     [8.39681578e-174 2.30014924e-002 9.76998508e-001]
     [1.00000000e+000 1.80932901e-018 1.10302647e-022]
     [2.50579885e-238 6.64231013e-008 9.99999934e-001]
     [5.75184031e-078 9.99875802e-001 1.24197848e-004]
     [8.84632729e-038 9.99999826e-001 1.74397898e-007]
     [1.73359791e-150 1.68582429e-001 8.31417571e-001]
     [1.59084399e-089 9.98828268e-001 1.17173212e-003]
     [1.13851113e-107 9.95801293e-001 4.19870749e-003]
     [1.00000000e+000 6.36418205e-020 1.46133651e-025]
     [1.00000000e+000 8.37027828e-020 4.29587932e-026]
     [1.55380263e-186 4.14969025e-004 9.99585031e-001]
     [1.06826351e-122 9.92054053e-001 7.94594699e-003]
     [1.00000000e+000 9.71175997e-016 4.10896853e-021]
     [1.00000000e+000 1.76286416e-020 3.28412528e-026]
     [5.95445700e-125 7.35965804e-001 2.64034196e-001]
     [2.46043743e-115 9.65759813e-001 3.42401875e-002]]
    

### Evaluation


```python
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
print('Confusion matrix:\n', confusion_matrix(y_test,y_pred)) 
print("Accuracy: ", accuracy_score(y_test,y_pred))
```

    Confusion matrix:
     [[10  0  0]
     [ 0 10  0]
     [ 0  1  9]]
    Accuracy:  0.9666666666666667
    

