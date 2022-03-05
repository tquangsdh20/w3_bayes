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
dat = pd.read_csv('./datasets/Iris.csv')
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



### Distribution types of specie plot


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
    

## Sử dụng GaussianNB để phân nhóm cho tập Titanic dataset
- dataset: 'Titanic_train.csv', 'Titanic_test.csv'



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

### Read the train and test dataset


```python
X_train = pd.read_csv('./datasets/Titanic_train.csv', index_col=0)
X_test = pd.read_csv('./datasets/Titanic_test.csv')
```

### Explore the dataset


```python
X_train.head()
```




<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test.head()
```




<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



#### The numberic features


```python
X_train.describe()
```




<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



#### The catetorical features


```python
X_train.groupby(['Ticket']).size()
```




    Ticket
    110152         3
    110413         3
    110465         2
    110564         1
    110813         1
                  ..
    W./C. 6608     4
    W./C. 6609     1
    W.E.P. 5734    1
    W/C 14208      1
    WE/P 5735      2
    Length: 681, dtype: int64




```python
X_train.groupby(['Cabin']).size()
```




    Cabin
    A10    1
    A14    1
    A16    1
    A19    1
    A20    1
          ..
    F33    3
    F38    1
    F4     2
    G6     4
    T      1
    Length: 147, dtype: int64




```python
X_train.groupby(['Sex']).size()
```




    Sex
    female    314
    male      577
    dtype: int64




```python
X_train.groupby(['Embarked']).size()
```




    Embarked
    C    168
    Q     77
    S    644
    dtype: int64




```python
print('About the train dataset: ')
print(f'The number of individuals: {X_train.shape[0]}')
print(f'The number of features: {X_train.shape[1]}')
print('\nAbout the test dataset: ')
print(f'The number of individuals: {X_test.shape[0]}')
print(f'The number of features: {X_test.shape[1]}')
```

    About the train dataset: 
    The number of individuals: 891
    The number of features: 11
    
    About the test dataset: 
    The number of individuals: 418
    The number of features: 11
    

### Descrition about dataset

<a href="https://www.kaggle.com/c/titanic/data">https://www.kaggle.com/c/titanic/data</a>

### Pre-Processing Data

#### Catacorial convert to numberic


```python
# Training Dataset
X_train['Sex'] = pd.Categorical(X_train['Sex'])
X_train['Sex'].replace(['female', 'male'],
                        [0, 1],
                       inplace=True
                      )
X_train['Embarked'] = pd.Categorical(X_train['Embarked'])
X_train['Embarked'].replace(['C', 'Q', 'S'],
                        [0, 1, 2],
                       inplace=True
                      )
# Testing Dataset
X_test['Sex'] = pd.Categorical(X_test['Sex'])
X_test['Sex'].replace(['female', 'male'],
                        [0, 1],
                       inplace=True
                      )
X_test['Embarked'] = pd.Categorical(X_test['Embarked'])
X_test['Embarked'].replace(['C', 'Q', 'S'],
                        [0, 1, 2],
                       inplace=True
                      )
```

### Features Selection

The features such as *Cabin*, *Ticket* will be ignored due to not including important information.


```python
features = ['Sex', 'Embarked' ,'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
drops = ['Cabin', 'PassengerId', 'Ticket', 'Name']
```

### Sovling Duplication


```python
### Checking for duplicate values
X_train.index.duplicated().sum()
```




    0



### Dealing with the missing data


```python
## Missing Data Checking:
X_train.isnull().sum()
```




    Survived      0
    Pclass        0
    Name          0
    Sex           0
    Age         177
    SibSp         0
    Parch         0
    Ticket        0
    Fare          0
    Cabin       687
    Embarked      2
    dtype: int64




```python
## Missing Data Checking:
X_test.isnull().sum()
```




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64



#### Drop the unimportant features

In this case, the features **cabin** has too many different values and missing values. Therefore, we can ignore this features. Also we can drop the record has the missing value of the feature **Fare**.


```python
# Drop all the record that has the missing value in Feature EMBARKED and FARE
X_train = X_train.dropna(subset=['Embarked'], how='any')
X_test = X_test.dropna(subset=['Fare'], how='any')
idx_test = X_test['PassengerId']
# Drop the unimportant features
# X_train = X_train.drop(['Cabin', 'Ticket'], axis=1)
# X_test = X_test.drop(['Cabin', 'Ticket'], axis=1)
```

#### Fill the missing value with previous value


```python
# Fill the missing value with previous value in the table
# X_train_pre = X_train.fillna(method='pad')
# X_test_pre = X_test.fillna(method='pad')
# print(f'The train dataset: {X_train_pre.shape[0]}')
# print(f'The test dataset : {X_test_pre.shape[0]}')
```

#### Fill the missing value with mean value

In this case, this method could be a good way to deal with missing values in dataset.


```python
# Fill the missing value with previous value in the table
X_train['Age'] = X_train['Age'].fillna(value=X_train['Age'].mean())
X_test['Age'] = X_test['Age'].fillna(value=X_test['Age'].mean())
# X_test_pre = X_test.fillna(method='pad')
print(f'The train dataset: {X_train.shape[0]}')
# print(f'The test dataset : {X_test_pre.shape[0]}')
```

    The train dataset: 889
    


```python
X_train.isnull().sum()
```




    Survived      0
    Pclass        0
    Name          0
    Sex           0
    Age           0
    SibSp         0
    Parch         0
    Ticket        0
    Fare          0
    Cabin       687
    Embarked      0
    dtype: int64




```python
X_test.isnull().sum()
```




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age              0
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          326
    Embarked         0
    dtype: int64



### Checking if the the dataset is balanced


```python
### Checking if the dataset is balanced.
count_Class=pd.value_counts(X_train["Survived"], sort= True)
print(count_Class)
count_Class.plot(kind= 'bar', color= ["blue", "orange"])
plt.show()
```

    0    549
    1    340
    Name: Survived, dtype: int64
    


<p align="center">
<img src="/.github/output_37_1.png">  
</p> 



```python
count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()
```


<p align="center">
<img src="/.github/output_38_0.png"> 
</p>  


### SMOTE Sampling for unbalanced dataset


```python
from imblearn.over_sampling import SMOTE
y_train = X_train['Survived'].values
# y_test = X_test['Survived'].values --> Label is the different file
# Read the labels of test dataset
y_test = pd.read_csv('./datasets/gender_submission.csv', index_col=0)
y_test = y_test['Survived'][idx_test]
# Test Dataset and Train Dataset features
X_test = X_test[features].values
X_train = X_train[features].values
```


```python
# Transform the dataset
oversample = SMOTE()
X_train, y_train  = oversample.fit_resample(X_train, y_train)
```

### Classification with GaussianNB Model


```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
# fit it to training data
model.fit(X_train,y_train)
model
```


    GaussianNB()



#### Prediction Status


```python
# predict using test data
y_pred = model.predict(X_test)
print(y_pred)
```

    [0 1 0 0 1 0 1 0 1 0 0 0 1 0 1 1 0 0 1 1 1 0 1 1 1 0 1 0 0 0 0 0 1 1 1 0 1
     1 0 0 0 0 0 1 1 0 1 0 1 1 1 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 1 1 0 1 1
     1 1 0 1 0 1 0 1 0 0 0 0 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0
     1 1 1 1 0 0 1 1 1 1 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0
     0 0 1 0 1 0 0 1 1 0 1 1 0 1 0 0 1 1 0 1 1 0 0 0 0 0 1 1 1 1 1 0 1 1 0 1 0
     1 0 0 0 0 0 0 0 0 0 1 1 0 1 1 0 1 1 0 1 1 0 1 0 0 0 0 1 0 0 1 1 1 0 1 0 1
     0 1 1 0 1 0 0 0 1 0 0 1 0 1 0 1 1 1 1 1 0 0 0 1 0 1 1 1 0 1 0 0 0 0 0 1 0
     0 0 1 1 0 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 1 0 1 1 1 0 0 1 0 0 0 1 0 1 0 0 1
     0 0 0 0 0 0 0 1 1 1 0 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 0 1 1 0 0 1 1 0 1
     0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 1 0 0 0 1 0 1 0 0 1 0 1 1 0 1 0 0 1 1 0 0
     1 0 0 1 1 0 0 0 0 0 0 1 1 0 1 0 0 0 0 1 1 0 0 0 1 0 1 0 0 1 0 1 1 1 0 0 1
     1 1 1 1 1 0 1 0 0 0]
    

#### Predicted probabilities


```python
# Compute predicted probabilities: y_pred_prob
y_pred_prob = model.predict_proba(X_test)
print(y_pred_prob)
```

    [[9.08254001e-01 9.17459989e-02]
     [3.37942589e-01 6.62057411e-01]
     [8.18489486e-01 1.81510514e-01]
     ...
     [9.37622532e-01 6.23774682e-02]
     [7.52943049e-01 2.47056951e-01]]
    

### Evaluating


```python
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
print('Confusion matrix:\n', confusion_matrix(y_test,y_pred)) 
print("Accuracy: ", accuracy_score(y_test,y_pred))
```

    Confusion matrix:
     [[228  37]
     [  7 145]]
    Accuracy:  0.894484412470024
    
