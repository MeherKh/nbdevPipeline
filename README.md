# PiPeline
> example of ML pipeline.


This file will become your README and also the index of your documentation.

## Install

`pip install PipelineMLNbdev`

## How to use

Fill me in please! Don't forget code examples:

```python
data = get_data("data/new_maisons-nan.csv")
```

```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>surface</th>
      <th>nb_chambre</th>
      <th>date_creation</th>
      <th>couleur</th>
      <th>prix</th>
      <th>classe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>300.0</td>
      <td>3.0</td>
      <td>10/02/2010</td>
      <td>bleu</td>
      <td>100</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200.0</td>
      <td>2.0</td>
      <td>03/09/2012</td>
      <td>vert</td>
      <td>90</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>250.0</td>
      <td>3.0</td>
      <td>21/08/2011</td>
      <td>bleu</td>
      <td>80</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>280.0</td>
      <td>3.0</td>
      <td>21/08/2010</td>
      <td>bleu</td>
      <td>85</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200.0</td>
      <td>NaN</td>
      <td>01/10/2012</td>
      <td>bleu</td>
      <td>82</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>



```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11 entries, 0 to 10
    Data columns (total 6 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   surface        10 non-null     float64
     1   nb_chambre     9 non-null      float64
     2   date_creation  11 non-null     object 
     3   couleur        11 non-null     object 
     4   prix           11 non-null     int64  
     5   classe         11 non-null     object 
    dtypes: float64(2), int64(1), object(3)
    memory usage: 656.0+ bytes


```python
X = data.iloc[:,:-2]
y = data["classe"]
```

### Example of our TransformeeMaison Class

```python
trsf = TransformeeMaison(dateTo='age')
new_data = trsf.fit(X).transform(X)
new_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>surface</th>
      <th>nb_chambre</th>
      <th>age</th>
      <th>couleur_bleu</th>
      <th>couleur_vert</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>300.0</td>
      <td>3.0</td>
      <td>12</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200.0</td>
      <td>2.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>250.0</td>
      <td>3.0</td>
      <td>11</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>280.0</td>
      <td>3.0</td>
      <td>12</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200.0</td>
      <td>4.0</td>
      <td>10</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Let's apply our Pipeline Pipy

```python
X_train,X_test,y_train,y_test = split_data(X,y,0.2)
```

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

Pipe = Pipeline([
                 ('trsf',TransformeeMaison()),
                 ('ss',StandardScaler()),
                 ('pca',PCA()),
                 ('svm',SVC())
                ])
##les params sont pour le gridsearch pour trouver la meilleur combinaison
Params ={
    'trsf__dateTo':('age','annee'),
    'pca__n_components' : (2,3),
    'svm__kernel':('linear','rbf')
}
```

## initialisation of Pipy Class

```python
p =Pipy(Pipe,Params)
```

## searching best Params

```python
p.gridSearchy(X_train,y_train)
```




    GridSearchCV(cv=3,
                 estimator=Pipeline(steps=[('trsf', TransformeeMaison()),
                                           ('ss', StandardScaler()), ('pca', PCA()),
                                           ('svm', SVC())]),
                 n_jobs=-1,
                 param_grid={'pca__n_components': (2, 3),
                             'svm__kernel': ('linear', 'rbf'),
                             'trsf__dateTo': ('age', 'annee')})



```python
p.gridBestEstimator()
```




    Pipeline(steps=[('trsf', TransformeeMaison(dateTo='age')),
                    ('ss', StandardScaler()), ('pca', PCA(n_components=2)),
                    ('svm', SVC(kernel='linear'))])


