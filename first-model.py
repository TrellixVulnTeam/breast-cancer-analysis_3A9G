import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("cancer-data.csv")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

X = dataset.iloc[:, 1:31].values
Y = dataset.iloc[:, 31].values

dataset.columns = ["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean"
    ,"concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se",
                   "smoothness_se","compactness_se","concavity_se",'concave_points_se',"symmetry_se","fractal_dimension_se",
                   'radius_worst',"texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
                   "concave_points_worst","symmetry_worst","fractal_dimension_worst"]

dataframe = pd.DataFrame(Y)

#Transformacion de valores categoricos(B y M) a valores numericos
# para mejor interpretacion por parte del modelo.
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Entrenamiento del modelo.
# Entrenamiento: 60% de los casos.
# Pruba: 40% de los casos.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.40, train_size=0.60, random_state = 0)

#Normalizacion de datos
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Aplicacion de regresion logistica al conjunto de datos
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Prediccion de los resultados de la prueba
Y_pred = classifier.predict(X_test)

#Creacion de matriz de confusion
cm = confusion_matrix(Y_test, Y_pred)
correct_predictions = cm[0, 0] + cm[1, 1]
total_number_predictions = cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]

accuracy = correct_predictions / total_number_predictions
print(f'Verdadero positivo: {cm[0,0]}')
print(f'Falso positivo: {cm[0,1]}')
print(f'Falso negativo: {cm[1, 0]}')
print(f'Verdadero negativo: {cm[1, 1]}')
print(f'Precision del modelo: {accuracy*100} %')