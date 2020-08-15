import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

data_train = pd.read_csv("KDDTrain+.csv")
data_test = pd.read_csv("KDDTest+.csv")

data_test = data_test.drop("service", axis=1)
data_train = data_train.drop("service", axis=1)

y_train = data_train['class']
X_train = data_train.drop('class', axis=1)

y_test = data_test['class']
X_test = data_test.drop('class', axis=1)

y_train = np.where(y_train == "normal", 1, 0)
y_test = np.where(y_test == "normal", 1, 0)

le_protocol = preprocessing.LabelEncoder()  # получение меток
le_protocol.fit(X_train["protocol_type"])  # кодирование меток
X_train["protocol_type"] = le_protocol.transform(X_train["protocol_type"])  # преобразование меток
X_test["protocol_type"] = le_protocol.transform(X_test["protocol_type"])

le_flag = preprocessing.LabelEncoder()
le_flag.fit(X_train["flag"])
X_train["flag"] = le_flag.transform(X_train["flag"])
X_test["flag"] = le_flag.transform(X_test["flag"])

stats_train = data_train.groupby("class").agg("count").duration
print("--------------------Статистика по тренировочной выборке--------------")
print(stats_train)

cat = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=5, loss_function='CrossEntropy',
                         eval_metric='Accuracy',
                         random_state=53)

train_data = Pool(
    data=X_train,
    label=y_train
)

cat.fit(train_data)
cat.save_model("model_binary")

y_pred = cat.predict(X_test)
y_true = y_test
print("---------Точность модели в определении аномалии на тестовых данных----------")
print("Accuracy: ", accuracy_score(y_true, y_pred))
print("Recall: ", recall_score(y_true, y_pred))
print("Precission: ", precision_score(y_true, y_pred))
print("F1: ", f1_score(y_true, y_pred))

y_pred = np.where(y_pred == 1, "normal", "anomaly")
X_test["pred_class"] = y_pred
X_test = X_test[X_test["pred_class"] == "anomaly"].drop("pred_class", axis=1)

train_multi = pd.read_csv("KDDTrain+_meted.csv")
test_multi = pd.read_csv("KDDTest+_meted.csv")

train_multi = train_multi.drop("Unnamed: 42", axis=1)
test_multi = test_multi.drop("Unnamed: 42", axis=1)
train_multi = train_multi.drop("service", axis=1)
test_multi = test_multi.drop("service", axis=1)

train_multi = pd.merge(train_multi, test_multi, how='outer')

train_multi = train_multi[train_multi["class"] != "normal"]

y_multi = train_multi["class"]
X_multi = train_multi.drop("class", axis=1)

le_protocol_multi = preprocessing.LabelEncoder()
le_protocol_multi.fit(X_multi["protocol_type"])
X_multi["protocol_type"] = le_protocol_multi.transform(X_multi["protocol_type"])

le_flag_multi = preprocessing.LabelEncoder()
le_flag_multi.fit(X_multi["flag"])
X_multi["flag"] = le_flag_multi.transform(X_multi["flag"])

le_label_multi = preprocessing.LabelEncoder()
le_label_multi.fit(y_multi)
y_multi = le_label_multi.transform(y_multi)

from sklearn.model_selection import train_test_split

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi,
                                                                            y_multi,
                                                                            test_size=0.1,
                                                                            random_state=42)

cat_multi = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=5, loss_function='MultiClass',
                               eval_metric='Accuracy',
                               random_state=53)

train_data_multi = Pool(
    data=X_train_multi,
    label=y_train_multi
)

cat_multi.fit(train_data_multi)
cat.save_model("model_multi")

y_pred_multi = cat_multi.predict(X_test_multi)
y_true_multi = y_test_multi
print("---------Точность модели в определении типа атаки на валидационных данных----------")
print("Accuracy: ", accuracy_score(y_pred_multi, y_true_multi))

X_test["protocol_type"] = le_protocol.inverse_transform(X_test["protocol_type"])

X_test["flag"] = le_flag.inverse_transform(X_test["flag"])

X_test["protocol_type"] = le_protocol_multi.transform(X_test["protocol_type"])

X_test["flag"] = le_flag_multi.transform(X_test["flag"])

y_test_multi_pred = cat_multi.predict(X_test)

y_test_multi_pred = le_label_multi.inverse_transform(y_test_multi_pred)

y_test_multi_pred = y_test_multi_pred.tolist()

count_pred = Counter(y_test_multi_pred)
print("---------Типы атак и их количество в тестовых данных----------")
print("Всего атак: ", len(y_test_multi_pred))
print("Процент атак в тестовых данных: ", len(y_test_multi_pred) / len(data_test) * 100, )
for key, num in zip(count_pred.keys(), count_pred.values()):
  print(key, num)

