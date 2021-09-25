import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_scaled_xy(path_train: str, path_test, seed: int) -> tuple:
  with open(path_test) as f:
    data_test = np.loadtxt(f, delimiter="\t", dtype=np.float64)
  with open(path_train) as f:
    data_train = np.loadtxt(f, delimiter="\t", dtype=np.float64)

  data = np.append(data_test, data_train, axis=0)

  train, test = train_test_split(data, train_size=0.7, test_size=0.3, random_state=seed)

  x_test = test[:,1:]
  y_test = test[:,0]
  x_train = train[:,1:]
  y_train = train[:,0]

  scaler = MinMaxScaler()
  scaler.fit(np.append(x_train, x_test, axis=0))

  x_train = scaler.transform(x_train)
  x_test  = scaler.transform(x_test)

  x_train = np.array([x.reshape(-1,1) for x in x_train])
  x_test = np.array([x.reshape(-1,1) for x in x_test])

  return (x_train, y_train), (x_test, y_test)
