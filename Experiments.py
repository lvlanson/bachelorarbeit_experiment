from DTWLVQ          import DTWLVQ
from Helper          import get_scaled_xy
from sklearn.metrics import accuracy_score
from Protocol        import Protocol
import numpy           as np
import dataclasses
import yaml
import os
import sys

# Eingabe von Programmaufruf auslesen
path_train = sys.argv[1]
path_test = sys.argv[2]
if sys.argv[3] == "fastdtw":
  distance_type = DTWLVQ.Distance.fastDTW
  distance_name = DTWLVQ.Distance.fastDTW.name
else:
  distance_type = DTWLVQ.Distance.linmDTW
  distance_name = DTWLVQ.Distance.linmDTW.name
if sys.argv[4] == "lvq1":
  lvq_type = DTWLVQ.Type.LVQ1
  lvq_name = DTWLVQ.Type.LVQ1.name
else:
  lvq_type = DTWLVQ.Type.GLVQ
  lvq_name = DTWLVQ.Type.GLVQ.name
prototype_count = int(sys.argv[5])
epochs          = int(sys.argv[6])
gpu             = True if sys.argv[7] == "1" else False
seed            = int(sys.argv[8])
_set            = int(sys.argv[9])

# Daten laden
(x_train, y_train), (x_test, y_test) = get_scaled_xy(path_train, path_test, 0)

# FastDTW LVQ1
model = DTWLVQ(epochs, 
               prototype_count=prototype_count, 
               dist_func=distance_type, 
               lvq_type=lvq_type, 
               protocol=True,
               random_seed=seed,
               gpu=gpu)
print(f"Training für {path_train} startet")

try:
  model.fit(x_train, y_train)
except Exception as e:
  for p in model._prototypes:
    print(p, "\n")
  print(e.message)
  sys.exit()
# Accuracy
y_pred   = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# Messungen
time_train, time_pred = model.get_runtime()
mem_train, mem_pred   = model.get_mem()
mem_train = mem_train / 1024
mem_pred  = mem_pred / 1024

# Standardabweichungen
data_std       = np.std([len(d) for d in np.append(x_train, x_test, axis=0)])
prototype_std  = np.std([len(d) for d in model.prototypes()], axis=0)

# Durchschnittliche Länge der Vektoren
data_average_len      = sum([len(d) for d in np.append(x_train, x_test, axis=0)]) / len(np.append(x_train, x_test, axis=0))
prototype_average_len = sum([len(d) for d in model.prototypes()]) // len(model.prototypes())

# Größe der Eingabedaten
data_size = len(np.append(x_train, x_test, axis=0))

# Daten zusammentragen
data = {"Accuracy": float(accuracy),
        "Zeit Training (in Sekunden)": time_train,
        "Zeit Vorhersage (in Sekunden)": time_pred,
        "Speicherauslastung Training (in kByte)": mem_train,
        "Speicherauslastung Vorhersage (in kByte)": mem_pred,
        "Datenlaenge Standardabweichung": float(data_std),
        "Prototyplaenge Standardabweichung": float(prototype_std),
        "Durchschnittliche Datenlaenge": data_average_len,
        "Durchschnittliche Prototyplaenge": prototype_average_len,
        "Datensatz Groesse": data_size}

protocol = Protocol(epoch=epochs,
                    prototype_count=prototype_count,
                    dataset=path_train.split(os.sep)[1],
                    seed=seed,
                    dist_func=distance_name,
                    lvq_type=lvq_name,
                    gpu=gpu,
                    data=data)
          
# Protokoll einfügen   
if os.path.exists("protocol.yml"):
  data = None
  with open("protocol.yml", "r") as f:
    data = yaml.safe_load(f)
  try:
    data[_set].append(dataclasses.asdict(protocol))
  except:
    data[_set] = [dataclasses.asdict(protocol)]
  finally:
    with open("protocol.yml", "w+") as f:
      yaml.safe_dump(data, f)
else:
  with open("protocol.yml", "w+") as f:
    yaml.safe_dump({_set: [dataclasses.asdict(protocol)]}, f)