from random import randint, seed
from enum import Enum
from linmdtw import linmdtw, get_path_cost
from fastdtw import fastdtw
from typing import Tuple
from dataclasses import dataclass
import tracemalloc
import numpy as np
import time

@dataclass
class Minimum:
  """
  Speichert Daten zum nächsten gefunden Prototypen
  Attributes:
    index: int          Index des Prototypen
    path: nd.array      Pfad des Prototypen zum verglichenen x
    distance: float     Distanz zwischen Prototyp und verglichenem x
    class_val: float    Enthält Klasse
  """
  index: int
  path: np.ndarray
  distance: float
  class_val: float

  def update(self, index: int, path: np.ndarray, distance: float, class_val:float):
    """
    Updated die Werte für ein gefundenes Minimum
    Args:
      index: int            Index des Prototypen
      path: nd.array        Pfad des Prototypen zum verglichenen x 
      distance: float       Distanz zwischen Prototyp und verglichenem x
      class_val: float      Enthält Klasse
    """
    self.index = index
    self.path = path
    self.distance = distance
    self.class_val = class_val

class DTWLVQ:
  """
  Diese Klasse stellt ein Modell für LVQ1 und GLVQ mit den Distanzfunktionen linmDTW und fastDTW im
  DTW-Space zur Verfügung. GLVQ wird mit der Sigmoid Funktion verwendet.
  Attributes:
    _epochs: int                  Enthält die Anzahl der durchzuführenden Epochen
    _learning_rate: float         Enthält die aktuelle Lernrate
    _X: np.ndarray                Enthält die Trainingsdaten
    _y: np.ndarray                Enthält die zu den Trainingsdaten korrespondierenden Klassen
    _prototypes: list             Enthält die zufällig ausgewählten Prototypen als Liste, wobei
                                  jedes Element ein Tupel mit den Vektordaten und der Klasse ist
                                  (v, c)
    _decrease_learning: float     Hält den Betrag für die monotone Senkung der Lernrate fest
    _prototype_count: int         Enthält wieviele Prototypen je Klasse initialisiert werden
    _time_train: dict             Laufzeiten und Speicherauslastung für Training werden dokumentiert
    _time_pred: dict              Laufzeiten und Speicherauslastung für Testen werden dokumentiert
    _mem_train: int               Maximale Speicherauslastung vom Training
    _mem_pred: int                Maximale Speicherauslastung der Vorhersage
    _do_protocol: boolean         Gibt an, ob Zeiten gemessen werden sollen
  """
  def __init__(self, 
               epochs: int, 
               learning_rate:float = 0.1, 
               prototype_count:int = 1,
               random_seed:int = 1,
               dist_func:int = 0,
               lvq_type:int = 0,
               gpu:bool = True,
               radius:int = 1,
               protocol:bool = False):
    """
    Args:
      epochs: int             Anzahl der durchzuführenden Epochen
      learning_rate: float    Lernrate
      prototype_count: int    Anzahl der Prototypen je Klasse
      random_seed: int        Wert zur Reproduzierbarkeit des Zufalls
      dist_func: Distance     Konfiguriert Distanzfunktion
      lvq_type: Type          Konfiguriert LVQ Typ
      gpu: boolean            Konfiguriert Einsatz der GPU für Linear Memory DTW
      radius: int             Konfiguriert Radius für FastDTW
      protocol: boolean       Konfiguriert Protokollierung der Laufzeit und dem Speicher der Klasse
    """
    self._epochs = epochs
    self._learning_rate = learning_rate
    self._X = None
    self._y = None

    self._prototypes        = []
    self._decrease_learning = self._learning_rate/self._epochs
    self._prototype_count   = prototype_count
    
    self._dist_func   = None
    self._update_func = None

    self._time_train  = 0
    self._time_pred   = 0
    self._mem_train   = 0
    self._mem_pred    = 0
    self._do_protocol = protocol
    
    # Distanzfunktion je nach DTW Typ einrichten
    if not dist_func or dist_func == self.Distance.linmDTW:

      def do_linmdtw(m,x) -> tuple:
        """
        Funktion um linear Memory DTW auszuführen
        """
        path = linmdtw(m,x,do_gpu=gpu)
        cost = get_path_cost(m,x,path)
        return cost, path

      self._dist_func = do_linmdtw

    elif dist_func == self.Distance.fastDTW:
      def do_fastdtw(m,x) -> tuple:
        """
        Funktion um fastDTW auszuführen
        """
        cost, path = fastdtw(m,x,radius=radius)
        return cost, path

      self._dist_func = do_fastdtw

    # Update Funktion je nach LVQ Typ einrichten
    if not lvq_type or lvq_type == self.Type.LVQ1:
      self._update = self.__update_lvq1
      
    elif lvq_type == self.Type.GLVQ:
      self._update = self.__update_glvq

    seed(random_seed) # Zufallsgenerator seeden

  def fit(self, X, y) -> None:
    """
    Führt den Trainingsprozess durch
    
    Args:
      X: np.ndarray - Trainingsdaten
      y: np.ndarray - Klassen
    
    Returns:
      None
    """
    
    if self._do_protocol:
      tracemalloc.start()
      start_time = time.perf_counter()
    self._X = X
    self._y = y
    self._set_prototypes()
    self._train()
    if self._do_protocol:
      self._time_train = time.perf_counter() - start_time
      _, self._mem_train = tracemalloc.get_traced_memory()
      tracemalloc.stop()

    # Speicher für Vorhersagemessung aufräumen
    del self._X, self._y, self._epochs, self._learning_rate, self._decrease_learning 
    del self._prototype_count
    
  def _set_prototypes(self) -> None:
    """
    Initiiert die Prototypen. Je Klasse werden self._prototype_count Prototypen initialisiert.
    Diese werden, wie in Kapitel 4.1 beschrieben, der Einfachheit halber aus der Menge der Trainingsdaten
    zufällig entnommen. Dabei wird auch geprüft ob genügend Elemente je Klasse für den angegebenen
    prototype_count vorhanden sind.

    Returns:
      None
    """

    # Test der Prototypen auslassen für Messung Zeit und Arbeitsspeicher
    if not self._do_protocol:
      # Es müssen genauso viele Prototypen wie verschiedene Elemente verschiedener Klassen vorhanden sein
      if not self._check_prototype_count:
        raise Exception("Nicht genug Elemente der Klassen für prototype_count vorhanden.")
      
    classes = set(self._y)                                  # Alle verschiedenen Klassen als Menge
    for c in classes:
      prototype_counter = 0                                 # Zählt mit wieviele Prototypen hinzugefügt werden
      while prototype_counter < self._prototype_count:      # Solange weniger Prototypen gewählt wurden, als festgelegt wurden
        i = randint(0, len(self._y)-1)                      # Zufälligen Index wählen
        if self._y[i] == c:                                 # Klassenzugehörigkeit des zufällig ausgewählten Prototypen überprüfen
          self._prototypes.append([self._X[i], c])          # Prototyp hinzufügen
          prototype_counter += 1                            # Kennzeichnen, dass ein Prototyp gefunden wurde

  def _check_prototype_count(self) -> None:
    """
    Prüft ob genügend Elemente je Klasse für den angegebenen prototype_count vorhanden sind.
    Returns:
      None
    """
    class_map = {i: 0 for i in range(len(set(self._y)))}    # Legt Dictionary für Zählen der Prototypen für jede Klasse an
    for y in self._y:                                       # Geht alle Prototypen durch
      class_map[y] += 1                                     # Fügt für jede Klasse einen Prototypen hinzu
    
    return all(count > self._prototype_count for count in list(class_map.values()))   # Wenn alle Prototypen häufiger als prototype_count vorhanden sind => True
      
  def _get_distance(self, m, x) -> tuple:
    """
    Berechnet den Pfad und die Kosten für m und x.
    Args:
      x: np.ndarray   Enthält Zeitreihensignal der Trainingsdaten
      m: np.ndarray   Enthält Zeitreihensignal des Prototypen
    Returns:
      None
    """
    cost, path = self._dist_func(m, x)
    return path, cost

  def _find_nearest_class(self, x: np.ndarray) -> float:
    """
    Findet die Klasse des nächsten Prototypen zu x
    Args:
      x: np.ndarray   Enthält Zeitreihensignal des Eingabevektors
    """
    _, cost  = self._get_distance(self._prototypes[0][0], x)
    smallest = (self._prototypes[0][1], cost)
    for prototype in self._prototypes[1:]:
      m           = prototype[0]
      class_value = prototype[1]
      _, temp_cost = self._get_distance(m, x)
      if temp_cost < smallest[1]:
        smallest = (class_value, temp_cost)
    return smallest[0]
       
  def _train(self) -> None:
    """
    Iteriert über Anzahl der Epochen. Wählt ein zufälliges x aus und sucht den nähesten Prototypen
    """
    for _ in range(self._epochs):
      index_x = randint(0, len(self._X)-1)
      self._update(index_x)
      self._learning_rate -= self._decrease_learning
  
  def _get_minimum(self, x) -> Minimum:
    """    
    Sucht den nächsten Prototypen zu Trainingsvektor x. Für LVQ1

    Args:
      x: np.ndarray   Trainingsvektor aus den Trainingsdaten

    Returns:
      Minimum         Enthält Daten zum nächsten Prototypen
    """
    prototypes = [m[0] for m in self._prototypes]
    classes    = [m[1] for m in self._prototypes]
   
    path, dist = self._get_distance(prototypes[0], x)
    minimum = Minimum(0, path, dist, classes[0])
    for i, m in enumerate(prototypes[1:], start=1):
      path, dist = self._get_distance(m, x)
      if minimum.distance > dist:
        minimum.update(i, path, dist, classes[i])
    return minimum

  def _get_minima(self, x, class_val) -> Tuple[Minimum, Minimum]:
    """
    Such die nächsten Prototypen derselben und einer anderen Klasse zu
    Trainingsvektor x. Für GLVQ

    Args:
      x: np.ndarray               Trainingsvektor aus den Trainingsdaten
      class_val: int              Klasse des Trainingsvektors

    Returns:
      Tuple(Minimum, Minimum)     Enhält Daten zu den nächsten Trainigsvektoren
                                  (d^+, d^-) 
    """

    prototypes = [m[0] for m in self._prototypes]
    classes    = [m[1] for m in self._prototypes]
   
    path, dist    = self._get_distance(prototypes[0], x)
    minimum_plus  = Minimum(0, path, dist, classes[0])    # d^+
    minimum_minus = Minimum(0, path, dist, classes[0])    # d^-

    for i, m in enumerate(prototypes[1:], start=1):
      # i ist um 1 versetzt durch    ^
      path, dist = self._get_distance(m, x)
      if class_val == classes[i] and minimum_plus.distance > dist:
        minimum_plus.update(i, path, dist, classes[i])
      elif class_val != classes[i] and minimum_minus.distance > dist:
        minimum_minus.update(i, path, dist, classes[i])
    return (minimum_plus, minimum_minus)
    
  def _gradient(self, m, x, path) -> np.array:
    """
    Berechnet den Gradienten aus Formel 4.11.

    Args:
      x: np.ndarray     Enthält Zeitreihensignal der Trainingsdaten
      m: np.ndarray     Enthält Zeitreihensignal des Prototypen
      path: np.ndarray  Enthält Warpingpfad

    Returns:
      Gradient: np.ndarray
    """
    
    ux = np.zeros(shape=(path[-1][0]+1, 1), dtype=np.single)
    vm = np.zeros(shape=(path[-1][0]+1, 1), dtype=np.single)

    for p in path:
      ux[p[0]] += x[p[1]]
      vm[p[0]] += 1
    for i, e in enumerate(vm):
      vm[i] = e*m[i]
    return 2*(vm - ux) 

  def predict(self, X: np.ndarray) -> list:
    """
    Sammelt die Vorhersagen in predictions und gibt diese auf Basis der trainierten Prototypen zurück

    Args:
      X: np.ndarray       Menge an Zeitreihendaten, für die eine Vorhersage getroffen werden soll
    
    returns:
      predictions: list   Vorhersagen für X in Form einer Liste
    """
    if self._do_protocol:
      tracemalloc.reset_peak()
      tracemalloc.start()
      start_time = time.perf_counter()
    predictions = []
    for x in X:
      predictions.append(self._find_nearest_class(x))
    if self._do_protocol:
      self._time_pred = time.perf_counter() - start_time
      _, self._mem_pred = tracemalloc.get_traced_memory()
      tracemalloc.stop()
    return predictions
  
  def _update(self, index_x):
    pass

  def __update_lvq1(self, index_x: int) -> None:
    """
    Aktualisierungsregel für LVQ1

    Args:
      index_x: int        Index der zufällig ausgewählten Zeitreihe

    returns:
      None
    """
    minimum  = self._get_minimum(self._X[index_x])                                                    # Kleinsten Prototypen finden, Daten in Objekt Minimum
    gradient = self._gradient(self._prototypes[minimum.index][0], self._X[index_x], minimum.path)     # Gradient berechnen
    if self._y[index_x] == minimum.class_val:                                                         # Wenn Klasse von x und m gleich sind
      self._prototypes[minimum.index][0] = self._prototypes[minimum.index][0] - \
                                           self._learning_rate *                \
                                           gradient                                                   # Prototyp aktualisieren
    else:                                                                                             # Wenn Klasse von x und m unterschiedlich sind
      self._prototypes[minimum.index][0] = self._prototypes[minimum.index][0] + \
                                           self._learning_rate *                \
                                           gradient                                                   # Prototyp aktualisieren

  def __update_glvq(self, index_x) -> None:
    """
    Aktualisierungsregel für GLVQ

    Args:
      index_x: int        Index der zufällig ausgewählten Zeitreihe

    returns:
      None
    """
    minimum_plus, minimum_minus = self._get_minima(self._X[index_x], self._y[index_x])                    # minimum für d^+ und d^- finden
    gradient_plus = self._gradient(self._prototypes[minimum_plus.index][0],                               # Gradienten für m^+ berechnen 
                                   self._X[index_x], 
                                   minimum_plus.path)
    gradient_minus = self._gradient(self._prototypes[minimum_minus.index][0],                             # Gradienten für m^- berechnen
                                    self._X[index_x], 
                                    minimum_minus.path)

    mu_plus   = 4 * minimum_plus.distance / (minimum_plus.distance + minimum_minus.distance + 0.001)**2   # Mu für m^+
    mu_minus  = 4 * minimum_minus.distance / (minimum_plus.distance + minimum_minus.distance + 0.001)**2  # Mu für m^-
    exp_plus  = (1/(1+np.exp(-mu_plus/2))) * (1 - 1/(1+np.exp(-mu_plus/2)))                               # Ableitung der Sigmoid Funktion für Mu^+ berechnen
    exp_minus = (1/(1+np.exp(-mu_minus/2))) * (1 - 1/(1+np.exp(-mu_minus/2)))                             # Ableitung der Sigmoid Funktion für Mu^- berechnen
    self._prototypes[minimum_plus.index][0] = self._prototypes[minimum_plus.index][0] - \
                                              self._learning_rate *                     \
                                              exp_plus *                                \
                                              gradient_plus                                               # m^+ Prototypen aktualisieren

    self._prototypes[minimum_minus.index][0] = self._prototypes[minimum_minus.index][0] + \
                                               self._learning_rate *                      \
                                               exp_minus *                                \
                                               gradient_minus                                             # m^- Prototypen aktualisieren

  def get_runtime(self) -> tuple:
    """
    Gibt Laufzeiten für Training und Vorhersage zurück
    returns:
      times: tuple    Tuple mit (zeit_training, zeit_prediction)
    """
    return (self._time_train, self._time_pred)

  def get_mem(self) -> tuple:
    """
    Gibt Speicherauslastung für Training und Vorhersage zurück
    returns:
      memory: tuple   Tuple mit (speicher_training, speicher_prediction)
    """
    return (self._mem_train, self._mem_pred)

  def prototypes(self) -> np.ndarray:
    """
    Gibt Prototypen zurück
    returns:
      prototypes: np.ndarray  Numpy Array mit Prototypen
    """
    prototypes = [p[0] for p in self._prototypes]
    
    return np.array(prototypes, dtype=np.object)

  class Distance(Enum):
    """ 
    Konfigurationsklasse für die Distanzfunktion
    """
    linmDTW = 1
    fastDTW = 2

  class Type(Enum):
    """ 
    Konfigurationsklasse für die Art der LVQ Variante
    """
    GLVQ = 1
    LVQ1 = 2
