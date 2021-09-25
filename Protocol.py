from dataclasses import dataclass
import dataclasses


@dataclass
class Protocol:
  epoch: int
  prototype_count: int
  dataset: str
  seed: int
  dist_func: str
  lvq_type: str
  data: dict
  gpu: bool