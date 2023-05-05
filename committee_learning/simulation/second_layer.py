import torch
import numpy as np

from .autograd import CommiteeMachine
from .base import BaseSimulation

from .._cython.risk import two_layer_square_risk
from .._config.python import scalar_type

class TwoLayerNeuralNetwork(CommiteeMachine):
  def __init__(self, input_size, hidden_size, W, a, activation, teacher = False):
    super().__init__(input_size, hidden_size, W, activation, teacher)

    self.second_layer = torch.nn.Linear(hidden_size, 1, bias=False)

    if teacher:
      with torch.no_grad():
        self.second_layer.weight = torch.nn.Parameter(torch.tensor(a).float())
    else:
      self.second_layer.weight = torch.nn.Parameter(torch.tensor(a).float())
  
  def forward(self, x):
    return 1/self.hidden_size * self.second_layer(
      self.activation(
        self.layer(x)/np.sqrt(self.input_size)
      )
    )
  
  @torch.no_grad()
  def get_sl_vector(self):
    return self.second_layer.weight.numpy()
  
    
class TwoLayerSimulation(BaseSimulation):
  def __init__(self, d, p, k, gamma, Wt, at, W0, a0, noise = 0., activation = 'square', spherical_weights=True, normalized_second_layer = True, seed = None, disable_QM_save = False, extra_metrics = {}):
    super().__init__(d, p, k, gamma, Wt, W0, noise, activation, seed, disable_QM_save, extra_metrics)
    self.teacher = TwoLayerNeuralNetwork(d, k, activation=activation, W=Wt, a=at, teacher=True)
    self.student = TwoLayerNeuralNetwork(d, p, activation=activation, W=W0, a=a0)

    self.at = at
    self.saved_as = list()

    if activation == 'square':
      self.theoretical_risk = two_layer_square_risk
    else: NotImplementedError

    self.spherical_weights = spherical_weights
    self.normalized_second_layer = normalized_second_layer

  def _gradient_descent_step(self, y_student, y_teacher_noised, x):
    loss = self.loss(y_student, y_teacher_noised)
    self.student.zero_grad()
    loss.backward()
    W, a = tuple(self.student.parameters())

    # First Layer
    W.data.sub_(W.grad.data * self.gamma)
    if self.spherical_weights:
      W.data = np.sqrt(self.d) * torch.nn.functional.normalize(W.data)

    # Second Layer
    a.data.sub_(self.gamma * a.grad.data)
    if self.normalized_second_layer:
      a.data = self.p * a.data / torch.sum(a.data)

  def _save_step(self, step):
    self.saved_steps.append(step)
    Ws = self.student.get_weight()

    M = (Ws @ self.Wt.T/self.d).astype(scalar_type)
    Q = (Ws @ Ws.T/self.d).astype(scalar_type)
    a = self.student.get_sl_vector()
    at = self.teacher.get_sl_vector()

    # Store metrics
    if not self.disable_QM_save:
      self.saved_as.append(a.copy())
      self.saved_Ms.append(M)
      self.saved_Qs.append(Q)

    self.saved_risks.append(self.theoretical_risk(Q,M,self.P, a, at))

    for metric_name, metric in self.extra_metrics.items():
      metric_list = getattr(self, metric_name)
      metric_list.append(metric(Q,M,self.P, a, at))
    
    

