import yaml
import os
import hashlib
import numpy as np

from yaml import CLoader as Loader, CDumper as Dumper
# from yaml import CBaseLoader as Loader, CBaseDumper as Dumper
# from yaml import SafeLoader as Loader, SafeDumper as Dumper
# from yaml import Loader as Loader, Dumper as Dumper

class BaseResult():
  """"
  This is the base abstract class for the results.
  """
  def __init__(self, initial_condition = None, id = 0, **kattributes):
    self.initial_condition = initial_condition
    self.id = id

    for attr, val in kattributes.items():
      setattr(self, attr, val)

  def from_file(self, filename=None, path = ''):
    if filename is None:
      filename = self.get_initial_condition_id()

    with open(path+filename+'.yaml', 'r') as file:
      data = yaml.load(file, Loader=Loader)
      for att, val in data.items():
        setattr(self, att, val)

  def to_file(self, filename=None, path = ''):
    if filename is None:
      filename = self.get_initial_condition_id()

    data = {}
    for att, val in self.__dict__.items():
      if not att.startswith('__') and not callable(val):
        data[att] = val
    full_path_filename = path+filename+'.yaml'
    os.makedirs(path, exist_ok=True)
    with open(full_path_filename, 'w') as file:
      yaml.dump(data, file, Dumper=Dumper)

  def get_initial_condition_id(self):
    # Achtung!! Changing this function make all previous generated data unacessible!
    # Consider producing a script of conversion before apply modifications.
    ic_string = self.initial_condition
    if ic_string is None:
      ic_string = np.random.randint(int(1e9))

    datastring = '_'.join([
      str(ic_string),
      *self.datastring # Must be defined in the subclass!
    ])
    # print(datastring)
    return hashlib.md5(datastring.encode('utf-8')).hexdigest()
