import os 
import sys
import _pickle as pickle
import numpy as np
import configparser
import glob
import string

from datetime import datetime
from tqdm import tqdm
from utilities import logger
from utilities import decorators_and_wrappers as decorator
from tqdm import tqdm

import json


@decorator.singleton
class Utils(object):

    def __init__(self):
    
      self._config_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
      self.config_filename = os.path.join(self._config_dir, 'file_config.ini')
      self.config = configparser.ConfigParser()
      self.config.read(self.config_filename)
  
      self.set_paths()
      self._project_name = self.set_project_name()


    @property
    def config_dir(self):
      return self._config_dir

    @property
    def project_path(self):
      return self._project_path

    @property
    def output_path(self):
      return self._output_path

    @property
    def data_path(self):
      return self._data_path
 
    @property
    def weight_path(self):
      return self._weight_path

    def set_project_name(self):

      name = ''

      if 'Experiment_Metadata' in self.config and 'project_name' in self.config['Experiment_Metadata']:
        name = self.config['Experiment_Metadata']['project_name']

      if name == '':
        name = datetime.now().strftime('Experiment_%B_%Y')

      
      return name


    def set_paths(self):

      relative_paths = ['.', '..', '~']

      if 'Experiment_Paths' in self.config:
        Experiment_Paths = self.config['Experiment_Paths']

        if 'project_root_path' in Experiment_Paths and Experiment_Paths['project_root_path'] not in relative_paths:
          self._project_path = Experiment_Paths['project_root_path']
        else:
          self._project_path = os.path.dirname(os.path.abspath(Experiment_Paths['project_root_path']))

        if 'output_path' in Experiment_Paths and Experiment_Paths['output_path'] != 'output':
          self._output_path = Experiment_Paths['output_path']
        else:
          self._output_path = os.path.join(self._project_path, 'output')

        if 'data_path' in Experiment_Paths and Experiment_Paths['data_path'] != 'data':
          self._data_path = Experiment_Paths['data_path']
        else:
          self._data_path = os.path.join(self._project_path, 'data')
        
        if 'weight_path' in Experiment_Paths and Experiment_Paths['weight_path'] != 'weight':
          self._weight_path = Experiment_Paths['weight_path']
        else:
          self._weight_path = os.path.join(self._project_path, 'weight')
        
      else:

        self._project_path = os.path.dirname(os.path.abspath('..'))
        self._output_path = os.path.join(self._project_path, 'output')
        self._data_path = os.path.join(self._project_path, 'data')
        self._weight_path = os.path.join(self._project_path, 'weight')


      self.path_exists(self._output_path, True)
      self.path_exists(self._data_path, True)
      self.path_exists(self._weight_path, True)



    def path_exists(self, path, createPath = False):
        
        if path == None: sys.exit('Path value invalid')

        if os.path.exists(path) == True: return path 

        if os.path.exists(path) is False and createPath == True:
            os.makedirs(path)
        else:
            sys.exit('path doesnt exists {}'.format(path))

        return path


    def setup_logger(self, 
                     dirname = 'output', 
                     stream = False,
                     filename = 'system.log',
                     level='INFO', 
                     path = None):
        
        log = logger.create_logger(name = self._project_name)
        
        if path == None:
          path = os.path.join(self.path_exists(os.path.join(self._output_path, dirname), True), filename)

        log.addHandler(logger.get_fileHandler(filename=path))
        
        if stream == True: 
            log.addHandler(logger.get_streamHandler())
        
        return log


    def unpickle_data(self, 
                      filename='default', 
                      dirname = 'data/', 
                      path = None):
        
        if path == None:
          path = os.path.join(self._project_path, '{}/{}'.format(dirname, filename))

        path = self.path_exists(path)

        with open(path, 'rb') as f:
              data = pickle.load(f)

        return data


    def pickle_data(self, 
                    data, 
                    filename = 'default', 
                    dirname = 'output/', 
                    path = None):
    
        if path == None:
            path = self.path_exists(os.path.join(self._project_path, dirname), True)
            path = os.path.join(path ,filename)
        

        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=2)


    def read_json(self, 
                  filename='default', 
                  dirname = 'data/', 
                  path = None, 
                  multiobj = False, 
                  pbar = True):
        
        if path == None:
          path = os.path.join(self._project_path, '{}/{}'.format(dirname, filename))

        path = self.path_exists(path)

        if multiobj:
          if pbar:
            data = [json.loads(f.strip()) for f in tqdm(open(path).readlines())]
          else:
            data = [json.loads(f.strip()) for f in open(path).readlines()]
          return data 

        with open(path, 'r') as f:
              data = json.load(f)

        return data


    def write_json(self, 
                   data, 
                   filename = 'default', 
                   dirname = 'output/', 
                   ensure_ascii = False,
                   path = None, 
                   mode = 'w'):
    
        if path == None:
            path = self.path_exists(os.path.join(self._project_path, dirname), True)
            path = os.path.join(path ,filename)
        
        with open(path, mode) as f:
            if mode == 'w':
              json.dump(data, f, ensure_ascii = ensure_ascii)
            else:
              f.write(json.dumps(data) + '\n')

    
    def read_text(self, 
                  filename='default', 
                  dirname = 'data/', 
                  path = None, 
                  convert = None):
        
        if path == None:
          path = os.path.join(self._project_path, '{}/{}'.format(dirname, filename))

        path = self.path_exists(path)

        if convert == 'int':
          return [int(s.strip()) for s in open(path).readlines()]
        if convert == 'float':
          return [float(s.strip()) for s in open(path).readlines()]

        return [s.strip() for s in open(path).readlines()]


    def write_text(self, 
                   data, 
                   filename = 'default', 
                   dirname = 'output/', 
                   mode = 'w',
                   path = None, 
                   list_data = False):
    
        if path == None:
            path = self.path_exists(os.path.join(self._project_path, dirname), True)
            path = os.path.join(path ,filename)

        with open(path, mode) as f:
          data = '\n'.join([str(s) for s in data]) if list_data else data
          f.write(data + '\n')
        

    def get_alpha_directories(self, base_path, subdirname = '_author', create_path = True):

        directories = [(a.lower(), '{}{}'.format(a, subdirname)) for a in string.ascii_uppercase] + [('unk', 'Other{}'.format(subdirname))]
        keys, values = list(zip(*directories))
        directories = dict(zip(keys, values))
        directories = {k: self.path_exists(os.path.join(base_path, v), create_path) for k, v in directories.items()}

        return directories



    def merge_into_one(self, parent_dir, scr_filename, dest_filename, remove_files = False):

        dest_filename = so.path.join(parent_dir, dest_filename)
        files = list(glob.glob(os.path.join(parent_dir, scr_filename)))

        for f in tqdm(files):
          content = self.read_text(path = f)
          self.write_text(content, path = dest_filename, mode = 'a', list_data = True) 




