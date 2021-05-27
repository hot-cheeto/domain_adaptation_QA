import time
import functools
import random 
import multiprocessing as mp
from multiprocessing import Process
import logging
import os 


def singleton(cls):
    return cls()

    
def create_chunks(data, num_chunk, static_chunk = True):
    
    data_chunks = []

    if static_chunk:
      total_data = len(data)
      k, m = divmod(total_data, num_chunk)
      data_chunks = [data[i * k + min(i, m):(i + 1) *k + min(i + 1, m)] for i in range(num_chunk)]

    else:
      all_file_sizes = [(f, os.stat(f).st_size) for f in data]
      all_file_sizes = sorted(all_file_sizes, key = lambda x: x[-1])[::-1]
      chunks = {i: [] for i in range(num_chunk)}
      id_ = 0

      for file_ in all_file_sizes:
        chunks[id_].append(file_[0])
        id_ += 1

        if id_ == num_chunk:
          id_ = 0

      data_chunks = list(chunks.values())

    assert len(data_chunks) == num_chunk

    return data_chunks


def doublewrap(function):
    
    @functools.wraps(function)
    
    def decorator(*args, **kwargs):
        
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):

    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    
    @property
    @functools.wraps(function)
    def decorator(self):
    
        if not hasattr(self, attribute):
          setattr(self, attribute, function(self))
        return getattr(self, attribute)
    
    return decorator


def multi_processor_wrapper(func,
                            data,
                            num_process,
                            params = [],
                            static_chunk = True, 
                            log = None):

    if log == None:
      mp.log_to_stderr()
      log = mp.get_logger()
      log.setLevel(logging.INFO)

    t0 = time.time()
    
    log.info('Creating data {} chunks'.format('static' if static_chunk else 'dynamic'))
    log.info('Number of process {} spawing'.format(num_process))
    chunks = create_chunks(data, num_process, static_chunk)
    
    log.info('Create {} chunks'.format(len(chunks)))
    procs = []

    for i, chunk in enumerate(chunks):
      proc = Process(target = func, args = params + [log, chunk, i])
      procs.append(proc)
      proc.start()

    for proc in procs:
      proc.join()

    log.info('Done with timelapse of {}'.format(time.time() - t0))


# TODO
# def one_job_main(params):

    # random.seed(43)
    # np.random.seed(43)

    # comm = MPI.COMM_WORLD
    # num_tasks = comm.size
    # rank = comm.Get_rank()

    # t_start = MPI.Wtime()

    # if rank == 0:
      
      # paths = create_paths(params)

      # print('Creating collecting queries: {}'.format(params.embedding_name))
      # embedding_path = utils.path_exists(os.path.join(params.root_path, params.queries, params.embedding_name))
      # start2author, lower_bound, all_keys = get_start2author(embedding_path)
      # data = list(glob.glob('{}/**/*.npy'.format(embedding_path)))

      # if params.debug:
        # data = data[:100]

      # print('Creating chunks for {} tasks'.format(num_tasks))
      # tasks_chunks = utils.create_chunks(data, num_tasks)

    # else:
      # dirs = None
      # paths = None
      # embedding_path = None
      # data = None
      # tasks_chunks = None 
      # start2author, lower_bound, all_keys = None, None, None

    # paths = comm.bcast(paths, root = 0)
    # tasks_chunks = comm.bcast(tasks_chunks, root = 0)
    # start2author = comm.bcast(start2author, root = 0)
    # lower_bound = comm.bcast(lower_bound, root = 0)
    # all_keys = comm.bcast(all_keys, root = 0)

    # comm.Barrier()

    # log = logging.getLogger('rank [{}] : starting process'.format(rank))
    # log.setLevel(logging.INFO)

    # mh = utils.MPIFileHandler(params.log_filename)
    # log.addHandler(mh)
    # log.info(rank)

    # chunk = tasks_chunks[rank]
    # log.info('[{}] - length of task chunk {}'.format(rank, len(chunk)))
    # get_metric_data(lower_bound, start2author, params.split_name, all_keys, paths, log, chunk, rank)
    
    # comm.Barrier()
    # t_diff = MPI.Wtime - t_start

    # print('[{}] done in {} seconds'.format(rank, t_diff))
    

