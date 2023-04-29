import sys
import os
import multiprocessing
import time
from datetime import timedelta
import signal
import traceback
import pickle

def _pool_initializer():
    """Ignore CTRL+C in the worker process"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class ParallelProcess():
    """Handles the asynchronous running of a parallel processes.
    """

    KILL_QUEUE_LISTENER = ("internal", "kill")

    def __init__(self, save_dir=None):
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        if save_dir is not None:
            self.save_dir = f"{save_dir}_{self.timestamp}"
            print(f"Saving data to {self.save_dir}")
            os.makedirs(self.save_dir)
        else:
            self.save_dir = None

        self.max_n_processes = -1

    def get_save_dir(self):
        return self.save_dir


    def write_pickle_file_output(self, filename, obj):
        with open(os.path.join(self.save_dir, filename), "wb") as file:
            pickle.dump(obj, file)


    def job(self, *args):
        raise NotImplementedError("job must be implemented in subclass")


    def output_handler(self, job_result):
        # raise NotImplementedError("output_handler must be implemented in subclass")
        pass


    def _queue_listener(self, queue, total_jobs):
        print(f"Queue handler loaded in process {os.getpid()}. {total_jobs} total jobs.")
        num_done = 0
        
        while True:
            try:
                queue_data = queue.get()

                if queue_data == ParallelProcess.KILL_QUEUE_LISTENER:
                    # Signals the process is done
                    break
                    
                channel, data = queue_data
                if channel == "job-result":
                    self.output_handler(data)
                    num_done += 1
                    print(f"Done with {num_done}/{total_jobs} jobs.")
            except (KeyboardInterrupt, SystemExit):
                print("Queue listener found interrupt")
                break
            except Exception:
                traceback.print_exc()


    def _job(self, args, queue):
        """Wrapper function to run a job function with the specified args and append the return value to the queue.

        Args:
            job_fn (function): Job function
            args (tuple, list): List of arguments for job function
            queue (multiprocessing Queue): Multithreading queue object
        """
        try:
            result = self.job(*args)
            if queue is None:
                return result
            else:
                queue.put(("job-result", result))
                return result
        except Exception:
            traceback.print_exc()
            print("Error doing job; exception above.")


    def run(self, args, n_processes: int=None, parallel: bool=True):
        """Runs a set of parallel processes.

        Args:
            args (list): List of arguments to be passed in self.job.
            n_processes (int, optional): Number of processes/threads are used to run tasks. Defaults to None, which is computed as N_CPUS+2.
            parallel (bool, optional): Whether the processes are run in parallel; useful for testing. Defaults to True.
        """

        total_jobs = len(args)
        if parallel: print(f"Computer has {multiprocessing.cpu_count()} CPUs. Main thread is process {os.getpid()}.")
        
        # Start jobs
        start_time = time.time()
        print(f"Waiting for {total_jobs} jobs to complete...")

        def done_msg():
            print(f"Finished {total_jobs} jobs in {timedelta(seconds=(time.time() - start_time))}")
        
        if not parallel:
            num_done = 0
            for a in args:
                result = self._job(a, None)
                self.output_handler(result)
                num_done += 1
                print(f"Done with {num_done}/{total_jobs} jobs.")
            done_msg()
            return

        # Based off: https://stackoverflow.com/a/35134329
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        if n_processes is None:
            n_processes = multiprocessing.cpu_count() + 2
        if self.max_n_processes > 0:
            n_processes = min(n_processes, self.max_n_processes)
        
        print(f"Running on {n_processes} threads")

        with multiprocessing.Pool(n_processes, initializer=_pool_initializer, maxtasksperchild=3) as pool:
            signal.signal(signal.SIGINT, original_sigint_handler)

            # Start the queue listener
            event_listener = pool.apply_async(self._queue_listener, args=(queue, total_jobs))

            # Run the jobs
            internal_job_args = [(a, queue) for a in args]
            try:
                result = pool.starmap_async(self._job, internal_job_args)
                job_results = result.get()
                queue.put(ParallelProcess.KILL_QUEUE_LISTENER)
                event_listener.get() # Wait for queue listener to finish handling tasks
                done_msg()
                return job_results
            except (KeyboardInterrupt, SystemExit):
                print("Caught interrupt; terminating workers")
                pool.terminate()
                pool.join()
                # self.queue.close()
                # self.queue.join_thread()
                pool.close()
                sys.exit(0)
                return
            except Exception:
                traceback.print_exc()
            
            return None