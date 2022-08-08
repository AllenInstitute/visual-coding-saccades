import sys
import os
import multiprocessing
import time
from datetime import timedelta
import signal
import traceback
import pickle


class ParallelProcess():
    def __init__(self, save_dir=None):
        if save_dir is not None:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = f"{save_dir}_{timestamp}"
            print(f"Saving data to {self.save_dir}")
            os.makedirs(self.save_dir)
        else:
            self.save_dir = None


    def get_save_dir(self):
        return self.save_dir


    def write_pickle_file_output(self, filename, obj):
        with open(os.path.join(self.save_dir, filename), "wb") as file:
            pickle.dump(obj, file)


    def queue_listener(self, queue, total_jobs, output_handler):
        print(f"Queue handler loaded in process {os.getpid()}. {total_jobs} total jobs.")
        num_done = 0
        
        while True:
            try:
                data = queue.get()
                if data == "kill":
                    break
                else:
                    output_handler(data)
                    num_done += 1
                    print(f"Done with {num_done}/{total_jobs} jobs.")
            except (KeyboardInterrupt, SystemExit):
                print("Queue listener found interrupt")
                break
            except Exception:
                traceback.print_exc()

    def job(self, job_fn, args, queue):
        try:
            result = job_fn(*args)
            queue.put(result)
        except Exception:
            traceback.print_exc()
            print("Error doing job; exception above.")


    def run(self, job_fn, args, output_handler):
        total_jobs = len(args)
        print(f"Computer has {multiprocessing.cpu_count()} CPUs. Main thread is process {os.getpid()}.")
        
        # Based off: https://stackoverflow.com/a/35134329
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        processes = multiprocessing.cpu_count() + 2

        def pool_initializer():
            # Ignore CTRL+C in the worker process
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        with multiprocessing.Pool(processes, pool_initializer, maxtasksperchild=3) as pool:
            signal.signal(signal.SIGINT, original_sigint_handler)
            event_listener = pool.apply_async(self.queue_listener, args=(queue, total_jobs, output_handler))

            # Start jobs
            start_time = time.time()            
            print(f"Waiting for {total_jobs} jobs to complete...")
            internal_job_args = [(job_fn, a, queue) for a in args]
            try:
                result = pool.starmap_async(self.job, internal_job_args)
                result = result.get()
                print(f"Finished {total_jobs} jobs in {timedelta(seconds=(time.time() - start_time))}")
            except (KeyboardInterrupt, SystemExit):
                print("Caught interrupt; terminating workers")
                pool.terminate()
                pool.join()
                # self.queue.close()
                # self.queue.join_thread()
                sys.exit(0)
                return
            except Exception:
                traceback.print_exc()