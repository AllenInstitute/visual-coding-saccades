import sys
sys.path.append("./")

import time
import traceback

import os
import gc
import time
import traceback

import pandas as pd

from helpers.parallel_process import ParallelProcess

# https://github.com/AllenInstitute/AllenSDK/issues/1436#issuecomment-610082420
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_project_api import EcephysProjectWarehouseApi
from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import RmaEngine


def job(cache, session_id):
    """
    Ensures a session is downloaded.
    """
    cache.get_session_data(session_id)


"""
Runs the above data-loading process in parallel.
"""
if __name__ == "__main__":
    process = ParallelProcess()

    def output_handler(result):
        if result is None:
            return

        print(f"Downloaded session ID {result}")

    drive_path = r"/Users/chase/Desktop/MindScope/ecephys-cache"
    manifest_file = os.path.join(drive_path, "manifest.json")
    timeout_mins = 60
    fetch_api = EcephysProjectWarehouseApi(RmaEngine(scheme="http", host="api.brain-map.org", timeout=timeout_mins*60))
    cache = EcephysProjectCache(manifest=manifest_file, fetch_api=fetch_api)
    sessions = cache.get_session_table()
    task_args = []

    for session_id, session_row in sessions.iterrows():
        task_args.append((cache, session_id))

    print(f"There are {len(task_args)} total ecephys sessions to process.")
    process.run(job, task_args, output_handler, parallel=False)