import os
import traceback
import pandas as pd
from tqdm import tqdm

from allensdk.core.brain_observatory_cache import BrainObservatoryCache


LIMS_DF = "/Users/chase/Desktop/MindScope/Documents/20200324_vis_cod_list.xls"
IS_WINDOWS = False

def to_mac_file(filename: str):
    return filename.replace("\\", "/").replace("//allen", "/Volumes")


# Hacky workaround to get the file name from a full file path
def basename(filename: str):
    # return os.path.basename(filename)
    return filename[filename.rindex("\\")+1:]


class LocalEyeDataRepository(object):
    INFO_FILE = "data_info.csv"
    SESSION_TYPE_TO_FILE_PREFIX = {
        "three_session_A": "a",
        "three_session_B": "b",
        "three_session_C": "c",
        "three_session_C2": "c" # TODO Are these right?
    }

    
    def __init__(self, sessions_df):
        self.sessions_df = sessions_df
        
        
    @classmethod
    def from_dir(cls, local_path, boc):
        sessions_df = pd.read_csv(os.path.join(local_path, cls.INFO_FILE)).set_index("session_id", drop=False)
        # Note: drop=False keeps the session_id column so we can get indices using row["session_id"] (alternatively row.index)
        return LocalEyeDataRepository(sessions_df)

    
    @classmethod
    def build_new_repo(cls, boc, should_include_session, targeted_structures=None, session_types=None, lims_df="20200324_vis_cod_list.xls"):
        # Create the sessions DF
        sessions_df = pd.DataFrame(columns=[
            "session_id", "container_id", "targeted_structure", "cre_line", "session_type", "published_at",
            "eye_video_file", "eye_tracking_file", "time_sync_file",
            "allen_eye_video_file", "allen_eye_tracking_file", "allen_time_sync_file",
            "allen_eye_video_file_mac", "allen_eye_tracking_file_mac", "allen_time_sync_file_mac",
        ])
        
        sessions_df.set_index("session_id")        
        lims_df = pd.read_excel(lims_df)
        containers = boc.get_experiment_containers(targeted_structures=targeted_structures)
        container_ids = [container["id"] for container in containers]

#         container_ids = []
#         for cre_line in boc.get_all_cre_lines():
#             containers = boc.get_experiment_containers(targeted_structures=targeted_structures, cre_lines=[cre_line])
#             ids = [container["id"] for container in containers]
#             container_ids.extend(ids)

        imaging_sessions = boc.get_ophys_experiments(experiment_container_ids=container_ids, session_types=session_types)

        for session in imaging_sessions:
            if not session["fail_eye_tracking"] and should_include_session(session, sessions_df):
                sid = session["id"]
                cid = session["experiment_container_id"]
                session_type = session["session_type"]

                # Load the file locations from the LIMS DF
                # TODO: Failsafe if row doesn't exist
                data = lims_df[lims_df["ec_id"] == cid]
                if len(data) == 0:
                    print(f"Could not find data in lims spreadsheet: session {sid} (container {cid})")
                    continue
                data = data.iloc[0]
                prefix = cls.SESSION_TYPE_TO_FILE_PREFIX[session_type]
                eye_video = data[f"{prefix}_eye_raw_video"]
                eye_tracking = data[f"{prefix}_eye_tracking"]
                
                if pd.isna(eye_tracking):
                    print(f"Ignoring session {sid} with null eye tracking")
                    continue

                # Time sync file (a bit hacky because the storage locations don't always follow the same structure)
                if IS_WINDOWS:
                    time_sync_dir = os.path.dirname(os.path.dirname(eye_tracking))
                else:
                    time_sync_dir = os.path.dirname(os.path.dirname(to_mac_file(eye_tracking))).replace("/", "\\")

                # time_sync_dir = eye_tracking[:eye_tracking.rindex("\\", eye_tracking.rindex("\\"))]

                if not "ophys_experiment_" in time_sync_dir:
                    # time_sync_dir = os.path.join(time_sync_dir, f"ophys_experiment_{sid}")
                    time_sync_dir = f"{time_sync_dir}\\ophys_experiment_{sid}"

                # time_sync = os.path.join(time_sync_dir, f"{sid}_time_synchronization.h5")
                time_sync = f"{time_sync_dir}\\{sid}_time_synchronization.h5"

                # Add row to data frame
                sessions_df.loc[sid] = {
                    "session_id": sid,
                    "container_id": cid,
                    "targeted_structure": session["targeted_structure"],
                    "cre_line": session["cre_line"],
                    "session_type": session["session_type"],
                    "published_at": data["published_at"],
                    "eye_tracking_file": basename(eye_tracking),
                    "time_sync_file": basename(time_sync),
                    "eye_video_file": basename(eye_video),
                    "allen_eye_tracking_file": eye_tracking,
                    "allen_time_sync_file": time_sync,
                    "allen_eye_video_file": eye_video,
                    "allen_eye_tracking_file_mac": to_mac_file(eye_tracking),
                    "allen_time_sync_file_mac": to_mac_file(time_sync),
                    "allen_eye_video_file_mac": to_mac_file(eye_video),
                }
        
        return LocalEyeDataRepository(sessions_df)
    
        
    def get_sessions_df(self):
        return self.sessions_df
    
    
    def get_session_ids(self):
        return self.sessions_df.index.values
    
    
    def get_session_data(self, session_id):
        return self.sessions_df.loc[session_id]
    
    
    def __len__(self):
        return len(self.sessions_df)
    
    
    def pretty_print(self, show_files=False):
        if show_files:
            for session_id, session_data in self.sessions_df.sort_values(["targeted_structure", "cre_line", "container_id", "session_type"]).iterrows():
                print(f"- {session_data['targeted_structure']} {session_data['cre_line']}: {session_id} ({session_data['session_type']})")
                print(f"    Tracking: {session_data['eye_tracking_file']}")
                print(f"    Video:    {session_data['eye_video_file']}")
                print(f"    Sync:     {session_data['time_sync_file']}")
        else:
            counts_series = self.sessions_df.sort_values(["targeted_structure", "cre_line", "container_id", "session_type"]) \
                .groupby(["targeted_structure", "cre_line", "session_type"])["session_id"].count()
            last = None
            
            for (targeted_structure, cre_line, session_type), count in counts_series.iteritems():
                if last != (targeted_structure, cre_line):
                    print(f"- {targeted_structure} {cre_line}:")
                last = (targeted_structure, cre_line)
                print(f"  * {session_type.split('_')[-1]}: {count} session{'' if count == 1 else 's'}")
        
        print()
        print("Total number of sessions:", len(self.sessions_df))
            
            
    def copy_files(self, dest_dir, copy_eye_tracking=True, copy_eye_videos=False):
        from shutil import copy2 as copy # copy2 preserves creation and modification times
        videos_to_copy = []
        success_count = 0
        n_total = 0
        filenotfound_count = 0
        fail_count = 0
        
        if not os.path.exists(dest_dir):
            os.mkdir()

        suffix = "" if IS_WINDOWS else "_mac"
        file_keys = (f"allen_eye_tracking_file{suffix}", f"allen_time_sync_file{suffix}")

#         if copy_eye_tracking: print("Copying eye tracking files...")
        with tqdm(total=len(self.sessions_df)) as pbar:
            for session_id, session_data in self.sessions_df.sort_values(["targeted_structure", "cre_line", "container_id", "session_type"]).iterrows():
                targeted_structure = session_data["targeted_structure"]
                cre_line = session_data["cre_line"]
                sid = session_data["session_id"]
                stype = session_data['session_type']
                desc_prefix = f"[{targeted_structure}] [{cre_line}] Session: {sid} ({stype[stype.rindex('_')+1:]}) --"
                pbar.set_description(f"{desc_prefix} Copying eye tracking files")
                
                if copy_eye_tracking:
                    for allen_file_key in file_keys:
                        allen_file = session_data[allen_file_key]
                        n_total +=  1
                        try:
                            copy(allen_file, dest_dir)
                            success_count += 1
                        except FileNotFoundError as e:
                            filenotfound_count += 1
                        except OSError as e:
                            traceback.print_exc()
                            fail_count += 1
           
                if copy_eye_videos and session_data["eye_video_file"] is not None:
                    videos_to_copy.append(session_data[f"allen_eye_video_file{suffix}"])

                pbar.update()

        if len(videos_to_copy) > 0:
            print("Copying video files (this will take a while)...")
            with tqdm(total=len(videos_to_copy)) as pbar:
                for video_file in videos_to_copy:
                    pbar.set_description(f"Copying {os.path.basename(video_file)}")
                    n_total += 1
                    try:
                        copy(video_file, dest_dir)
                        success_count += 1
                    except FileNotFoundError as e:
                        filenotfound_count += 1
                    except OSError as e:
                        traceback.print_exc()
                        fail_count += 1
                    pbar.update()

        # Copy over the sessions dataframe
        # if success_count > 0:
        self.sessions_df.to_csv(os.path.join(dest_dir, LocalEyeDataRepository.INFO_FILE), index=False)
        print(f"Done! Copied {success_count}/{n_total} files (couldn't find {filenotfound_count} files; failed {fail_count} files).")


def main():
    drive_path = "/Users/chase/Desktop/MindScope/visual-coding-2p"
    manifest_file =  os.path.join(drive_path, "manifest.json")
    boc = BrainObservatoryCache(manifest_file=manifest_file)
    # session_types = [s for s in boc.get_all_session_types() if s.startswith("three_session_")]
    session_types = [
        "three_session_A",
        "three_session_B",
        "three_session_C",
        "three_session_C2",
    ]

    def should_include_session(session, session_id):
        # Include every session
        return True

    repo = LocalEyeDataRepository.build_new_repo(boc, should_include_session, session_types=session_types, lims_df=LIMS_DF)
    repo.pretty_print()

    dest_dir = "/Users/chase/Desktop/eye_data"
    repo.copy_files(dest_dir, copy_eye_tracking=True, copy_eye_videos=False)


if __name__ == "__main__":
    main()