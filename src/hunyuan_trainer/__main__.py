# NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/hunyuan_video.toml

# NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/hunyuan_video.toml --resume_from_checkpoint

import subprocess
import toml
import os
import sys
import argparse
from pathlib import Path
import platform
import shutil

script_dir = Path(__file__).parent / "src" / "hunyuan_trainer"

def setup_training_folder(target_folder):
    training_folder = Path.home() / "hunyuan_training"

    if training_folder.exists() and training_folder.is_dir():
        print(f"Using {training_folder}")
    else:
        choice = input(f"Folder '{training_folder}' does not exist. Create it? (y/n)").strip().lower()
        if choice.lower() == 'y':
            try:
                training_folder.mkdir(parents=True, exist_ok=True)
                print(f"Folder '{training_folder}' created successfully.")
            except Exception as e:
                print(f"Error creating folder: {e}")
                sys.exit(1)
        else:
            print("Folder not created. Exiting.")
            sys.exit(1)

    with open(f"{script_dir}/dataset.toml", "r") as dataset_file:
        dataset = toml.load(dataset_file)

    with open(f"{script_dir}/hunyuan_video.toml", "r") as video_file:
        video = toml.load(video_file)

    input_folder = f"{training_folder}/{target_folder}/input"
    print(target_folder)

    try:
        shutil.move(target_folder, input_folder)
        print(f"Successfully created {input_folder}")
    except Exception as e:
        print(f"Error creating {input_folder}")
        sys.exit(1)

    dataset["resolutions"] = [512]
    dataset["directory"][0]["path"] = input_folder


    output_dir = f"{training_folder}/{target_folder}/output"

    if Path(output_dir).exists():
        print(f"{output_dir} already exists. This shouldn't happen")
        sys.exit(1)
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)


    video["output_dir"] = output_dir
    video["dataset"] = f"{training_folder}/{target_folder}/dataset.toml"

    with open(f"{training_folder}/{target_folder}/dataset.toml", "w") as dataset_file_out:
        toml.dump(dataset, dataset_file_out)

    with open(f"{training_folder}/{target_folder}/hunyuan_video.toml", "w") as video_file_out:
        toml.dump(video, video_file_out)


def main():
    if platform.system() != "Linux":
        print("This works only on linux")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Process TOML files and store them in a target directory.")
    parser.add_argument("target_folder", type=str, help="Target folder to store the output files")
    args = parser.parse_args()

    setup_training_folder(args.target_folder)

    # session_name = "my_session"

    #
    # subprocess.run(["screen", "-dmS", session_name, "bash", "-c", 'NCCL_P2P_DISABLE="1"', 'NCCL_IB_DISABLE="1"', "deepspeed", "--num_gpus=1", "train.py", "--deepspeed", "--config", "hunyuan_video.toml"])
