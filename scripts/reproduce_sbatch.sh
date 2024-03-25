#!/bin/bash
#SBATCH --output=output/logs/%A_%a.log  # please change accordingly
#SBATCH --error=output/logs/%A_%a.log   # please change accordingly
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-4  # number of scenes, 0-7 for Replica, 0-2 for TUM_RGBD, 0-5 for ScanNet, 0-4 for ScanNet++

dataset="Replica" # set dataset
if [ "$dataset" == "Replica" ]; then
    scenes=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")
    INPUT_PATH="data/Replica-SLAM"
elif [ "$dataset" == "TUM_RGBD" ]; then
    scenes=("rgbd_dataset_freiburg1_desk" "rgbd_dataset_freiburg2_xyz" "rgbd_dataset_freiburg3_long_office_household")
    INPUT_PATH="data/TUM_RGBD-SLAM"
elif [ "$dataset" == "ScanNet" ]; then
    scenes=("scene0000_00" "scene0059_00" "scene0106_00" "scene0169_00" "scene0181_00" "scene0207_00")
    INPUT_PATH="data/scannet/scans"
elif [ "$dataset" == "ScanNetPP" ]; then
    scenes=("b20a261fdf" "8b5caf3398" "fb05e13ad1" "2e74812d00" "281bc17764")
    INPUT_PATH="data/scannetpp/data"
else
    echo "Dataset not recognized!"
    exit 1
fi

OUTPUT_PATH="output"
CONFIG_PATH="configs/${dataset}"
EXPERIMENT_NAME="reproduce"
SCENE_NAME=${scenes[$SLURM_ARRAY_TASK_ID]}

source <path-to-conda.sh> # please change accordingly
conda activate gslam

echo "Job for dataset: $dataset, scene: $SCENE_NAME"
echo "Starting on: $(date)"
echo "Running on node: $(hostname)"

# Your command to run the experiment
python run_slam.py "${CONFIG_PATH}/${SCENE_NAME}.yaml" \
                   --input_path "${INPUT_PATH}/${SCENE_NAME}" \
                   --output_path "${OUTPUT_PATH}/${dataset}/${EXPERIMENT_NAME}/${SCENE_NAME}" \
                   --group_name "${EXPERIMENT_NAME}" \

echo "Job for scene $SCENE_NAME completed."
echo "Started at: $START_TIME"
echo "Finished at: $(date)"