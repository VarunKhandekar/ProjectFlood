#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=vk223
#SBATCH --partition gpgpuC --gres=gpu:1
#SBATCH --output=convLSTM_separate_branches_%j.out

export PROJECT_FLOOD_DATA="/homes/vk223/ProjectFlood/static/imperial_data_paths.json"
export PROJECT_FLOOD_CORE_PATHS="/homes/vk223/ProjectFlood/static/core_paths.json"
export PROJECT_FLOOD_REPO_DIR="/homes/vk223/ProjectFlood"
source /vol/bitbucket/vk223/project_flood/projectfloodvenv/bin/activate
source /vol/cuda/11.8.0/setup.sh
export PYTHONPATH=$PYTHONPATH:${PROJECT_FLOOD_REPO_DIR}/
export PROJECT_FLOOD_SCRIPTPATH=model_runs/convLSTM_separate_branches_run.py
export PROJECT_FLOOD_SCRIPTLOC=${PROJECT_FLOOD_REPO_DIR}/${PROJECT_FLOOD_SCRIPTPATH}
TERM=vt100 
#TERM=xterm

export NINPUT=2500

echo -e "\n\n\n\n\n\n\n\n"
echo -e "\n******************************************************************************"
echo -e "\n**********************************STARTING************************************"
echo -e "\n******************************************************************************"
/usr/bin/nvidia-smi
python3 -V

echo -e "\n\n\n"
python3 $PROJECT_FLOOD_SCRIPTLOC -n $NINPUT


echo -e "\n\n\n\n\n\n\n\n"
echo -e "\n******************************************************************************"
echo -e "\n**********************************COMPLETE************************************"
echo -e "\n******************************************************************************"
/usr/bin/nvidia-smi
uptime