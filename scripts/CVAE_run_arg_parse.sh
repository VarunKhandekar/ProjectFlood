#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=vk223
#SBATCH --partition gpgpu --gres=gpu:1
#SBATCH --output=CVAE_%j.out

export PROJECT_FLOOD_DATA="/homes/vk223/ProjectFlood/static/imperial_data_paths.json"
export PROJECT_FLOOD_CORE_PATHS="/homes/vk223/ProjectFlood/static/imperial_core_paths.json"
export PROJECT_FLOOD_REPO_DIR="/homes/vk223/ProjectFlood"
source /vol/bitbucket/vk223/project_flood/projectfloodvenv/bin/activate
source /vol/cuda/11.8.0/setup.sh
export PYTHONPATH=$PYTHONPATH:${PROJECT_FLOOD_REPO_DIR}/
export PROJECT_FLOOD_SCRIPTPATH=model_runs/CVAE_run_arg_parse.py
export PROJECT_FLOOD_SCRIPTLOC=${PROJECT_FLOOD_REPO_DIR}/${PROJECT_FLOOD_SCRIPTPATH}
TERM=vt100 
#TERM=xterm


echo -e "\n\n\n\n\n\n\n\n"
echo -e "\n******************************************************************************"
echo -e "\n**********************************STARTING************************************"
echo -e "\n******************************************************************************"
/usr/bin/nvidia-smi
python3 -V

NUM_EPOCHS=(1000)
TRAIN_BATCH_SIZES=(32)
LEARNING_RATES=(0.001)
DROPOUT_PROBS=(0.25)
LATENT_DIMENSIONS=(16)
OPTIMIZERS=('RMSprop')
CRITERION_BETAS=(1)
RESOLUTIONS=(256)
TRANSFORMS=('False')


for num_epochs in "${NUM_EPOCHS[@]}"; do
  for train_batch_size in "${TRAIN_BATCH_SIZES[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
      for dropout_prob in "${DROPOUT_PROBS[@]}"; do
        for latent_dims in "${LATENT_DIMENSIONS[@]}"; do
          for optimizer_str in "${OPTIMIZERS[@]}"; do
            for criterion_beta in "${CRITERION_BETAS[@]}"; do
              for resolution in "${RESOLUTIONS[@]}"; do
                for transforms in "${TRANSFORMS[@]}"; do
                  
                  echo -e "\n\n\n"
                  python3 $PROJECT_FLOOD_SCRIPTLOC \
                      --num_epochs $num_epochs \
                      --train_batch_size $train_batch_size \
                      --learning_rate $learning_rate \
                      --dropout_prob $dropout_prob \
                      --latent_dims $latent_dims \
                      --optimizer_str $optimizer_str \
                      --criterion_beta $criterion_beta \
                      --resolution $resolution \
                      --transforms $transforms

                done
              done
            done
          done
        done
      done
    done
  done
done


echo -e "\n\n\n\n\n\n\n\n"
echo -e "\n******************************************************************************"
echo -e "\n**********************************COMPLETE************************************"
echo -e "\n******************************************************************************"
/usr/bin/nvidia-smi
uptime