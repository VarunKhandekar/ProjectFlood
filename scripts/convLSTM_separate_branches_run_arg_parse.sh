#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=vk223
#SBATCH --partition gpgpuC --gres=gpu:1
#SBATCH --output=convLSTM_separate_branches_%j.out

export PROJECT_FLOOD_DATA="/homes/vk223/ProjectFlood/static/imperial_data_paths.json"
export PROJECT_FLOOD_CORE_PATHS="/homes/vk223/ProjectFlood/static/imperial_core_paths.json"
export PROJECT_FLOOD_REPO_DIR="/homes/vk223/ProjectFlood"
source /vol/bitbucket/vk223/project_flood/projectfloodvenv/bin/activate
source /vol/cuda/11.8.0/setup.sh
export PYTHONPATH=$PYTHONPATH:${PROJECT_FLOOD_REPO_DIR}/
export PROJECT_FLOOD_SCRIPTPATH=model_runs/convLSTM_separate_branches_run_arg_parse.py
export PROJECT_FLOOD_SCRIPTLOC=${PROJECT_FLOOD_REPO_DIR}/${PROJECT_FLOOD_SCRIPTPATH}
TERM=vt100 
#TERM=xterm


echo -e "\n\n\n\n\n\n\n\n"
echo -e "\n******************************************************************************"
echo -e "\n**********************************STARTING************************************"
echo -e "\n******************************************************************************"
/usr/bin/nvidia-smi
python3 -V

NUM_EPOCHS=(5000)
TRAIN_BATCH_SIZES=(32)
LEARNING_RATES=(0.001 0.0001)
PRECEDING_RAINFALL_DAYS=(1 3)
DROPOUT_PROBS=(0.2 0.3 0.5)
OUTPUT_CHANNELS=(16)
CONV_BLOCK_LAYERS=(4)
CONVLSTM_LAYERS=(2)
OPTIMIZERS=('RMSprop')
CRITERIONS=('BCELoss')
RESOLUTIONS=(256)
TRANSFORMS=(False True)


for num_epochs in "${NUM_EPOCHS[@]}"; do
  for train_batch_size in "${TRAIN_BATCH_SIZES[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
      for preceding_rainfall_days in "${PRECEDING_RAINFALL_DAYS[@]}"; do
        for dropout_prob in "${DROPOUT_PROBS[@]}"; do
          for output_channels in "${OUTPUT_CHANNELS[@]}"; do
            for conv_block_layers in "${CONV_BLOCK_LAYERS[@]}"; do
              for convLSTM_layers in "${CONVLSTM_LAYERS[@]}"; do
                for optimizer_str in "${OPTIMIZERS[@]}"; do
                  for criterion_str in "${CRITERIONS[@]}"; do
                    for resolution in "${RESOLUTIONS[@]}"; do
                      for transforms in "${TRANSFORMS[@]}"; do
                        
                        echo -e "\n\n\n"
                        python3 $PROJECT_FLOOD_SCRIPTLOC \
                            --num_epochs $num_epochs \
                            --train_batch_size $train_batch_size \
                            --learning_rate $learning_rate \
                            --preceding_rainfall_days $preceding_rainfall_days \
                            --dropout_prob $dropout_prob \
                            --output_channels $output_channels \
                            --conv_block_layers $conv_block_layers \
                            --convLSTM_layers $convLSTM_layers \
                            --optimizer_str $optimizer_str \
                            --criterion_str $criterion_str \
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
    done
  done
done


echo -e "\n\n\n\n\n\n\n\n"
echo -e "\n******************************************************************************"
echo -e "\n**********************************COMPLETE************************************"
echo -e "\n******************************************************************************"
/usr/bin/nvidia-smi
uptime