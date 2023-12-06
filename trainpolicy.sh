num_gpu=$(nvidia-smi -L | wc -l)
gpu_list=$(echo $num_gpu | awk '{for(i=0;i<$1;i++)printf i}')
cuda_devices=$(echo ${gpu_list} | awk '{ split($0, chars, ""); printf(chars[1]); for(i=2; i<=length(chars); ++i) { printf(","chars[i]); } }')
CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python minizero/learner/train2.py go dan_training_laststep dan_training_new/go_19x19_gaz_6bx256_n18-0c403e.cfg TrainingDataset/dan_train_train.sgf TrainingDataset/dan_train._val.sgf  
