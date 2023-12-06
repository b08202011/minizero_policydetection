num_gpu=$(nvidia-smi -L | wc -l)
gpu_list=$(echo $num_gpu | awk '{for(i=0;i<$1;i++)printf i}')
cuda_devices=$(echo ${gpu_list} | awk '{ split($0, chars, ""); printf(chars[1]); for(i=2; i<=length(chars); ++i) { printf(","chars[i]); } }')
echo "CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python minizero/learner/train.py go  "
CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python minizero/learner/trainloss2.py go kyu_training_newloss2 dan_training_new/go_19x19_gaz_6bx256_n18-0c403e.cfg TrainingDataset/kyu_train_train.sgf TrainingDataset/kyu_train._val.sgf 
