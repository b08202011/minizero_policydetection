num_gpu=$(nvidia-smi -L | wc -l)
gpu_list=$(echo $num_gpu | awk '{for(i=0;i<$1;i++)printf i}')
cuda_devices=$(echo ${gpu_list} | awk '{ split($0, chars, ""); printf(chars[1]); for(i=2; i<=length(chars); ++i) { printf(","chars[i]); } }')
#echo "CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python minizero/learner/train.py go  "
#CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python minizero/learner/train.py go dan_training_oringinnoloss2 dan_training_oringinnoloss2/go_19x19_gaz_6bx256_n18-0c403e.cfg TrainingDataset/dan_train_train.sgf TrainingDataset/dan_train._val.sgf
echo "CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python minizero/learner/train.py go  "
#CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python minizero/learner/trainsimilarity.py go style_training_transformer3block load_training.cfg TrainingDataset/play_style_train_train.sgf TrainingDataset/play_style_train._val.sgf  
CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python minizero/learner/trainsimilarity.py go style_training_transformernew load_training.cfg TrainingDataset/play_style_train_train.sgf TrainingDataset/play_style_train._val.sgf  
         
