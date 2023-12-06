#!/usr/bin/env python

import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from minizero.network.py.create_network import create_network
from tools.analysis import analysis
import matplotlib.pyplot as plt
from minizero.learner.transformer import ViT
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
from numpy import dot
from numpy.linalg import norm

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)


class MinizeroDadaLoader:
    def __init__(self, conf_file_name):
        self.data_loader = py.DataLoader(conf_file_name)
        self.data_loader.initialize()
        self.data_list = []

        # allocate memory
        self.sampled_index = np.zeros(py.get_batch_size() * 2, dtype=np.int32)
        self.features = np.zeros(py.get_batch_size() * py.get_nn_num_input_channels() * py.get_nn_input_channel_height() * py.get_nn_input_channel_width()*3, dtype=np.float32)
        self.loss_scale = np.zeros(py.get_batch_size(), dtype=np.float32)
        self.value_accumulator = np.ones(1) if py.get_nn_discrete_value_size() == 1 else np.arange(-int(py.get_nn_discrete_value_size() / 2), int(py.get_nn_discrete_value_size() / 2) + 1)
        if py.get_nn_type_name() == "alphazero":
            self.action_features = None
            self.policy = np.zeros(py.get_batch_size() * py.get_nn_action_size(), dtype=np.float32)
            self.value = np.zeros(py.get_batch_size() * py.get_nn_discrete_value_size(), dtype=np.float32)
            self.reward = None
            self.player_name = np.zeros(py.get_batch_size() * 3, dtype=np.float32)
            #add player name
        else:
            self.action_features = np.zeros(py.get_batch_size() * py.get_muzero_unrolling_step() * py.get_nn_num_action_feature_channels()
                                            * py.get_nn_hidden_channel_height() * py.get_nn_hidden_channel_width(), dtype=np.float32)
            self.policy = np.zeros(py.get_batch_size() * (py.get_muzero_unrolling_step() + 1) * py.get_nn_action_size(), dtype=np.float32)
            self.value = np.zeros(py.get_batch_size() * (py.get_muzero_unrolling_step() + 1) * py.get_nn_discrete_value_size(), dtype=np.float32)
            self.reward = np.zeros(py.get_batch_size() * py.get_muzero_unrolling_step() * py.get_nn_discrete_value_size(), dtype=np.float32)
            
    def load_data(self, training_file):
        
        file_name = f"{training_file}"
            #file_name = f"{training_dir}/sgf/{i}.sgf"
            #if file_name in self.data_list:
            #    continue
        self.data_loader.load_data_from_file(file_name)
        self.data_list.append(file_name)
        print(len(self.data_list))
            #if len(self.data_list) > py.get_zero_replay_buffer():
            #    self.data_list.pop(0)

    def sample_data(self, device='cpu'):
        
        self.data_loader.sample_data(self.features, self.action_features, self.policy, self.value, self.reward, self.loss_scale, self.sampled_index, self.player_name)
        #print("sample")
        features = torch.FloatTensor(self.features).view(py.get_batch_size(),3, py.get_nn_num_input_channels(), py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(device)
        action_features = None if self.action_features is None else torch.FloatTensor(self.action_features).view(py.get_batch_size(),
                                                                                                                 -1,
                                                                                                                 py.get_nn_num_action_feature_channels(),
                                                                                                                 py.get_nn_hidden_channel_height(),
                                                                                                                 py.get_nn_hidden_channel_width()).to(device)
        policy = torch.FloatTensor(self.policy).view(py.get_batch_size(), -1, py.get_nn_action_size()).to(device)
        value = torch.FloatTensor(self.value).view(py.get_batch_size(), -1, py.get_nn_discrete_value_size()).to(device)
        reward = None if self.reward is None else torch.FloatTensor(self.reward).view(py.get_batch_size(), -1, py.get_nn_discrete_value_size()).to(device)
        loss_scale = torch.FloatTensor(self.loss_scale / np.amax(self.loss_scale)).to(device)
        #print(loss_scale)
        sampled_index = self.sampled_index
        player_name = torch.FloatTensor(self.player_name).view(py.get_batch_size(), -1, 3).to(device)
        #add player name

        return features, action_features, policy, value, reward, loss_scale, sampled_index, player_name

    def update_priority(self, sampled_index, batch_values):
        batch_values = (batch_values * self.value_accumulator).sum(axis=1)
        self.data_loader.update_priority(sampled_index, batch_values)


class Model:
    def __init__(self):
        self.training_step = 0
        self.network = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.scheduler = None

    def load_model(self, model_file):
        self.training_step = 0
        #self.network = torch.jit.load(f"{training_dir}/model/{model_file}", map_location=self.device)
        self.network = create_network(py.get_game_name(),
                                      py.get_nn_num_input_channels(),
                                      py.get_nn_input_channel_height(),
                                      py.get_nn_input_channel_width(),
                                      py.get_nn_num_hidden_channels(),
                                      py.get_nn_hidden_channel_height(),
                                      py.get_nn_hidden_channel_width(),
                                      py.get_nn_num_action_feature_channels(),
                                      py.get_nn_num_blocks(),
                                      py.get_nn_action_size(),
                                      py.get_nn_num_value_hidden_channels(),
                                      py.get_nn_discrete_value_size(),
                                      py.get_nn_type_name())
        self.network.to(self.device)
        self.network2 = ViT(input_dim=py.get_nn_action_size()*3,
                               output_dim=256,
                               dim=512,
                               depth=12,
                               heads=3,
                               mlp_dim=1024,
                               pool='mean',
                               dropout=0.1,
                               emb_dropout=0.1)
        self.network2.to(self.device)
        self.optimizer = optim.SGD(list(self.network.parameters())+list(self.network2.parameters()),
                                   lr=py.get_learning_rate(),
                                   momentum=py.get_momentum(),
                                   weight_decay=py.get_weight_decay())
        #
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000000, gamma=0.5)  # gamma: 0.5, 
        
        #print(self.network)
        
        
        if model_file:
            snapshot = torch.load(f"{model_file}", map_location=torch.device('cpu'))
            #print(snapshot)
            #self.training_step = snapshot['training_step']
            
            self.network = torch.jit.load(f"{model_file.replace('.pkl','.pt')}", map_location=self.device)
            self.network.load_state_dict(snapshot['network'])
            self.network2.load_state_dict(snapshot['network2'])
            #print(self.network)
            #self.optimizer.load_state_dict(snapshot['optimizer'])
            self.optimizer.param_groups[0]["lr"] = py.get_learning_rate()
            #self.scheduler.load_state_dict(snapshot['scheduler'])
            
            print(self.optimizer.param_groups[0]["lr"])
        
        # for multi-gpu
        self.network = nn.DataParallel(self.network)
        self.network2 = nn.DataParallel(self.network2)

class PolicyPlay():
    def __init__(self, game_type, conf_file,test_file,train_file):
        _temps = __import__(f'build.{game_type}', globals(), locals(), ['minizero_py'], 0)
        py = _temps.minizero_py
        py.load_config_file(conf_file)

        self.env = py.Env()
        self.model = Model()
        
        #self.load_model('style_training_resnet/model/weight_iter_20000.pkl')ver1
        self.load_model('style_training_transformernew/model/weight_iter_35000.pkl')
        
        self.data_loader = py.TestDataLoader(conf_file)
        self.data_loader.load_test_data_from_file(test_file)

        self.can_data_loader = MinizeroDadaLoader(conf_file)
        self.can_data_loader.load_data(train_file)
        self.model.network.eval()
        self.model.network2.eval()

    def resetEnv(self):
        self.env.reset()

    def act(self, action):
        return self.env.act(action)

    def load_model(self, nn_file_name):
        self.model.load_model(nn_file_name)
    def calculate_similarity(self, candidate_embeds, query_embeds, topk = 4) : 
        #print('start cal similarity') 
        #print("[{}] ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))) 
        
        predict=[] 
        compare_table =[] 
             
        for embeds_player_id in candidate_embeds : 
            a = np.squeeze(np.asarray(query_embeds.cpu().detach(), dtype = np.float64)) 
            b = np.squeeze(np.asarray(embeds_player_id.cpu().detach(), dtype = np.float64)) 
            dists = 1 - dot(a, b) /(norm(a) * norm(b)) 
            compare_table.append(dists) 
                
        arry = np.array(compare_table) 
        arry_index = np.argsort(arry, axis = 0) 

        top5 = arry_index[ : 5]
        small_batch = py.get_batch_size()/3
        top5 = np.floor(top5/small_batch).astype(np.int32) 
        counts = np.bincount(top5)
        predict.append(np.argmax(counts))
        print(top5)

        top3 = arry_index[ : 3]
        top3 = np.floor(top3/small_batch).astype(np.int32) 
        counts = np.bincount(top3)
        predict.append(np.argmax(counts))

        top1 = arry_index[ : 1]
        top1 = np.floor(top1/small_batch).astype(np.int32) 
        predict.append(top1[0])


        predict = np.array(predict)
        print(predict)
        counts = np.bincount(predict) 
        answer = np.argmax(counts)
            
       #print('end cal similarity') 
        #print("[{}] ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))) 
        return answer


    def generate_games(self):
        #envloader = self.data_loader.getenvloader()
            
        for i in range(self.data_loader.get_loader_size()):
            '''
            features = []
            features.append(envs[i].get_features())
        # forward
            features = np.concatenate(features, axis=0)
            features = torch.FloatTensor(features).view(len(envs), py.get_nn_num_input_channels(), py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(self.model.device)
            '''
            features = self.data_loader.get_game_feature(i) 
            features = torch.FloatTensor(features).view(1,3,py.get_nn_num_input_channels(), py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(self.model.device)
            
            can_features, action_features, label_policy, label_value, label_reward, loss_scale, sampled_index, player_name = self.can_data_loader.sample_data(self.model.device)
            can_network_output = self.model.network(can_features[:,0])
            can_network_output2 = self.model.network(can_features[:,1]) 
            can_network_output3 = self.model.network(can_features[:,2]) 
            can_network_output =  torch.cat((can_network_output["policy_logit"],can_network_output2["policy_logit"],can_network_output3["policy_logit"]),1)
            
            can_network_output = self.model.network2(can_network_output)["style"]
            #print(features[0][16][0][0])
            #print(features.shape)
            network_output = self.model.network(features[:,0])
            network_output2 = self.model.network(features[:,1]) 
            network_output3 = self.model.network(features[:,2]) 
            network_output =  torch.cat((network_output["policy_logit"],network_output2["policy_logit"],network_output3["policy_logit"]),1)
            
            network_output = self.model.network2(network_output)["style"]
            max_output = self.calculate_similarity(can_network_output,network_output) 
            print(max_output)
            
            
            #print(max_output.shape)
            df = open('TestingDataset_Private/public_private_submission_template.csv').read().splitlines()
            games_id = [k.split(',',2)[0] for k in df]
            with open('test.csv','a') as f:
                answer_row = games_id[i+54676] + ',' + str(max_output+1) + '\n'
                
                f.write(answer_row)
            
        # return list of generated games
       
        

        


if __name__ == '__main__':
    if len(sys.argv) == 5:
        game_type = sys.argv[1]
        conf_file_name = sys.argv[2]
        eval_data = sys.argv[3]
        train_data = sys.argv[4]
        
        # import pybind library
        _temps = __import__(f'build.{game_type}', globals(), locals(), ['minizero_py'], 0)
        py = _temps.minizero_py
    else:
        eprint("python train.py game_type conf_file eval.sgf train.sgf")
        exit(0)
    print(eval_data)
    policy_play = PolicyPlay(game_type, conf_file_name,eval_data,train_data)
    policy_play.generate_games()
