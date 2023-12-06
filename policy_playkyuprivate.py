#!/usr/bin/env python

import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from minizero.network.py.create_network import create_network


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)


class Model:
    def __init__(self):
        self.training_step = 0
        self.network = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.scheduler = None

    def load_model(self, model_file):
        self.training_step = 0
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
        self.optimizer = optim.SGD(self.network.parameters(),
                                   lr=py.get_learning_rate(),
                                   momentum=py.get_momentum(),
                                   weight_decay=py.get_weight_decay())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000000, gamma=0.1)

        if model_file:
            snapshot = torch.load(model_file, map_location=torch.device('cpu'))
            self.training_step = snapshot['training_step']
            self.network.load_state_dict(snapshot['network'])
            self.optimizer.load_state_dict(snapshot['optimizer'])
            self.optimizer.param_groups[0]["lr"] = py.get_learning_rate()
            self.scheduler.load_state_dict(snapshot['scheduler'])

        # for multi-gpu & evaluation
        self.network = nn.DataParallel(self.network)
        self.network.eval()


class PolicyPlay():
    def __init__(self, game_type, conf_file,test_file):
        _temps = __import__(f'build.{game_type}', globals(), locals(), ['minizero_py'], 0)
        py = _temps.minizero_py
        py.load_config_file(conf_file)

        self.env = py.Env()
        self.model = Model()
        #self.load_model('dan_training_new/model/weight_iter_160000.pkl')
        self.load_model('kyu_training_new/model/weight_iter_300000.pkl')
        #self.load_model('kyu_training_newloss2/model/weight_iter_240000.pkl')
        self.data_loader = py.TestDataLoader(conf_file)
        
        self.data_loader.load_test_data_from_file(test_file)
        

    def resetEnv(self):
        self.env.reset()

    def act(self, action):
        return self.env.act(action)

    def load_model(self, nn_file_name):
        self.model.load_model(nn_file_name)

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
            
            features = torch.FloatTensor(features).view(1,py.get_nn_num_input_channels(), py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(self.model.device)
            print(features[0][16][0][0])
            network_outputs = self.model.network(features)

        # apply softmax temperature & play
            
            #print(network_outputs["policy_logit"].shape)
            top_values, top_indices = torch.topk(nn.functional.log_softmax(network_outputs["policy_logit"], dim=1), k=5,dim=1)
            #policy = torch.softmax(network_outputs["policy_logit"][index] / temperature, dim=0).cpu().detach().numpy()
            #print(top_indices)

            top_indices=top_indices.cpu().numpy()
            '''
            chars = 'abcdefghijklmnopqrs'
            coordinates = {k:v for v,k in enumerate(chars)}
            chartonumbers = {k:v for k,v in enumerate(chars)}
            '''
            def number_to_char(number):
                content = chr(ord('a') + number % 19) + \
                        chr(ord('a') + (19 - 1 - number // 19))
                return content
            def top_5_preds_with_chars(predictions):
                resulting_preds_chars = np.vectorize(number_to_char)(predictions)
                return resulting_preds_chars
            prediction_chars = top_5_preds_with_chars(top_indices)  
            
            #print(prediction_chars)
            #print(prediction_chars)
            df = open('TestingDataset_Private/public_private_submission_template.csv').read().splitlines()
            games_id = [k.split(',',2)[0] for k in df]
            with open('best1.csv','a') as f:
                answer_row = games_id[i+32336] + ',' + ','.join(prediction_chars[0]) + '\n'
                
                f.write(answer_row)
            
                

        # return list of generated games
       
        

        


if __name__ == '__main__':
    if len(sys.argv) == 4:
        game_type = sys.argv[1]
        conf_file_name = sys.argv[2]
        eval_data = sys.argv[3]
        
        # import pybind library
        _temps = __import__(f'build.{game_type}', globals(), locals(), ['minizero_py'], 0)
        py = _temps.minizero_py
    else:
        eprint("python train.py game_type conf_file eval.sgf")
        exit(0)
    print(eval_data)
    policy_play = PolicyPlay(game_type, conf_file_name,eval_data)
    policy_play.generate_games()
