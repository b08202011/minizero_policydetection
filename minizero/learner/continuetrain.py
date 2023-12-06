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


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)


class MinizeroDadaLoader:
    def __init__(self, conf_file_name):
        self.data_loader = py.DataLoader(conf_file_name)
        self.data_loader.initialize()
        self.data_list = []

        # allocate memory
        self.sampled_index = np.zeros(py.get_batch_size() * 2, dtype=np.int32)
        self.features = np.zeros(py.get_batch_size() * py.get_nn_num_input_channels() * py.get_nn_input_channel_height() * py.get_nn_input_channel_width(), dtype=np.float32)
        self.loss_scale = np.zeros(py.get_batch_size(), dtype=np.float32)
        self.value_accumulator = np.ones(1) if py.get_nn_discrete_value_size() == 1 else np.arange(-int(py.get_nn_discrete_value_size() / 2), int(py.get_nn_discrete_value_size() / 2) + 1)
        if py.get_nn_type_name() == "alphazero":
            self.action_features = None
            self.policy = np.zeros(py.get_batch_size() * py.get_nn_action_size(), dtype=np.float32)
            self.value = np.zeros(py.get_batch_size() * py.get_nn_discrete_value_size(), dtype=np.float32)
            self.reward = None
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
            
            #if len(self.data_list) > py.get_zero_replay_buffer():
            #    self.data_list.pop(0)

    def sample_data(self, device='cpu'):
        self.data_loader.sample_data(self.features, self.action_features, self.policy, self.value, self.reward, self.loss_scale, self.sampled_index)
        features = torch.FloatTensor(self.features).view(py.get_batch_size(), py.get_nn_num_input_channels(), py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(device)
        action_features = None if self.action_features is None else torch.FloatTensor(self.action_features).view(py.get_batch_size(),
                                                                                                                 -1,
                                                                                                                 py.get_nn_num_action_feature_channels(),
                                                                                                                 py.get_nn_hidden_channel_height(),
                                                                                                                 py.get_nn_hidden_channel_width()).to(device)
        policy = torch.FloatTensor(self.policy).view(py.get_batch_size(), -1, py.get_nn_action_size()).to(device)
        value = torch.FloatTensor(self.value).view(py.get_batch_size(), -1, py.get_nn_discrete_value_size()).to(device)
        reward = None if self.reward is None else torch.FloatTensor(self.reward).view(py.get_batch_size(), -1, py.get_nn_discrete_value_size()).to(device)
        loss_scale = torch.FloatTensor(self.loss_scale / np.amax(self.loss_scale)).to(device)
        sampled_index = self.sampled_index

        return features, action_features, policy, value, reward, loss_scale, sampled_index

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

    def load_model(self, training_dir, model_file):
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
        self.optimizer = optim.SGD(self.network.parameters(),
                                   lr=py.get_learning_rate(),
                                   momentum=py.get_momentum(),
                                   weight_decay=py.get_weight_decay())
        #
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000000, gamma=0.5)  # gamma: 0.5, 
        #print(self.network)
        
        
        if model_file:
            snapshot = torch.load(f"./dan_training_new/model/weight_iter_400000.pkl", map_location=torch.device('cpu'))
            #print(snapshot)
            self.training_step = snapshot['training_step']
            
            self.network = torch.jit.load(f"./dan_training_new/model/weight_iter_400000.pt", map_location=self.device)
            self.network.load_state_dict(snapshot['network'])
            #print(self.network)
            #self.optimizer.load_state_dict(snapshot['optimizer'])
            self.optimizer.param_groups[0]["lr"] = py.get_learning_rate()
            #self.scheduler.load_state_dict(snapshot['scheduler'])
        
        # for multi-gpu
        self.network = nn.DataParallel(self.network)

    def save_model(self, training_dir):
        snapshot = {'training_step': self.training_step,
                    'network': self.network.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}
        torch.save(snapshot, f"{training_dir}/model/weight_iter_{self.training_step}.pkl")
        torch.jit.script(self.network.module).save(f"{training_dir}/model/weight_iter_{self.training_step}.pt")


def calculate_loss(network_output, label_policy, label_value, label_reward, loss_scale):
    # policy
    if py.use_gumbel():
        loss_policy = (nn.functional.kl_div(nn.functional.log_softmax(network_output["policy_logit"], dim=1), label_policy, reduction='none').sum(dim=1) * loss_scale).mean()
    else:
        loss_policy = -((label_policy * nn.functional.log_softmax(network_output["policy_logit"], dim=1)).sum(dim=1) * loss_scale).mean()
        

    return loss_policy
def calculate_loss2(network_output, label_policy, label_value, label_reward, loss_scale):
    # policy
    #print(network_output["policy_logit"].shape)
    top_values, top_indices = torch.topk(network_output["policy_logit"], k=5,dim=1)
    
    #print(label_policy.shape)
    logsoftmax = nn.functional.log_softmax(network_output["policy_logit"], dim=1)
    
    selected_label = torch.gather(label_policy, 1,top_indices)
    one_tensor = torch.ones_like(selected_label)
    selected_logsoftmax = torch.gather(logsoftmax,1,top_indices)
    #print((selected_label*selected_logsoftmax).shape)
    loss_policy2 =  -(torch.max((one_tensor-selected_label)*selected_logsoftmax,dim=1)[0]*loss_scale).mean()
    print(loss_policy2)

    return loss_policy2

def plot_training_curve(training_dir,list,train=True,loss=True):
    plt.figure(0)
    if train:
        plt.xlabel('training step')
    else:
        plt.xlabel('training step')
    if loss:
        plt.ylabel('loss')
    else:
        plt.ylabel('accuracy')
    plt.plot(list)
    plt.savefig(f'{training_dir}/Train{train}loss{loss}.png')
    plt.close()
    


def add_training_info(training_info, key, value):
    if key not in training_info:
        training_info[key] = 0
    training_info[key] += value


def calculate_accuracy(output, label, batch_size):
    max_output = np.argmax(output.to('cpu').detach().numpy(), axis=1)
    max_label = np.argmax(label.to('cpu').detach().numpy(), axis=1)
    return (max_output == max_label).sum() / batch_size


def train(model, training_dir, data_loader, val_loader,training_file,validation_file=None):

    # load data
    data_loader.load_data(training_file)
    if validation_file!=None:
        #print('val')
        val_loader.load_data(validation_file)
    #model.save_model(training_dir)
    print("start training")

    loss_policys =[]
    accuracy_policys =[]
    val_loss =[]
    val_accruacy =[]
    best_accuracy = 0
    early_stop =0
    model.optimizer.param_groups[0]["lr"] *=0.1
    update_step = 70000
    for i in range(300000):
    #while(early_stop<10):
        model.network.train()
        model.optimizer.zero_grad()
        features, action_features, label_policy, label_value, label_reward, loss_scale, sampled_index = data_loader.sample_data(model.device)
        #print(features[:,17,:])
        if py.get_nn_type_name() == "alphazero":
            network_output = model.network(features)
            
        
            #print(nn.functional.log_softmax(network_output["policy_logit"], dim=1))
            #print(network_output["policy_logit"]) 
            loss_policy= calculate_loss(network_output, label_policy[:, 0], label_value[:, 0], None, loss_scale)
            #print(label_policy[:, 0])
            loss_policy += calculate_loss2(network_output, label_policy[:, 0], label_value[:, 0], None, loss_scale) 
            loss_policy += sum((p.pow(2)/(1+p.pow(2))).sum()for p in model.network.parameters())*0.0000001
            loss = loss_policy 

            # record training info
            print('loss_policy:',loss_policy.item())
            print('accuracy_policy:',calculate_accuracy(network_output["policy_logit"], label_policy[:, 0], py.get_batch_size()))
            loss_policys.append(loss_policy.item())
            accuracy_policys.append(calculate_accuracy(network_output["policy_logit"], label_policy[:, 0], py.get_batch_size()))

            

        loss.backward()
        model.optimizer.step()
        model.scheduler.step()
        model.training_step += 1

        #if model.training_step != 0 and model.training_step % py.get_training_display_step() == 0 and validation_file!=None:
        if model.training_step != 0 and model.training_step % py.get_training_display_step() == 0 and validation_file!=None:
            tempaccuracy = 0
            temploss=0
            for j in range(10):
                with torch.no_grad():
                    valloss = 0.0
                    valaccuracy=0.0
                    features, action_features, label_policy, label_value, label_reward, loss_scale, sampled_index = val_loader.sample_data(model.device)
                    if py.get_nn_type_name() == "alphazero":
                        network_output = model.network(features) 
                        valloss= calculate_loss(network_output, label_policy[:, 0], label_value[:, 0], None, loss_scale)
                        print('val_loss_policy:',valloss.item())
                        valaccuracy = calculate_accuracy(network_output["policy_logit"], label_policy[:, 0], py.get_batch_size())
                        print('val_accuracy_policy:',valaccuracy)
                        
                        val_loss.append(valloss.item())
                        val_accruacy.append(valaccuracy)
            

        if model.training_step != 0 and model.training_step % update_step== 0:
            model.optimizer.param_groups[0]["lr"] *=0.5
            update_step/=2

        
        if model.training_step != 0 and model.training_step % 10000== 0:
            model.save_model(training_dir)
            #print("Optimization_Done", model.training_step, flush=True)
            plot_training_curve(training_dir,loss_policys,True,True)
            plot_training_curve(training_dir,accuracy_policys,True,False)
            plot_training_curve(training_dir,val_loss,False,True)
            plot_training_curve(training_dir,val_accruacy,False,False) 
    #eprint("Optimization_Done", model.training_step)
    #analysis(training_dir, "analysis")


if __name__ == '__main__':
    print('start')
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        game_type = sys.argv[1]
        training_dir = sys.argv[2]
        conf_file_name = sys.argv[3]
        training_file = sys.argv[4]
        validation_file = None
        if len(sys.argv) == 6: 
            validation_file = sys.argv[5]
    
        # import pybind library
        _temps = __import__(f'build.{game_type}', globals(), locals(), ['minizero_py'], 0)
        py = _temps.minizero_py
    else:
        eprint("python train.py game_type training_dir conf_file training_file")
        exit(0)

    py.load_config_file(conf_file_name)
    data_loader = MinizeroDadaLoader(conf_file_name)
    val_loader = MinizeroDadaLoader(conf_file_name) 
    model = Model()
    model_file = f'weight_iter_100000.pkl'
                # skip loading model if the model is loaded
    
    if model.network is None:
        model.load_model(training_dir, model_file)
    
    train(model, training_dir, data_loader,val_loader, training_file,validation_file)
            
