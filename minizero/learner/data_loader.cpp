#include "data_loader.h"
#include "configuration.h"
#include "environment.h"
#include "random.h"
#include "rotation.h"
#include <algorithm>
#include <fstream>
#include <utility>
#include <iostream>

namespace minizero::learner {

using namespace minizero;
using namespace minizero::utils;

ReplayBuffer::ReplayBuffer()
{
    num_data_ = 0;
    game_priority_sum_ = 0.0f;
    game_priorities_.clear();
    position_priorities_.clear();
    env_loaders_.clear();
    game_counter_ = 0;
}

void ReplayBuffer::addData(const EnvironmentLoader& env_loader)
{
    std::pair<int, int> data_range = env_loader.getDataRange();
    //std::cout<<data_range<<std::endl;
    std::deque<float> position_priorities(data_range.second + 1, 0.0f);
    float game_priority = 0.0f;
    for (int i = data_range.first; i <= data_range.second; ++i) {
        position_priorities[i] = std::pow((config::learner_use_per ? env_loader.getPriority(i) : 1.0f), config::learner_per_alpha);
        game_priority += position_priorities[i];
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // add new data to replay buffer
    num_data_ += (data_range.second - data_range.first + 1);
    position_priorities_.push_back(position_priorities);
    game_priorities_.push_back(game_priority);
    env_loaders_.push_back(env_loader);

    // remove old data if replay buffer is full
    const size_t replay_buffer_max_size = config::zero_replay_buffer * config::zero_num_games_per_iteration;
    while (position_priorities_.size() > replay_buffer_max_size) {
        data_range = env_loaders_.front().getDataRange();
        num_data_ -= (data_range.second - data_range.first + 1);
        position_priorities_.pop_front();
        game_priorities_.pop_front();
        env_loaders_.pop_front();
    }
}

std::pair<int, int> ReplayBuffer::sampleEnvAndPos()
{
    //std::lock_guard<std::mutex> lock(mutex_);
    //int env_id =0;
    int env_id = sampleIndex(game_priorities_);
    //std::cout<<position_priorities_[game_counter_].size()<<std::endl;
    int pos_id = sampleIndex(position_priorities_[env_id]);
    //int pos_id = position_priorities_[env_id].size()-1;
    //std::cout<< game_counter_<<std::endl;
    //std::cout<< pos_id<<std::endl; 
    return {env_id, pos_id};
}

int ReplayBuffer::sampleIndex(const std::deque<float>& weight)
{
    std::discrete_distribution<> dis(weight.begin(), weight.end());
    //std::lock_guard<std::mutex> lock(mutex_);
    //int temp =game_counter_;

    //game_counter_+=1;
    //game_counter_=game_counter_%game_priorities_.size();
    //std::cout<<game_priorities_.size()<<std::endl;
    return dis(Random::generator_);
    //return temp;
}

float ReplayBuffer::getLossScale(const std::pair<int, int>& p)
{
    if (!config::learner_use_per) { return 1.0f; }

    // calculate importance sampling ratio
    int env_id = p.first, pos = p.second;
    float prob = position_priorities_[env_id][pos] / game_priority_sum_;
    return std::pow((num_data_ * prob), (-config::learner_per_init_beta));
}

std::string DataLoaderSharedData::getNextEnvString()
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string env_string = "";
    if (!env_strings_.empty()) {
        env_string = env_strings_.front();
        //std::cout<< env_string <<std::endl;
        env_strings_.pop_front();
    }
    
    return env_string;
}

int DataLoaderSharedData::getNextBatchIndex()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return (batch_index_ < config::learner_batch_size ? batch_index_++ : config::learner_batch_size);
}


void DataLoaderThread::initialize()
{
    int seed = config::program_auto_seed ? std::random_device()() : config::program_seed + id_;
    Random::seed(seed);
}

void DataLoaderThread::runJob()
{
    if (!getSharedData()->env_strings_.empty()) {
        while (addEnvironmentLoader()) {}
    } else {
        while (sampleData()) {}
    }
}

bool DataLoaderThread::addEnvironmentLoader()
{
    std::string env_string = getSharedData()->getNextEnvString();
    if (env_string.empty()) { return false; }
    //todo  modify envloader
    EnvironmentLoader env_loader;
    if (env_loader.loadFromString(env_string)) { getSharedData()->replay_buffer_.addData(env_loader); }
    return true;
}

bool DataLoaderThread::sampleData()
{   
    //std::lock_guard<std::mutex> lock(mutex_); 
    int batch_index = getSharedData()->getNextBatchIndex();
    if (batch_index >= config::learner_batch_size) { return false; }

    if (config::nn_type_name == "alphazero") {
        setAlphaZeroTrainingData(batch_index);
    } else if (config::nn_type_name == "muzero") {
        setMuZeroTrainingData(batch_index);
    } else {
        return false; // should not be here
    }

    return true;
}

void DataLoaderThread::setAlphaZeroTrainingData(int batch_index)
{
    
    // random pickup one position
    std::pair<int, int> p = getSharedData()->replay_buffer_.sampleEnvAndPos();
    int env_id = p.first, pos = p.second;

    // AlphaZero training data
    const EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
    Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
    float loss_scale = getSharedData()->replay_buffer_.getLossScale(p);
    std::vector<float> features = env_loader.getFeatures(pos, rotation);
    std::vector<float> policy = env_loader.getPolicy(pos, rotation);
    std::vector<float> value = env_loader.getValue(pos);
    //add player name

    // write data to data_ptr
    getSharedData()->getDataPtr()->loss_scale_[batch_index] = loss_scale;
    getSharedData()->getDataPtr()->sampled_index_[2 * batch_index] = p.first;
    getSharedData()->getDataPtr()->sampled_index_[2 * batch_index + 1] = p.second;
    std::copy(features.begin(), features.end(), getSharedData()->getDataPtr()->features_ + features.size() * batch_index);
    std::copy(policy.begin(), policy.end(), getSharedData()->getDataPtr()->policy_ + policy.size() * batch_index);
    std::copy(value.begin(), value.end(), getSharedData()->getDataPtr()->value_ + value.size() * batch_index);
    //add player name
}

void DataLoaderThread::setMuZeroTrainingData(int batch_index)
{
    // random pickup one position
    std::pair<int, int> p = getSharedData()->replay_buffer_.sampleEnvAndPos();
    int env_id = p.first, pos = p.second;

    // MuZero training data
    const EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
    Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
    float loss_scale = getSharedData()->replay_buffer_.getLossScale(p);
    std::vector<float> features = env_loader.getFeatures(pos, rotation);
    std::vector<float> action_features, policy, value, reward, tmp;
    for (int step = 0; step <= config::learner_muzero_unrolling_step; ++step) {
        // action features
        if (step < config::learner_muzero_unrolling_step) {
            tmp = env_loader.getActionFeatures(pos + step, rotation);
            action_features.insert(action_features.end(), tmp.begin(), tmp.end());
        }

        // policy
        tmp = env_loader.getPolicy(pos + step, rotation);
        policy.insert(policy.end(), tmp.begin(), tmp.end());

        // value
        tmp = env_loader.getValue(pos + step);
        value.insert(value.end(), tmp.begin(), tmp.end());

        // reward
        if (step < config::learner_muzero_unrolling_step) {
            tmp = env_loader.getReward(pos + step);
            reward.insert(reward.end(), tmp.begin(), tmp.end());
        }
    }

    // write data to data_ptr
    getSharedData()->getDataPtr()->loss_scale_[batch_index] = loss_scale;
    getSharedData()->getDataPtr()->sampled_index_[2 * batch_index] = p.first;
    getSharedData()->getDataPtr()->sampled_index_[2 * batch_index + 1] = p.second;
    std::copy(features.begin(), features.end(), getSharedData()->getDataPtr()->features_ + features.size() * batch_index);
    std::copy(action_features.begin(), action_features.end(), getSharedData()->getDataPtr()->action_features_ + action_features.size() * batch_index);
    std::copy(policy.begin(), policy.end(), getSharedData()->getDataPtr()->policy_ + policy.size() * batch_index);
    std::copy(value.begin(), value.end(), getSharedData()->getDataPtr()->value_ + value.size() * batch_index);
    std::copy(reward.begin(), reward.end(), getSharedData()->getDataPtr()->reward_ + reward.size() * batch_index);
}

DataLoader::DataLoader(const std::string& conf_file_name)
{
    env::setUpEnv();
    config::ConfigureLoader cl;
    config::setConfiguration(cl);
    cl.loadFromFile(conf_file_name);
}

void DataLoader::initialize()
{
    createSlaveThreads(config::learner_num_thread);
    getSharedData()->createDataPtr();
}

void DataLoader::loadDataFromFile(const std::string& file_name)
{
    std::ifstream fin(file_name, std::ifstream::in);
    for (std::string content; std::getline(fin, content);) { getSharedData()->env_strings_.push_back(content); }

    for (auto& t : slave_threads_) { t->start(); }
    for (auto& t : slave_threads_) { t->finish(); }
    getSharedData()->replay_buffer_.game_priority_sum_ = std::accumulate(getSharedData()->replay_buffer_.game_priorities_.begin(), getSharedData()->replay_buffer_.game_priorities_.end(), 0.0f);
}

void DataLoader::sampleData()
{
    getSharedData()->batch_index_ = 0;
    for (auto& t : slave_threads_) { t->start(); }
    for (auto& t : slave_threads_) { t->finish(); }
}

void DataLoader::updatePriority(int* sampled_index, float* batch_values)
{
    // TODO: use multiple threads
    for (int batch_index = 0; batch_index < config::learner_batch_size; ++batch_index) {
        int env_id = sampled_index[2 * batch_index];
        int pos_id = sampled_index[2 * batch_index + 1];

        EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
        for (int step = 0; step <= config::learner_muzero_unrolling_step; ++step) {
            float new_value = utils::invertValue(batch_values[step * config::learner_batch_size + batch_index]);
            env_loader.setActionPairInfo(pos_id + step, "V", std::to_string(new_value));
        }
        getSharedData()->replay_buffer_.position_priorities_[env_id][pos_id] = std::pow(env_loader.getPriority(pos_id), config::learner_per_alpha);
    }

    // recalculate priority to correct floating number error (TODO: speedup this)
    for (size_t i = 0; i < getSharedData()->replay_buffer_.game_priorities_.size(); ++i) {
        getSharedData()->replay_buffer_.game_priorities_[i] = std::accumulate(getSharedData()->replay_buffer_.position_priorities_[i].begin(), getSharedData()->replay_buffer_.position_priorities_[i].end(), 0.0f);
    }
    getSharedData()->replay_buffer_.game_priority_sum_ = std::accumulate(getSharedData()->replay_buffer_.game_priorities_.begin(), getSharedData()->replay_buffer_.game_priorities_.end(), 0.0f);
}


TestDataLoader::TestDataLoader(std::string conf_file)
{
    minizero::config::ConfigureLoader cl;
    minizero::config::setConfiguration(cl);
    cl.loadFromFile(conf_file);
    minizero::env::setUpEnv();
    env_loaders_.clear();
}
void TestDataLoader::loadTestDataFromFile(const std::string& file_name)
{
    std::cerr << "Read " << file_name << "..." << std::endl;
    std::ifstream fin(file_name, std::ifstream::in);
    
    for (std::string content; std::getline(fin, content);) {
        minizero::utils::SGFLoader sgf_loader;
        if (!sgf_loader.loadFromString(content)) { continue; }
        if (std::stoi(sgf_loader.getTags().at("SZ")) != 19) { continue; }

        EnvironmentLoader env_loader;
        env_loader.reset();
        env_loader.addTag("SZ", sgf_loader.getTags().at("SZ"));
        //env_loader.addTag("KM", sgf_loader.getTags().at("KM"));
        //env_loader.addTag("PB", sgf_loader.getTags().at("PB"));
        //env_loader.addTag("PW", sgf_loader.getTags().at("PW"));
        //env_loader.addTag("BR", sgf_loader.getTags().at("BR"));
        //env_loader.addTag("WR", sgf_loader.getTags().at("WR"));
        for (auto& action_string : sgf_loader.getActions()) {
            env_loader.addActionPair(Action(action_string.first, std::stoi(sgf_loader.getTags().at("SZ"))), action_string.second);
        }
        //std::cerr<<"add data"<<std::endl;
        //std::string PB_name = env_loader.getTag("PB");
        //std::string PW_name = env_loader.getTag("PW");
        //std::cerr <<PB_name <<PW_name<<std::endl;
        
        env_loaders_.push_back(env_loader);
        //std::cerr<<env_loaders_.size() <<std::endl;
        //} else if (player_name == PW_name){
        
    }
}
std::vector<float> TestDataLoader::calculateGameFeatures(int game_id)
{
    std::vector<float> features_;
    std::vector<float> game_features_;
    
    game_features_.clear();
 
    env_.reset();

    int move_counts = env_loaders_[game_id].getActionPairs().size();
    //std::string PB_name = env_loaders_[player][game_id].getTag("PB");
    //minizero::env::Player color = !PB_name.empty() ? minizero::env::Player::kPlayer1 : minizero::env::Player::kPlayer2;
  
    for (int i = 0; i < move_counts; ++i) {
        const Action& action = env_loaders_[game_id].getActionPairs()[i].first;
        env_.act(action);
        if (i ==  move_counts-1) {
            features_ = env_.getFeatures(); // (1,18*19*19)
            for (size_t j = 0; j < features_.size(); j++) {
                game_features_.push_back(features_[j]); // (n_frames,18*19*19)
            }
                
        }
    }
    

    return game_features_;
}
}
 // namespace minizero::learner
