#include "data_loader.h"
#include "configuration.h"
#include "environment.h"
#include "rotation.h"
#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

namespace minizero::learner {

using namespace minizero;
using namespace minizero::utils;

DataLoader::DataLoader(std::string conf_file_name)
{
    env::setUpEnv();
    config::ConfigureLoader cl;
    config::setConfiguration(cl);
    cl.loadFromFile(conf_file_name);
}

void DataLoader::loadDataFromFile(const std::string& file_name)
{
    std::ifstream fin(file_name, std::ifstream::in);
    for (std::string content; std::getline(fin, content);) {
        EnvironmentLoader env_loader;
        if (!env_loader.loadFromString(content)) { continue; }
        int total_length = env_loaders_.empty() ? 0 : env_loaders_.back().second;
        env_loaders_.push_back({env_loader, total_length + env_loader.getActionPairs().size()});
    }
}

AlphaZeroData DataLoader::getAlphaZeroTrainingData()
{
    // random pickup one position
    std::pair<int, int> p = getEnvIDAndPosition(Random::randInt() % getDataSize());
    int env_id = p.first, pos = p.second;

    // replay the game until to the selected position
    const EnvironmentLoader& env_loader = env_loaders_[env_id].first;
    env_.reset();
    for (int i = 0; i < pos; ++i) { env_.act(env_loader.getActionPairs()[i].first); }

    // calculate training data
    AlphaZeroData data;
    Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
    data.features_ = env_.getFeatures(rotation);
    data.policy_ = getPolicyDistribution(env_loader, pos, rotation);
    data.value_ = env_loader.getReturn();

    return data;
}

MuZeroData DataLoader::getMuZeroTrainingData(int unrolling_step)
{
    // random pickup one position
    std::pair<int, int> p = getEnvIDAndPosition(Random::randInt() % getDataSize());
    int env_id = p.first, pos = p.second;
    while (pos + unrolling_step >= static_cast<int>(env_loaders_[env_id].first.getActionPairs().size())) {
        // random again until we can unroll all steps
        p = getEnvIDAndPosition(Random::randInt() % getDataSize());
        env_id = p.first, pos = p.second;
    }

    // replay the game until to the selected position
    const EnvironmentLoader& env_loader = env_loaders_[env_id].first;
    env_.reset();
    for (int i = 0; i < pos; ++i) { env_.act(env_loader.getActionPairs()[i].first); }

    // calculate training data
    MuZeroData data;
    Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
    data.features_ = env_.getFeatures(rotation);
    for (int step = 0; step <= unrolling_step; ++step) {
        const Action& action = env_loader.getActionPairs()[pos + step].first;
        std::vector<float> action_features = env_.getActionFeatures(action, rotation);
        std::vector<float> policy = getPolicyDistribution(env_loader, pos + step, rotation);
        if (step < unrolling_step) { data.action_features_.insert(data.action_features_.end(), action_features.begin(), action_features.end()); }
        data.policy_.insert(data.policy_.end(), policy.begin(), policy.end());
        env_.act(action);
    }
    data.value_ = env_loader.getReturn();
    return data;
}

std::pair<int, int> DataLoader::getEnvIDAndPosition(int index) const
{
    int left = 0, right = env_loaders_.size();
    index %= env_loaders_.back().second;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (index >= env_loaders_[mid].second) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return {left, (left == 0 ? index : index - env_loaders_[left - 1].second)};
}

std::vector<float> DataLoader::getPolicyDistribution(const EnvironmentLoader& env_loader, int pos, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/)
{
    std::vector<float> policy(env_loader.getPolicySize(), 0.0f);
    const std::string& distribution = env_loader.getActionPairs()[pos].second;
    if (distribution.empty()) {
        const Action& action = env_loader.getActionPairs()[pos].first;
        policy[env_loader.getRotatePosition(action.getActionID(), rotation)] = 1.0f;
    } else {
        float sum = 0.0f;
        std::string tmp;
        std::istringstream iss(distribution);
        while (std::getline(iss, tmp, ',')) {
            int position = env_loader.getRotatePosition(std::stoi(tmp.substr(0, tmp.find(":"))), rotation);
            float count = std::stof(tmp.substr(tmp.find(":") + 1));
            policy[position] = count;
            sum += count;
        }
        for (auto& p : policy) { p /= sum; }
    }
    return policy;
}

} // namespace minizero::learner
