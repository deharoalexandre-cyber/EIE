// EIE — Model Manager
// Apache License 2.0
#pragma once
#include "../backends/compute_backend.h"
#include "scheduling.h"
#include <map>
#include <filesystem>
#include <iostream>

namespace eie {

class ModelManager {
    std::map<std::string, std::string> registry_; // alias -> path
    std::map<std::string, std::unique_ptr<ComputeBackend>> backends_;
    std::map<std::string, Slot> slots_;
public:
    void reg(const std::string& alias, const std::string& path) {
        registry_[alias] = path;
        std::cout << "[Models] registered: " << alias << std::endl;
    }

    void discover(const std::string& dir) {
        if (!std::filesystem::exists(dir)) {
            std::cout << "[Models] directory not found: " << dir << std::endl;
            return;
        }
        for (auto& e : std::filesystem::directory_iterator(dir)) {
            if (e.path().extension() == ".gguf")
                reg(e.path().stem().string(), e.path().string());
        }
    }

    bool load(const std::string& alias, const KvConfig& kv, int gpu_id = 0) {
        auto it = registry_.find(alias);
        if (it == registry_.end()) {
            std::cerr << "[Models] unknown: " << alias << std::endl;
            return false;
        }
        auto backend = detectBackend(gpu_id);
        if (!backend) return false;
        ModelParams p;
        p.path = it->second;
        p.alias = alias;
        p.kv = kv;
        if (!backend->load(p)) return false;
        Slot s;
        s.alias = alias;
        s.kv = kv;
        slots_[alias] = s;
        backends_[alias] = std::move(backend);
        return true;
    }

    void unloadModel(const std::string& alias) {
        auto it = backends_.find(alias);
        if (it != backends_.end()) {
            it->second->unload();
            backends_.erase(it);
            slots_.erase(alias);
        }
    }

    ComputeBackend* get(const std::string& alias) {
        auto it = backends_.find(alias);
        return it != backends_.end() ? it->second.get() : nullptr;
    }

    std::map<std::string, ComputeBackend*> getAll() {
        std::map<std::string, ComputeBackend*> m;
        for (auto& [k, v] : backends_) m[k] = v.get();
        return m;
    }

    std::vector<std::string> loaded() {
        std::vector<std::string> r;
        for (auto& [k, v] : backends_)
            if (v->loaded) r.push_back(k);
        return r;
    }

    std::vector<std::string> available() {
        std::vector<std::string> r;
        for (auto& [k, _] : registry_) r.push_back(k);
        return r;
    }

    std::string findEvictCandidate() {
        std::string best;
        int64_t oldest = INT64_MAX;
        for (auto& [k, s] : slots_) {
            if (!s.pinned && s.last_used < oldest) {
                oldest = s.last_used;
                best = k;
            }
        }
        return best;
    }
};

} // namespace eie
