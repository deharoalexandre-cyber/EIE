// EIE — VRAM Manager
// Apache License 2.0
#pragma once
#include "../backends/compute_backend.h"
#include <map>
#include <iostream>

namespace eie {

struct VramConfig {
    size_t reserve_mb = 512;
    float low_wm = 0.85f, crit_wm = 0.95f;
    bool group_isolation = false;
    std::map<std::string, size_t> group_budgets_mb;
};

class VramManager {
    VramConfig cfg_;
    std::map<int, ComputeBackend*> gpus_;
public:
    VramManager(VramConfig c = {}) : cfg_(c) {}
    void addGpu(int id, ComputeBackend* b) { gpus_[id] = b; }

    enum Pressure { NORMAL, LOW, CRITICAL };
    Pressure check(int gpu = 0) {
        auto it = gpus_.find(gpu);
        if (it == gpus_.end()) return NORMAL;
        float u = it->second->vram().util();
        if (u >= cfg_.crit_wm) return CRITICAL;
        if (u >= cfg_.low_wm) return LOW;
        return NORMAL;
    }

    bool canLoad(size_t bytes, int gpu = 0) {
        auto it = gpus_.find(gpu);
        if (it == gpus_.end()) return false;
        auto v = it->second->vram();
        return v.free_bytes > bytes + cfg_.reserve_mb * 1024 * 1024;
    }

    void report() {
        for (auto& [id, b] : gpus_) {
            auto v = b->vram();
            std::cout << "[VRAM] GPU " << id << ": "
                      << v.used_bytes / (1 << 20) << "/" << v.total_bytes / (1 << 20)
                      << " MB (" << (int)(v.util() * 100) << "%)" << std::endl;
        }
    }
};

} // namespace eie
