// EIE: bounded per-request hard top-k statistics. Apache-2.0.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace eie {
enum class RoutingPhase { Prefill = 0, Decode = 1 };
struct ExpertAccess { uint64_t selected = 0, hits = 0, misses = 0; };
struct PhaseRouting {
    uint64_t callbacks = 0, first_accesses = 0;
    std::vector<ExpertAccess> experts;
    std::vector<uint64_t> reuse_distance; // d distinct intervening experts; LRU hit iff d < capacity.
};
struct LayerRouting {
    uint64_t expert_weight_bytes = 0;
    std::array<PhaseRouting, 2> phases;
    std::vector<int> stack; // MRU first; shared across phases, reset at request boundary.
};

class RoutingHistogram {
    bool active_ = false;
    RoutingPhase phase_ = RoutingPhase::Prefill;
public:
    bool supported = false, enabled = false;
    uint64_t request_sequence = 0;
    int expert_count = 0, top_k = 0, slots = 0;
    std::string outcome = "not_started";
    std::map<int, LayerRouting> layers;

    void begin(bool enable, int experts, int k, int cache_slots,
               const std::map<int, uint64_t>& layer_bytes) {
        supported = true; enabled = active_ = enable; ++request_sequence;
        expert_count = experts; top_k = k; slots = cache_slots;
        outcome = enable ? "running" : "disabled";
        phase_ = RoutingPhase::Prefill;
        layers.clear();
        if (!enable) return;
        for (const auto& entry : layer_bytes) {
            auto& layer = layers[entry.first];
            layer.expert_weight_bytes = entry.second;
            layer.stack.reserve(experts);
            for (auto& phase : layer.phases) {
                phase.experts.resize(experts);
                phase.reuse_distance.resize(experts);
            }
        }
    }
    void phase(RoutingPhase phase) { phase_ = phase; }
    void end(const std::string& result) { if (enabled) outcome = result; active_ = false; }
    void callback(int layer) {
        if (active_) ++layers.at(layer).phases[static_cast<int>(phase_)].callbacks;
    }
    void access(int layer_id, int expert, bool hit) {
        if (!active_) return;
        auto& layer = layers.at(layer_id);
        auto& phase = layer.phases[static_cast<int>(phase_)];
        auto& counter = phase.experts.at(expert);
        ++counter.selected;
        if (hit) ++counter.hits; else ++counter.misses;
        auto it = std::find(layer.stack.begin(), layer.stack.end(), expert);
        if (it == layer.stack.end()) {
            ++phase.first_accesses;
        } else {
            ++phase.reuse_distance[static_cast<size_t>(it - layer.stack.begin())];
            layer.stack.erase(it);
        }
        layer.stack.insert(layer.stack.begin(), expert);
    }
    std::string json() const {
        std::ostringstream out;
        out << "{\"schema\":\"eie.routing/v1\",\"supported\":" << (supported ? "true" : "false")
            << ",\"enabled\":" << (enabled ? "true" : "false")
            << ",\"request_sequence\":" << request_sequence << ",\"outcome\":\"" << outcome
            << "\",\"expert_count\":" << expert_count << ",\"top_k\":" << top_k
            << ",\"slots\":" << slots << ",\"layers\":[";
        bool comma = false;
        for (const auto& entry : layers) {
            if (comma) out << ',';
            comma = true;
            out << "{\"layer\":" << entry.first << ",\"expert_weight_bytes\":" << entry.second.expert_weight_bytes;
            for (int p = 0; p < 2; ++p) {
                const auto& phase = entry.second.phases[p];
                out << ",\"" << (p == 0 ? "prefill" : "decode") << "\":{\"callbacks\":" << phase.callbacks
                    << ",\"first_accesses\":" << phase.first_accesses << ",\"experts\":[";
                bool ec = false;
                for (size_t id = 0; id < phase.experts.size(); ++id) {
                    const auto& c = phase.experts[id];
                    if (!c.selected) continue;
                    if (ec) out << ',';
                    ec = true;
                    out << "{\"expert\":" << id << ",\"selected\":" << c.selected
                        << ",\"hits\":" << c.hits << ",\"misses\":" << c.misses << '}';
                }
                out << "],\"reuse_distance\":[";
                for (size_t d = 0; d < phase.reuse_distance.size(); ++d) {
                    if (d) out << ',';
                    out << phase.reuse_distance[d];
                }
                out << "]}";
            }
            out << '}';
        }
        out << "]}";
        return out.str();
    }
};
} // namespace eie
