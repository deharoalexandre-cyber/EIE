#include "backends/routing_histogram.h"
#include <iostream>
#include <random>
#include <stdexcept>

static int checks = 0;
static void expect(bool ok) { ++checks; if (!ok) throw std::runtime_error("routing assertion failed"); }
int main() {
    eie::RoutingHistogram h;
    expect(!h.supported);
    h.begin(false, 288, 8, 8, {{1, 14000000}});
    h.callback(1); h.access(1, 287, false);
    expect(h.layers.empty() && !h.enabled);
    h.begin(true, 288, 8, 8, {{1, 14000000}, {2, 14000000}});
    h.callback(1);
    for (int i = 280; i < 288; ++i) h.access(1, i, false);
    h.phase(eie::RoutingPhase::Decode);
    h.callback(1);
    for (int i = 280; i < 288; ++i) h.access(1, i, true);
    expect(h.layers.at(1).phases[0].first_accesses == 8);
    expect(h.layers.at(1).phases[1].first_accesses == 0);
    expect(h.layers.at(1).phases[1].reuse_distance[7] == 8);
    expect(h.layers.at(1).phases[1].experts[287].hits == 1);
    expect(h.layers.at(2).stack.empty());
    h.end("cancelled");
    h.access(1, 287, true);
    expect(h.layers.at(1).phases[1].experts[287].selected == 1);
    expect(h.outcome == "cancelled" && h.request_sequence == 2);

    // Independent direct LRU simulation, every capacity, seeded mixed accesses.
    std::mt19937 rng(71);
    const int n = 32, length = 10000;
    h.begin(true, n, 1, 8, {{0, 128}});
    std::vector<std::vector<int>> caches(n + 1);
    std::vector<uint64_t> hits(n + 1);
    for (int t = 0; t < length; ++t) {
        int id = rng() % (t % 3 ? 6 : n);
        if (t == 5000) h.phase(eie::RoutingPhase::Decode);
        h.callback(0); h.access(0, id, false);
        for (int capacity = 1; capacity <= n; ++capacity) {
            auto& cache = caches[capacity];
            auto it = std::find(cache.begin(), cache.end(), id);
            if (it != cache.end()) { ++hits[capacity]; cache.erase(it); }
            cache.insert(cache.begin(), id);
            if (cache.size() > size_t(capacity)) cache.pop_back();
        }
    }
    uint64_t cumulative = 0;
    for (int capacity = 1; capacity <= n; ++capacity) {
        for (const auto& phase : h.layers.at(0).phases) cumulative += phase.reuse_distance[capacity - 1];
        expect(cumulative == hits[capacity]);
    }
    h.end("complete");
    std::cout << "PASS " << checks << " routing assertions; 10000 accesses, 32 independent LRU capacities\n";
}
