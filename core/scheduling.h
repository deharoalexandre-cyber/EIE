// EIE — Scheduling: Policy Engine + Group Scheduler
// Apache License 2.0
#pragma once

#include "../backends/compute_backend.h"
#include <functional>
#include <future>
#include <thread>
#include <algorithm>
#include <iostream>
#include <chrono>

namespace eie {

// ═══════════════════════════════════════════
// Data Structures
// ═══════════════════════════════════════════

struct GroupConfig {
    std::string name;
    std::vector<std::string> models;
    int required = 1;
    bool pinned = false;
    std::string fallback = "strict";
    std::string replacement;
    std::string type = "parallel"; // parallel/sequential/fanout/standard
    float max_latency_ms = 5000;
    KvConfig kv_override;
};

struct GroupResult {
    std::string group;
    std::vector<InferenceResult> responses;
    int completed = 0, required = 0;
    std::string status;
    float latency_ms = 0;
};

struct Slot {
    std::string alias, group;
    bool pinned = false;
    int64_t last_used = 0;
    KvConfig kv;
};

// ═══════════════════════════════════════════
// PolicyStrategy — THE pluggable interface
// ═══════════════════════════════════════════

class PolicyStrategy {
public:
    virtual ~PolicyStrategy() = default;
    virtual std::string name() const = 0;
    virtual bool shouldEvict(const Slot& slot, const VramStatus& v) = 0;
    virtual int requiredResponses(const std::string& group) = 0;
    virtual KvConfig adaptKv(const Slot& slot, const HealthStatus& h) = 0;

    enum class Fallback { FAIL, PARTIAL, RETRY, REPLACE };
    virtual Fallback onFailure(const std::string& group, int completed, int required) = 0;

    std::map<std::string, GroupConfig> groups;
};

// ═══════════════════════════════════════════
// Built-in Strategy: Generic (Ollama-like)
// ═══════════════════════════════════════════

class GenericStrategy : public PolicyStrategy {
public:
    std::string name() const override { return "generic"; }

    bool shouldEvict(const Slot& s, const VramStatus& v) override {
        return !s.pinned && v.util() > 0.85f;
    }

    int requiredResponses(const std::string& g) override {
        auto it = groups.find(g);
        return it != groups.end() ? it->second.required : 1;
    }

    KvConfig adaptKv(const Slot& s, const HealthStatus& h) override {
        return s.kv; // no adaptation
    }

    Fallback onFailure(const std::string&, int, int) override {
        return Fallback::FAIL;
    }
};

// ═══════════════════════════════════════════
// Built-in Strategy: Pinned Group
// ═══════════════════════════════════════════

class PinnedGroupStrategy : public PolicyStrategy {
public:
    std::string name() const override { return "pinned-group"; }

    bool shouldEvict(const Slot& s, const VramStatus& v) override {
        return !s.pinned && v.util() > 0.85f;
    }

    int requiredResponses(const std::string& g) override {
        auto it = groups.find(g);
        return it != groups.end() ? it->second.required : 1;
    }

    Fallback onFailure(const std::string& g, int c, int r) override {
        auto it = groups.find(g);
        if (it == groups.end()) return Fallback::FAIL;
        auto& fb = it->second.fallback;
        if (fb == "partial") return Fallback::PARTIAL;
        if (fb == "retry_once") return Fallback::RETRY;
        if (fb == "replace_with") return Fallback::REPLACE;
        return Fallback::FAIL;
    }

    KvConfig adaptKv(const Slot& s, const HealthStatus& h) override {
        // Health-check: downgrade KV if latency too high
        if (h.latency_ms > 5000 && s.kv.type_v == "turbo3") {
            KvConfig d = s.kv;
            d.type_v = "turbo2";
            std::cout << "[Policy] health-check: downgrade " << s.alias
                      << " turbo3->turbo2 (lat=" << h.latency_ms << "ms)" << std::endl;
            return d;
        }
        if (h.latency_ms > 3000 && s.kv.type_v == "turbo4") {
            KvConfig d = s.kv;
            d.type_v = "turbo3";
            std::cout << "[Policy] health-check: downgrade " << s.alias
                      << " turbo4->turbo3 (lat=" << h.latency_ms << "ms)" << std::endl;
            return d;
        }
        return s.kv;
    }
};

// ═══════════════════════════════════════════
// Built-in Strategy: Multi-Group
// ═══════════════════════════════════════════

class MultiGroupStrategy : public PinnedGroupStrategy {
public:
    std::string name() const override { return "multi-group"; }
};

// ═══════════════════════════════════════════
// Built-in Strategy: Fixed Appliance
// ═══════════════════════════════════════════

class FixedStrategy : public PolicyStrategy {
public:
    std::string name() const override { return "fixed-appliance"; }
    bool shouldEvict(const Slot&, const VramStatus&) override { return false; }
    int requiredResponses(const std::string& g) override {
        auto it = groups.find(g);
        return it != groups.end() ? it->second.required : 1;
    }
    KvConfig adaptKv(const Slot& s, const HealthStatus&) override { return s.kv; }
    Fallback onFailure(const std::string&, int, int) override { return Fallback::PARTIAL; }
};

// ═══════════════════════════════════════════
// Strategy Factory
// ═══════════════════════════════════════════

inline std::unique_ptr<PolicyStrategy> createStrategy(const std::string& n) {
    if (n == "generic") return std::make_unique<GenericStrategy>();
    if (n == "pinned-group") return std::make_unique<PinnedGroupStrategy>();
    if (n == "multi-group") return std::make_unique<MultiGroupStrategy>();
    if (n == "fixed-appliance") return std::make_unique<FixedStrategy>();
    // Plugin support: if (n.substr(0,7) == "plugin:") { dlopen... }
    std::cerr << "[EIE] unknown strategy '" << n << "', using generic" << std::endl;
    return std::make_unique<GenericStrategy>();
}

// ═══════════════════════════════════════════
// Group Scheduler
// ═══════════════════════════════════════════

class GroupScheduler {
    PolicyStrategy* policy_;
public:
    GroupScheduler(PolicyStrategy* p) : policy_(p) {}

    // ── Parallel: same prompt to N models simultaneously ──
    GroupResult execParallel(const GroupConfig& g, const std::string& prompt,
                            const SamplingParams& sp,
                            std::map<std::string, ComputeBackend*>& backends) {
        auto t0 = std::chrono::steady_clock::now();
        GroupResult r;
        r.group = g.name;
        r.required = g.required;

        std::vector<std::future<InferenceResult>> futs;
        for (auto& alias : g.models) {
            auto it = backends.find(alias);
            if (it == backends.end() || !it->second->loaded) continue;

            // Health-check pre-inference
            auto h = it->second->health();
            if (h.latency_ms > g.max_latency_ms) {
                auto kv = policy_->adaptKv(
                    Slot{alias, g.name, g.pinned, 0, g.kv_override}, h);
                it->second->adaptKv(kv);
            }

            auto* b = it->second;
            futs.push_back(std::async(std::launch::async,
                [b, &prompt, &sp] { return b->chat(prompt, sp); }));
        }

        for (auto& f : futs) {
            try {
                auto res = f.get();
                r.responses.push_back(res);
                if (res.ok) r.completed++;
            } catch (const std::exception& e) {
                r.responses.push_back(InferenceResult{"", "", e.what(), 0, 0, false});
            }
        }

        r.latency_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        if (r.completed >= r.required) {
            r.status = "complete";
        } else {
            auto fb = policy_->onFailure(g.name, r.completed, r.required);
            r.status = (fb == PolicyStrategy::Fallback::PARTIAL ||
                        fb == PolicyStrategy::Fallback::RETRY) ? "partial" : "failed";
        }
        return r;
    }

    // ── Sequential: output(N) -> input(N+1) ──
    GroupResult execSequential(const GroupConfig& g, const std::string& prompt,
                              const SamplingParams& sp,
                              std::map<std::string, ComputeBackend*>& backends) {
        auto t0 = std::chrono::steady_clock::now();
        GroupResult r;
        r.group = g.name;
        r.required = 1;

        std::string input = prompt;
        for (auto& alias : g.models) {
            auto it = backends.find(alias);
            if (it == backends.end() || !it->second->loaded) {
                r.status = "failed";
                return r;
            }
            auto res = it->second->chat(input, sp);
            r.responses.push_back(res);
            if (!res.ok) { r.status = "failed"; return r; }
            input = res.text;
            r.completed++;
        }

        r.latency_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        r.status = "complete";
        return r;
    }

    // ── Fan-out: same prompt, best response wins ──
    GroupResult execFanout(const GroupConfig& g, const std::string& prompt,
                          const SamplingParams& sp,
                          std::map<std::string, ComputeBackend*>& backends) {
        auto pr = execParallel(g, prompt, sp, backends);
        GroupResult r;
        r.group = g.name;
        r.required = 1;
        r.latency_ms = pr.latency_ms;

        InferenceResult best;
        for (auto& res : pr.responses) {
            if (res.ok && res.text.size() > best.text.size()) best = res;
        }
        if (!best.text.empty()) {
            r.responses.push_back(best);
            r.completed = 1;
            r.status = "complete";
        } else {
            r.status = "failed";
        }
        return r;
    }

    // ── Dispatch based on group type ──
    GroupResult exec(const GroupConfig& g, const std::string& prompt,
                     const SamplingParams& sp,
                     std::map<std::string, ComputeBackend*>& backends) {
        if (g.type == "sequential") return execSequential(g, prompt, sp, backends);
        if (g.type == "fanout") return execFanout(g, prompt, sp, backends);
        return execParallel(g, prompt, sp, backends);
    }
};

} // namespace eie
