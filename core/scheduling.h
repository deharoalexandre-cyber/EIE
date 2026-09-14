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
    // Surcharge KV du groupe : VIDE par défaut (type_k/type_v ""), sinon les
    // valeurs par défaut de KvConfig (turbo3) masquent le réglage global du
    // preset — les modèles de groupe se chargeaient en turbo3 malgré `type_k: f16`.
    KvConfig kv_override = [] { KvConfig k; k.type_k.clear(); k.type_v.clear(); return k; }();
};

struct GroupAttempt {
    std::string requested_model, action;
    InferenceResult result;
};

struct GroupResult {
    std::string group;
    std::vector<InferenceResult> responses;
    std::vector<GroupAttempt> attempts; // Initial failures remain visible after recovery.
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

/** Rend le prompt pour un backend donné : template de chat natif du modèle
 *  (métadonnées GGUF) si disponible, sinon le repli fourni par l'appelant. */
inline std::string renderPrompt(ComputeBackend* b,
                                const std::vector<ChatMessage>& msgs,
                                const std::string& fallback) {
    if (b && !msgs.empty()) {
        std::string p = b->formatChat(msgs);
        if (!p.empty()) return p;
    }
    return fallback;
}

class GroupScheduler {
    PolicyStrategy* policy_;
    PolicyStrategy::Fallback fallback(const GroupConfig& g, int completed) {
        if (g.fallback == "retry_once") return PolicyStrategy::Fallback::RETRY;
        if (g.fallback == "replace_with") return PolicyStrategy::Fallback::REPLACE;
        if (g.fallback == "partial") return PolicyStrategy::Fallback::PARTIAL;
        return policy_->onFailure(g.name, completed, g.required);
    }

    InferenceResult invoke(const std::string& alias, const GroupConfig& g,
                           const std::vector<ChatMessage>& msgs, const std::string& prompt,
                           const SamplingParams& sp, const std::map<std::string, ComputeBackend*>& backends) {
        InferenceResult result;
        result.model = alias;
        bool emitted = false;
        try {
            if (sp.should_continue && !sp.should_continue()) {
                result.ok = false; result.error = "request cancelled"; result.finish_reason = "cancelled";
                return result;
            }
            auto it = backends.find(alias);
            if (it == backends.end() || !it->second || !it->second->loaded) {
                result.ok = false; result.error = "model not loaded: " + alias; result.finish_reason = "error";
                return result;
            }
            auto* backend = it->second;
            const auto health = backend->health();
            if (health.latency_ms > g.max_latency_ms)
                backend->adaptKv(policy_->adaptKv(Slot{alias, g.name, g.pinned, 0, g.kv_override}, health));
            auto sampling = sp;
            if (sp.on_token) sampling.on_token = [&](const std::string& piece) {
                if (!piece.empty()) emitted = true;
                return sp.on_token(piece);
            };
            result = backend->chat(renderPrompt(backend, msgs, prompt), sampling);
            result.model = alias;
        } catch (const std::exception& e) {
            result.ok = false; result.error = e.what(); result.finish_reason = "error";
        } catch (...) {
            result.ok = false; result.error = "unknown backend exception"; result.finish_reason = "error";
        }
        // Never append a second generation after externally visible partial output.
        if (!result.ok && emitted) result.finish_reason = "partial_output";
        return result;
    }

    void recover(InferenceResult& result, const std::string& requested, const GroupConfig& g,
                 const std::vector<ChatMessage>& msgs, const std::string& prompt,
                 const SamplingParams& sp, const std::map<std::string, ComputeBackend*>& backends,
                 GroupResult& group) {
        if (result.ok || result.finish_reason == "cancelled" || result.finish_reason == "partial_output" ||
            (sp.should_continue && !sp.should_continue())) return;
        auto action = fallback(g, group.completed);
        if (action != PolicyStrategy::Fallback::RETRY && action != PolicyStrategy::Fallback::REPLACE) return;
        const auto target = action == PolicyStrategy::Fallback::RETRY ? requested : g.replacement;
        if (target.empty()) {
            result = InferenceResult{};
            result.ok = false; result.finish_reason = "error";
            result.error = "replace_with requires a replacement alias in the group configuration";
        } else result = invoke(target, g, msgs, prompt, sp, backends);
        group.attempts.push_back({requested, action == PolicyStrategy::Fallback::RETRY ? "retry" : "replace", result});
    }
public:
    GroupScheduler(PolicyStrategy* p) : policy_(p) {}

    // ── Parallel: same messages to N models simultaneously ──
    // Chaque modèle du groupe templte les mêmes messages avec son propre
    // template natif ; fallback_prompt sert aux modèles sans template.
    GroupResult execParallel(const GroupConfig& g,
                            const std::vector<ChatMessage>& msgs,
                            const std::string& fallback_prompt,
                            const SamplingParams& sp,
                            std::map<std::string, ComputeBackend*>& backends) {
        auto t0 = std::chrono::steady_clock::now();
        GroupResult r;
        r.group = g.name;
        r.required = g.required;

        std::vector<std::future<InferenceResult>> futs;
        for (auto& alias : g.models) {
            futs.push_back(std::async(std::launch::async,
                [&, alias] { return invoke(alias, g, msgs, fallback_prompt, sp, backends); }));
        }

        for (size_t i = 0; i < futs.size(); ++i) {
            auto res = futs[i].get();
            r.attempts.push_back({g.models[i], "initial", res});
            if (res.ok) ++r.completed;
            r.responses.push_back(std::move(res));
        }
        // Recover only missing successes, in declared order, never successful members.
        for (size_t i = 0; i < r.responses.size() && r.completed < r.required; ++i) {
            if (r.responses[i].ok) continue;
            recover(r.responses[i], g.models[i], g, msgs, fallback_prompt, sp, backends, r);
            if (r.responses[i].ok) ++r.completed;
        }

        r.latency_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        if (r.completed >= r.required) {
            r.status = "complete";
        } else {
            r.status = fallback(g, r.completed) == PolicyStrategy::Fallback::PARTIAL ? "partial" : "failed";
        }
        return r;
    }

    // Raw prompt passthrough — aucun template appliqué.
    GroupResult execParallel(const GroupConfig& g, const std::string& prompt,
                            const SamplingParams& sp,
                            std::map<std::string, ComputeBackend*>& backends) {
        return execParallel(g, {}, prompt, sp, backends);
    }

    // ── Sequential: output(N) -> input(N+1) ──
    // La sortie du modèle N devient le tour utilisateur du modèle N+1,
    // re-templatée avec le template natif de chaque maillon.
    GroupResult execSequential(const GroupConfig& g,
                              const std::vector<ChatMessage>& msgs,
                              const std::string& fallback_prompt,
                              const SamplingParams& sp,
                              std::map<std::string, ComputeBackend*>& backends) {
        auto t0 = std::chrono::steady_clock::now();
        GroupResult r;
        r.group = g.name;
        r.required = static_cast<int>(g.models.size());

        std::vector<ChatMessage> step_msgs = msgs;
        std::string step_fallback = fallback_prompt;
        for (auto& alias : g.models) {
            auto res = invoke(alias, g, step_msgs, step_fallback, sp, backends);
            r.attempts.push_back({alias, "initial", res});
            recover(res, alias, g, step_msgs, step_fallback, sp, backends, r);
            r.responses.push_back(res);
            if (!res.ok) break;
            step_msgs = msgs.empty() ? std::vector<ChatMessage>{} : std::vector<ChatMessage>{{"user", res.text}};
            step_fallback = res.text;
            r.completed++;
        }

        r.latency_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        r.status = !g.models.empty() && r.completed == r.required ? "complete" : "failed";
        return r;
    }

    // Raw prompt passthrough — la chaîne circule sans template.
    GroupResult execSequential(const GroupConfig& g, const std::string& prompt,
                              const SamplingParams& sp,
                              std::map<std::string, ComputeBackend*>& backends) {
        return execSequential(g, {}, prompt, sp, backends);
    }

    // ── Fan-out: same prompt, best response wins ──
    GroupResult execFanout(const GroupConfig& g,
                          const std::vector<ChatMessage>& msgs,
                          const std::string& fallback_prompt,
                          const SamplingParams& sp,
                          std::map<std::string, ComputeBackend*>& backends) {
        auto pr = execParallel(g, msgs, fallback_prompt, sp, backends);
        GroupResult r;
        r.group = g.name;
        r.required = 1;
        r.latency_ms = pr.latency_ms;
        r.attempts = pr.attempts;

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

    GroupResult execFanout(const GroupConfig& g, const std::string& prompt,
                          const SamplingParams& sp,
                          std::map<std::string, ComputeBackend*>& backends) {
        return execFanout(g, {}, prompt, sp, backends);
    }

    // ── Dispatch based on group type ──
    GroupResult exec(const GroupConfig& g,
                     const std::vector<ChatMessage>& msgs,
                     const std::string& fallback_prompt,
                     const SamplingParams& sp,
                     std::map<std::string, ComputeBackend*>& backends) {
        if (g.type == "sequential") return execSequential(g, msgs, fallback_prompt, sp, backends);
        if (g.type == "fanout") return execFanout(g, msgs, fallback_prompt, sp, backends);
        return execParallel(g, msgs, fallback_prompt, sp, backends);
    }

    GroupResult exec(const GroupConfig& g, const std::string& prompt,
                     const SamplingParams& sp,
                     std::map<std::string, ComputeBackend*>& backends) {
        if (g.type == "sequential") return execSequential(g, prompt, sp, backends);
        if (g.type == "fanout") return execFanout(g, prompt, sp, backends);
        return execParallel(g, prompt, sp, backends);
    }
};

} // namespace eie
