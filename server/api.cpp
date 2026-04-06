// EIE — API Server Implementation
// Layer 1: OpenAI-compatible | Layer 2: Generic extensions
// Apache License 2.0
//
// Uses cpp-httplib (bundled with llama.cpp) for HTTP.
// Build with llama.cpp submodule to get httplib.h.
// Without it, this file provides the route stubs for integration.

#include "api.h"
#include <iostream>
#include <sstream>
#include <ctime>
#include <thread>
#include <chrono>

// If httplib available (from llama.cpp vendor):
// #include "httplib.h"
// #define HAS_HTTPLIB 1

namespace eie {

// ═══════════════════════════════════════════
// JSON Helpers
// ═══════════════════════════════════════════

static std::string escapeJson(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out;
}

static std::string chatCompletionJson(const InferenceResult& r, const std::string& model) {
    std::ostringstream ss;
    ss << "{\"id\":\"eie-" << std::time(nullptr) << "\","
       << "\"object\":\"chat.completion\","
       << "\"model\":\"" << model << "\","
       << "\"choices\":[{\"index\":0,"
       << "\"message\":{\"role\":\"assistant\",\"content\":\"" << escapeJson(r.text) << "\"},"
       << "\"finish_reason\":\"stop\"}],"
       << "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":" << r.tokens
       << ",\"total_tokens\":" << r.tokens << "}}";
    return ss.str();
}

static std::string groupResultJson(const GroupResult& r) {
    std::ostringstream ss;
    ss << "{\"group\":\"" << r.group << "\",\"responses\":[";
    for (size_t i = 0; i < r.responses.size(); i++) {
        if (i > 0) ss << ",";
        ss << "{\"model\":\"" << r.responses[i].model << "\","
           << "\"content\":\"" << escapeJson(r.responses[i].text) << "\","
           << "\"latency_ms\":" << r.responses[i].latency_ms << ","
           << "\"ok\":" << (r.responses[i].ok ? "true" : "false") << "}";
    }
    ss << "],\"completed\":" << r.completed
       << ",\"required\":" << r.required
       << ",\"status\":\"" << r.status << "\""
       << ",\"latency_ms\":" << r.latency_ms << "}";
    return ss.str();
}

static std::string modelsJson(const std::vector<std::string>& list) {
    std::ostringstream ss;
    ss << "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < list.size(); i++) {
        if (i > 0) ss << ",";
        ss << "{\"id\":\"" << list[i] << "\",\"object\":\"model\",\"owned_by\":\"local\"}";
    }
    ss << "]}";
    return ss.str();
}

// ═══════════════════════════════════════════
// Server Implementation
// ═══════════════════════════════════════════

void startServer(const ServerConfig& cfg, ModelManager& models,
                 GroupScheduler& scheduler, PolicyStrategy* policy,
                 Metrics& metrics, AuditTrail& audit) {

#if defined(HAS_HTTPLIB) && HAS_HTTPLIB
    httplib::Server svr;

    // ════════════════════════════════════
    // LAYER 1 — OpenAI Compatible
    // ════════════════════════════════════

    // POST /v1/chat/completions
    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        // TODO: parse JSON body for model, messages, temperature, max_tokens
        // For now: extract model name, concatenate messages as prompt
        std::string model = "default";
        std::string prompt = req.body; // simplified
        SamplingParams sp;

        auto* backend = models.get(model);
        if (!backend) {
            res.status = 404;
            res.set_content("{\"error\":\"model not found\"}", "application/json");
            return;
        }

        auto result = backend->chat(prompt, sp);
        metrics.recordModel(model, result.latency_ms, result.tokens);
        res.set_content(chatCompletionJson(result, model), "application/json");
    });

    // GET /v1/models
    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(modelsJson(models.loaded()), "application/json");
    });

    // GET /health
    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(metrics.healthJson(), "application/json");
    });

    // ════════════════════════════════════
    // LAYER 2 — Generic Extensions
    // ════════════════════════════════════

    // POST /v1/batch/execute — parallel group execution
    svr.Post("/v1/batch/execute", [&](const httplib::Request& req, httplib::Response& res) {
        // TODO: parse group name and messages from JSON body
        std::string group_name = "core"; // simplified
        std::string prompt = req.body;
        SamplingParams sp;

        auto it = policy->groups.find(group_name);
        if (it == policy->groups.end()) {
            res.status = 404;
            res.set_content("{\"error\":\"group not found\"}", "application/json");
            return;
        }

        auto backends = models.getAll();
        auto result = scheduler.exec(it->second, prompt, sp, backends);
        metrics.recordGroup(group_name, result.status == "complete", result.latency_ms);
        audit.record(group_name, result.status, "prompt_hash");
        res.set_content(groupResultJson(result), "application/json");
    });

    // POST /v1/chain/execute — sequential chain
    svr.Post("/v1/chain/execute", [&](const httplib::Request& req, httplib::Response& res) {
        std::string group_name = "chain"; // simplified
        std::string prompt = req.body;
        SamplingParams sp;

        auto it = policy->groups.find(group_name);
        if (it == policy->groups.end()) {
            res.status = 404;
            res.set_content("{\"error\":\"group not found\"}", "application/json");
            return;
        }

        auto backends = models.getAll();
        auto result = scheduler.execSequential(it->second, prompt, sp, backends);
        metrics.recordGroup(group_name, result.status == "complete", result.latency_ms);
        res.set_content(groupResultJson(result), "application/json");
    });

    // GET /v1/admin/models/discover
    svr.Get("/v1/admin/models/discover", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(modelsJson(models.available()), "application/json");
    });

    // GET /v1/admin/vram/status
    svr.Get("/v1/admin/vram/status", [&](const httplib::Request&, httplib::Response& res) {
        // TODO: serialize VramStatus for each GPU
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // GET /v1/admin/scheduling/status
    svr.Get("/v1/admin/scheduling/status", [&](const httplib::Request&, httplib::Response& res) {
        std::ostringstream ss;
        ss << "{\"strategy\":\"" << policy->name() << "\","
           << "\"groups\":" << policy->groups.size() << "}";
        res.set_content(ss.str(), "application/json");
    });

    // POST /v1/admin/config/reload
    svr.Post("/v1/admin/config/reload", [&](const httplib::Request&, httplib::Response& res) {
        // TODO: reload config from file
        res.set_content("{\"status\":\"reload not yet implemented\"}", "application/json");
    });

    // GET /metrics — Prometheus
    svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(metrics.prometheus(), "text/plain");
    });

    // GET /v1/admin/health/deep
    svr.Get("/v1/admin/health/deep", [&](const httplib::Request&, httplib::Response& res) {
        // TODO: test inference on each loaded model
        res.set_content(metrics.healthJson(), "application/json");
    });

    std::cout << "[EIE] Server listening on " << cfg.host << ":" << cfg.port << std::endl;
    svr.listen(cfg.host, cfg.port);

#else
    // No httplib — standalone mode for testing
    std::cout << "[EIE] httplib not available — running in standalone test mode" << std::endl;
    std::cout << "[EIE] Server would listen on " << cfg.host << ":" << cfg.port << std::endl;
    std::cout << std::endl;

    // Demo: execute a group if any is configured
    if (!policy->groups.empty()) {
        auto& [gname, group] = *policy->groups.begin();
        std::cout << "[EIE] Demo: executing group '" << gname << "' ("
                  << group.models.size() << " models, " << group.type << ")" << std::endl;

        auto backends = models.getAll();
        if (!backends.empty()) {
            SamplingParams sp;
            auto result = scheduler.exec(group, "Hello, this is a test prompt.", sp, backends);

            std::cout << "\n[EIE] Group result:" << std::endl;
            std::cout << "  Status: " << result.status << std::endl;
            std::cout << "  Completed: " << result.completed << "/" << result.required << std::endl;
            std::cout << "  Latency: " << result.latency_ms << " ms" << std::endl;
            for (auto& r : result.responses) {
                std::cout << "  [" << r.model << "] " << r.text << " (" << r.latency_ms << "ms)" << std::endl;
            }

            metrics.recordGroup(gname, result.status == "complete", result.latency_ms);
            audit.record(gname, result.status, "test_prompt_hash");
        }
    }

    // Print metrics
    std::cout << "\n[EIE] Metrics:\n" << metrics.prometheus() << std::endl;
    std::cout << "[EIE] Health: " << metrics.healthJson() << std::endl;
    std::cout << "[EIE] Audit records: " << audit.size() << std::endl;

    std::cout << "\n[EIE] Standalone test complete." << std::endl;
    std::cout << "[EIE] To enable HTTP server, build with llama.cpp submodule (provides httplib.h)" << std::endl;
    std::cout << "[EIE] Then define HAS_HTTPLIB before including api.cpp" << std::endl;
#endif
}

} // namespace eie
