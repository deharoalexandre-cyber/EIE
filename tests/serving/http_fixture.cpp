#include "server/api.h"
#include "backends/text_output.h"
#include <stdexcept>
#include <mutex>
#include <map>

namespace eie {
class FixtureBackend : public ComputeBackend {
    std::mutex mutex_;
    std::map<std::string, int> calls_;
    RoutingHistogram routing_;
    bool trace_default_ = false;
public:
    BackendType type() const override { return BackendType::CPU; }
    std::string name() const override { return "HTTP test fixture, not inference"; }
    bool init(int) override { return true; }
    bool load(const ModelParams& p) override { alias = p.alias; trace_default_ = p.ews_trace; loaded = true; return true; }
    void unload() override { loaded = false; }
    VramStatus vram() override { return {}; }
    HealthStatus health() override { return {loaded, 0, "fixture"}; }
    std::vector<float> embed(const std::string&) override { return {1, 0, 0}; }
    InferenceResult chat(const std::string& prompt, const SamplingParams& s) override {
        std::lock_guard<std::mutex> lock(mutex_);
        InferenceResult r;
        r.model = alias; r.prompt_tokens = 11; r.reused_tokens = s.one_shot ? -1 : 3;
        if (prompt.rfind("recover-", 0) == 0 && alias == "fixture" && ++calls_[prompt] == 1) {
            r.ok = false; r.error = "first call failed"; r.finish_reason = "error"; return r;
        }
        routing_.begin(s.ews_trace || trace_default_, 288, 8, 8, {{1, 384}});
        if (s.ews_trace || trace_default_) {
            routing_.callback(1);
            for (int id = 280; id < 288; ++id) routing_.access(1, id, false);
        }
        routing_.end("complete"); // Synthetic data tests API plumbing only.
        if (prompt == "error") { r.ok = false; r.error = "injected test error"; return r; }
        if (prompt == "overflow") { r.ok = false; r.error = "context_length_exceeded"; return r; }
        TextOutput output(s.stop, s.on_token);
        std::vector<std::string> pieces = {"caf\xC3", "\xA9 ", "E", "ND tail"};
        r.finish_reason = "length";
        for (const auto& piece : pieces) {
            if (r.tokens >= s.max_tokens) break;
            ++r.tokens;
            if (!output.push(piece)) break;
        }
        output.finish();
        r.text = output.text();
        if (output.stopped()) r.finish_reason = "stop";
        if (output.cancelled()) { r.finish_reason = "cancelled"; r.ok = false; }
        return r;
    }
    RoutingHistogram streamingRouting() override {
        std::lock_guard<std::mutex> lock(mutex_);
        return routing_;
    }
};
std::unique_ptr<ComputeBackend> detectBackend(int) { return std::make_unique<FixtureBackend>(); }
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    eie::ServerConfig cfg;
    cfg.host = "127.0.0.1"; cfg.port = std::stoi(argv[1]);
    if (argc == 3) cfg.auth_token = argv[2]; // Public test credential only.
    auto policy = eie::createStrategy("generic");
    for (const auto* action : {"retry_once", "replace_with"}) {
        eie::GroupConfig group;
        group.name = action; group.models = {"fixture"}; group.fallback = action; group.replacement = "idle";
        policy->groups[group.name] = group;
    }
    eie::GroupScheduler scheduler(policy.get());
    eie::ModelManager models;
    models.setExpertTrace("idle", true);
    for (const auto& alias : {"fixture", "idle", "embedder"}) {
        models.reg(alias, "no-model-needed"); models.load(alias, {});
    }
    eie::Metrics metrics;
    eie::AuditTrail audit(false);
    eie::startServer(cfg, models, scheduler, policy.get(), metrics, audit);
}
