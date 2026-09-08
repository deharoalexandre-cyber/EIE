#include "server/api.h"
#include "backends/text_output.h"
#include <stdexcept>

namespace eie {
class FixtureBackend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::CPU; }
    std::string name() const override { return "HTTP test fixture, not inference"; }
    bool init(int) override { return true; }
    bool load(const ModelParams& p) override { alias = p.alias; loaded = true; return true; }
    void unload() override { loaded = false; }
    VramStatus vram() override { return {}; }
    HealthStatus health() override { return {loaded, 0, "fixture"}; }
    std::vector<float> embed(const std::string&) override { return {1, 0, 0}; }
    InferenceResult chat(const std::string& prompt, const SamplingParams& s) override {
        InferenceResult r;
        r.model = alias; r.prompt_tokens = 11; r.reused_tokens = s.one_shot ? -1 : 3;
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
};
std::unique_ptr<ComputeBackend> detectBackend(int) { return std::make_unique<FixtureBackend>(); }
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    eie::ServerConfig cfg;
    cfg.host = "127.0.0.1"; cfg.port = std::stoi(argv[1]);
    auto policy = eie::createStrategy("generic");
    eie::GroupScheduler scheduler(policy.get());
    eie::ModelManager models;
    for (const auto& alias : {"fixture", "idle", "embedder"}) {
        models.reg(alias, "no-model-needed"); models.load(alias, {});
    }
    eie::Metrics metrics;
    eie::AuditTrail audit(false);
    eie::startServer(cfg, models, scheduler, policy.get(), metrics, audit);
}
