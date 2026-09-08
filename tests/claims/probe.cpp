// Characterization, not feature acceptance: checks remaining limitations.
// No llama.cpp, model, GPU API, HTTP server or production state is used.
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include "core/config.h"
#include "monitoring/monitoring.h"
#include "backends/cpu_backend.cpp"

static void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
    std::cout << "OBSERVED: " << message << '\n';
}

class FailingBackend : public eie::CpuBackend {
public:
    int calls = 0;
    eie::InferenceResult chat(const std::string&, const eie::SamplingParams&) override {
        ++calls;
        eie::InferenceResult r;
        r.ok = false;
        r.error = "injected failure";
        return r;
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("supply fixture.yaml");
        auto cfg = eie::loadConfig(argv[1]);
        check(cfg.vram.reserve_mb == 123, "reserve_mb is parsed (not an enforcement test)");
        check(cfg.vram.low_wm == 0.85f && cfg.vram.crit_wm == 0.95f && !cfg.vram.group_isolation,
              "watermarks and isolation fixture settings are ignored");
        auto g = cfg.groups.at("core");
        auto effective = g.kv_override.type_k.empty() ? cfg.default_kv : g.kv_override;
        check(effective.n_ctx == 4096 && effective.type_k == "turbo3",
              "group defaults mask configured f16 / 8192 global KV settings");

        eie::PinnedGroupStrategy policy;
        policy.groups[g.name] = g;
        eie::GroupScheduler scheduler(&policy);
        FailingBackend primary, replacement;
        primary.loaded = replacement.loaded = true;
        std::map<std::string, eie::ComputeBackend*> backends{{"primary", &primary}, {"backup", &replacement}};
        eie::SamplingParams sp;
        auto retry = scheduler.execParallel(g, "probe", sp, backends);
        check(primary.calls == 1 && retry.status == "partial", "retry_once performs one call, then returns partial");
        g.fallback = "replace_with";
        g.replacement = "backup";
        policy.groups[g.name] = g;
        auto replaced = scheduler.execParallel(g, "probe", sp, backends);
        check(primary.calls == 2 && replacement.calls == 0 && replaced.status == "failed",
              "replace_with never invokes the replacement backend");
        check(primary.health().latency_ms == 0, "backend health supplies zero latency");

        eie::CudaBackend cuda;
        eie::HipBackend hip;
        auto cv = cuda.vram(), hv = hip.vram();
        check(cv.total_bytes == (16ULL << 30) && hv.total_bytes == (48ULL << 30),
              "no-llama CUDA/HIP placeholders still fabricate capacities; not device telemetry");
        eie::Metrics metrics;
        check(metrics.healthJson(2).find("\"models\":2") != std::string::npos,
              "health uses the supplied loaded registry count before any activity");
        metrics.recordModel("not-a-loaded-model", 1, 1);
        check(metrics.healthJson(2).find("\"models\":2") != std::string::npos,
              "recorded activity cannot change the supplied loaded registry count");
        std::cout << "CHARACTERIZATION COMPLETE: 9 observations; NOT 9 implemented features.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Characterization changed or failed: " << e.what() << '\n';
        return 1;
    }
}
