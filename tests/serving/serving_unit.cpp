// No model, GPU, HTTP listener or private state. Exercise the same output,
// serialization and metrics code used by the server, not a second matcher.
#include "backends/text_output.h"
#include "backends/cpu_backend.cpp"
#include "server/api.cpp"
#include <atomic>
#include <stdexcept>
#include <thread>

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) throw std::runtime_error(label);
}

static void output_case(const std::vector<std::string>& pieces,
                        const std::vector<std::string>& stops,
                        const std::string& expected, bool stopped) {
    std::string wire;
    eie::TextOutput streamed(stops, [&](const auto& chunk) { wire += chunk; return true; });
    eie::TextOutput buffered(stops);
    for (const auto& p : pieces) { streamed.push(p); buffered.push(p); }
    check(streamed.finish() && buffered.finish(), "normal finish must succeed");
    check(wire == expected && streamed.text() == expected && buffered.text() == expected,
          "streamed and buffered output differ");
    check(streamed.stopped() == stopped && buffered.stopped() == stopped, "wrong stop outcome");
    streamed.finish();
    check(wire == expected, "finish is not idempotent");
}

static void stops_and_utf8() {
    output_case({"hello", " world"}, {}, "hello world", false);
    output_case({"hello EN", "D hidden"}, {"END"}, "hello ", true);
    output_case({"beforeENDafter"}, {"END"}, "before", true);
    output_case({"ENDall"}, {"END"}, "", true);
    output_case({"hello EN"}, {"END"}, "hello EN", false);
    output_case({"xS", "T", "O", "P"}, {"", "STOP", "STOP"}, "x", true);
    output_case({"anything"}, {""}, "anything", false);
    output_case({"ab", "X", "abc"}, {"abc"}, "abX", true);
    output_case({"abcd"}, {"abcd", "bc"}, "a", true);
    output_case({"abc"}, {"bc", "abc"}, "", true);
    output_case({"caf\xC3", "\xA9 EN", "D"}, {"END"}, "caf\xC3\xA9 ", true);
    output_case({"\xF0", "\x9F", "\x98", "\x80"}, {}, "\xF0\x9F\x98\x80", false);
    output_case({"a\xC3", "\xA9z"}, {"\xC3\xA9"}, "a", true);
    output_case({"x\xC3"}, {}, "x\xEF\xBF\xBD", false);
    output_case({"\xFF", "X"}, {}, "\xEF\xBF\xBDX", false);
    output_case({"\xC3", "A"}, {}, "\xEF\xBF\xBD" "A", false);

    // Every byte split: output cannot depend on token piece boundaries.
    for (const auto& input : {std::string("xENDtail"), std::string("abcd")}) {
        const auto stops = input[0] == 'x' ? std::vector<std::string>{"END"}
                                         : std::vector<std::string>{"abcd", "bc"};
        const std::string expected = input[0] == 'x' ? "x" : "a";
        for (unsigned mask = 0; mask < (1u << (input.size()-1)); ++mask) {
            std::vector<std::string> pieces;
            size_t start = 0;
            for (size_t i = 0; i+1 < input.size(); ++i)
                if (mask & (1u << i)) { pieces.push_back(input.substr(start, i+1-start)); start = i+1; }
            pieces.push_back(input.substr(start));
            output_case(pieces, stops, expected, true);
        }
    }
    std::string wire;
    eie::TextOutput live({"END"}, [&](const auto& p) { wire += p; return true; });
    check(live.push("hello") && wire == "hello", "stream must not wait for end of generation");
    check(live.push(" E") && wire == "hello ", "only possible stop prefix is held");
    check(live.push("x") && wire == "hello Ex", "mismatched prefix is released");

    int callbacks = 0;
    eie::TextOutput cancelled({"END"}, [&](const auto&) { ++callbacks; return false; });
    check(!cancelled.push("visible") && cancelled.cancelled(), "disconnect not propagated");
    cancelled.push("later"); cancelled.finish();
    check(callbacks == 1, "emission continued after disconnect");
    eie::TextOutput cancelled_flush({"END"}, [](const auto&) { return false; });
    check(cancelled_flush.push("EN") && !cancelled_flush.finish(), "disconnect during final flush");
}

static void usage_and_loaded_state() {
    eie::InferenceResult r;
    r.text = "quote\" newline\n control\x01";
    r.prompt_tokens = 23; r.tokens = 7; r.reused_tokens = 19;
    auto body = eie::chatCompletionJson(r, "model\"id");
    check(body.find("\"prompt_tokens\":23") != std::string::npos, "prompt usage missing");
    check(body.find("\"total_tokens\":30") != std::string::npos, "total usage wrong");
    check(body.find("\"cached_tokens\":19") != std::string::npos, "cached usage wrong");
    check(body.find("\\u0001") != std::string::npos && body.find("model\\\"id") != std::string::npos,
          "invalid JSON escaping");
    r.reused_tokens = -1;
    check(eie::chatCompletionJson(r, "m").find("\"cached_tokens\":0") != std::string::npos,
          "one-shot cached usage must not be negative");

    eie::Metrics metrics;
    eie::ModelManager models;
    models.reg("idle", "unused-placeholder.gguf");
    check(models.load("idle", {}), "placeholder model load");
    check(metrics.healthJson(models.loaded().size()).find("\"models\":1") != std::string::npos,
          "loaded but idle model absent from health");
    metrics.recordModel("not-loaded", 1, 4);
    check(metrics.prometheus(models.loaded().size()).find("eie_models_loaded 1\n") != std::string::npos,
          "request history changed loaded count");
    models.unloadModel("idle");
    check(metrics.healthJson(models.loaded().size()).find("\"models\":0") != std::string::npos,
          "unloaded model still counted");
}

static void concurrent_metrics() {
    eie::Metrics m;
    std::atomic<bool> valid{true};
    std::vector<std::thread> writers;
    for (int i=0; i<8; ++i) writers.emplace_back([&] {
        for (int j=0; j<1000; ++j) {
            m.recordModel("shared", 2, 3);
            m.recordGroup("group", true, 1);
            if (j % 10 == 0 && m.prometheus(2).find("eie_models_loaded 2\n") == std::string::npos) valid = false;
        }
    });
    for (auto& t : writers) t.join();
    auto p = m.prometheus(2);
    check(valid && p.find("eie_model_requests_total{model=\"shared\"} 8000\n") != std::string::npos,
          "concurrent request counts lost");
    check(p.find("eie_model_tokens_total{model=\"shared\"} 24000\n") != std::string::npos,
          "concurrent token counts lost");
    check(p.find("eie_group_executions_total{group=\"group\"} 8000\n") != std::string::npos,
          "concurrent group counts lost");
}

static void backend_exception() {
    class Throws : public eie::CpuBackend {
        eie::InferenceResult chat(const std::string&, const eie::SamplingParams&) override {
            throw std::runtime_error("injected backend exception");
        }
    } backend;
    backend.loaded = true;
    eie::GroupConfig group;
    group.name = "test"; group.models = {"throws"}; group.required = 1; group.fallback = "strict";
    eie::PinnedGroupStrategy policy;
    policy.groups[group.name] = group;
    eie::GroupScheduler scheduler(&policy);
    std::map<std::string, eie::ComputeBackend*> backends{{"throws", &backend}};
    const auto result = scheduler.execParallel(group, "test", {}, backends);
    check(result.status == "failed" && result.responses.size() == 1, "backend exception escaped scheduler");
    check(!result.responses[0].ok && result.responses[0].error == "injected backend exception",
          "backend exception did not preserve its failure status");
}

int main() {
    try {
        stops_and_utf8(); usage_and_loaded_state(); concurrent_metrics(); backend_exception();
        std::cout << "PASS: " << checks << " assertions (no model/GPU).\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
