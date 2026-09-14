#include "core/scheduling.h"
#include <stdexcept>

static int checks = 0;
static void expect(bool ok) { ++checks; if (!ok) throw std::runtime_error("recovery assertion " + std::to_string(checks)); }
class Backend : public eie::ComputeBackend {
public:
    int calls = 0, failures = 0;
    bool throws = false, cancel = false, partial_output = false;
    std::vector<std::string> prompts;
    Backend(std::string id, int fail = 0) { alias = id; loaded = true; failures = fail; }
    eie::BackendType type() const override { return eie::BackendType::CPU; }
    std::string name() const override { return "fixture"; }
    bool init(int) override { return true; }
    bool load(const eie::ModelParams&) override { return true; }
    void unload() override {}
    eie::VramStatus vram() override { return {}; }
    eie::HealthStatus health() override { return {}; }
    std::string formatChat(const std::vector<eie::ChatMessage>& m) override { return alias + ":" + m.back().content; }
    eie::InferenceResult chat(const std::string& prompt, const eie::SamplingParams& sp) override {
        ++calls; prompts.push_back(prompt);
        eie::InferenceResult r; r.model = alias;
        if (partial_output && sp.on_token) sp.on_token("partial");
        if (calls <= failures) {
            if (throws) throw std::runtime_error("injected exception");
            r.ok = false; r.error = "injected failure"; r.finish_reason = cancel ? "cancelled" : "error";
        } else r.text = alias + " output";
        return r;
    }
};
int main() { try {
    for (const auto& strategy : {"generic", "pinned-group", "multi-group", "fixed-appliance"}) {
        for (const auto& type : {"parallel", "sequential", "fanout"}) {
            for (const auto& action : {"retry_once", "replace_with"}) {
                auto policy = eie::createStrategy(strategy);
                eie::GroupScheduler scheduler(policy.get());
                eie::GroupConfig g; g.name = "test"; g.models = {"primary"}; g.type = type;
                g.fallback = action; g.replacement = "backup";
                Backend primary("primary", 1), backup("backup");
                std::map<std::string, eie::ComputeBackend*> backends{{"primary", &primary}, {"backup", &backup}};
                auto r = scheduler.exec(g, {{"user", "question"}}, "fallback", {}, backends);
                expect(r.status == "complete" && r.completed == 1 && r.responses.size() == 1);
                expect(r.attempts.size() == 2 && !r.attempts[0].result.ok && r.attempts[1].result.ok);
                expect(r.attempts[0].result.error == "injected failure");
                bool retry = g.fallback == "retry_once";
                expect(primary.calls == (retry ? 2 : 1) && backup.calls == (retry ? 0 : 1));
                expect(r.responses[0].model == (retry ? "primary" : "backup"));
                expect((retry ? primary.prompts.back() : backup.prompts.back()) == (retry ? "primary:question" : "backup:question"));
            }
        }
    }
    eie::GenericStrategy policy; eie::GroupScheduler scheduler(&policy);
    eie::GroupConfig g; g.name = "test"; g.models = {"first", "second"}; g.required = 2; g.fallback = "retry_once";
    Backend first("first"), second("second", 1), backup("backup");
    std::map<std::string, eie::ComputeBackend*> backends{{"first", &first}, {"second", &second}, {"backup", &backup}};
    auto r = scheduler.exec(g, "p", {}, backends);
    expect(r.completed == 2 && first.calls == 1 && second.calls == 2 && r.attempts.size() == 3);
    second.calls = 0; g.required = 1;
    r = scheduler.exec(g, "p", {}, backends);
    expect(second.calls == 1 && r.attempts.size() == 2); // quorum: no needless retry
    g.models = {"second"}; second.calls = 0; second.failures = 9; second.throws = true;
    r = scheduler.exec(g, "p", {}, backends);
    expect(second.calls == 2 && r.status == "failed" && r.attempts.size() == 2);
    expect(r.attempts[0].result.error == "injected exception" && r.completed == 0);
    second.calls = 0; second.throws = false; second.cancel = true;
    r = scheduler.exec(g, "p", {}, backends);
    expect(second.calls == 1 && r.attempts.size() == 1);
    second.cancel = false; second.calls = 0; second.partial_output = true;
    eie::SamplingParams sp; sp.on_token = [](const std::string&) { return true; };
    r = scheduler.exec(g, "p", sp, backends);
    expect(second.calls == 1 && r.responses[0].finish_reason == "partial_output");
    sp.should_continue = [] { return false; }; second.calls = 0;
    r = scheduler.exec(g, "p", sp, backends);
    expect(second.calls == 0 && r.attempts.size() == 1);
    g.fallback = "replace_with"; g.replacement = "absent";
    r = scheduler.exec(g, "p", {}, backends);
    expect(r.status == "failed" && r.attempts.size() == 2 && r.responses[0].error == "model not loaded: absent");
    g.replacement = "backup"; g.models = {"absent"};
    r = scheduler.exec(g, "p", {}, backends);
    expect(r.status == "complete" && r.responses[0].model == "backup");
    g.type = "sequential"; g.models = {"absent", "first"};
    r = scheduler.exec(g, {{"user", "p"}}, "p", {}, backends);
    expect(r.status == "complete" && r.completed == 2 && r.required == 2);
    expect(first.prompts.back() == "first:backup output");
    r = scheduler.exec(g, "raw", {}, backends);
    expect(backup.prompts.back() == "raw" && first.prompts.back() == "backup output");
    g.models = {"absent"}; g.replacement.clear();
    r = scheduler.exec(g, "p", {}, backends);
    expect(r.status == "failed" && r.attempts.size() == 2 &&
           r.responses[0].error == "replace_with requires a replacement alias in the group configuration");
    std::cout << "PASS " << checks << " recovery assertions (fake backends, production scheduler)\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
