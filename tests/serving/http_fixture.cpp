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
    ChatPrompt formatChatRequest(const std::vector<ChatMessage>& msgs,
                                 const ChatFormatOptions& options) override {
        ChatPrompt out;
        if (options.tools.empty()) return out; // Preserve legacy prompt tests.

        const bool schema = options.tools.size() == 1 &&
            options.tools[0].name == "lookup_weather" &&
            options.tools[0].description == "Look up the weather" &&
            options.tools[0].parameters.find("\"city\"") != std::string::npos &&
            options.tools[0].parameters.find("\"required\"") != std::string::npos;
        const bool history = msgs.size() == 3 && msgs[1].role == "assistant" &&
            msgs[1].reasoning_content == "Checking weather." &&
            msgs[1].tool_calls.size() == 1 &&
            msgs[1].tool_calls[0].id == "call_fixture_weather" &&
            msgs[1].tool_calls[0].name == "lookup_weather" &&
            msgs[1].tool_calls[0].arguments == "{\"city\":\"Paris\"}" &&
            msgs[2].role == "tool" &&
            msgs[2].tool_call_id == "call_fixture_weather" &&
            msgs[2].content == "{\"temperature_c\":18}";

        if (options.reasoning_effort == "none" && options.enable_thinking)
            out.prompt = "fixture-tool-invalid-thinking";
        else if (!schema) out.prompt = "fixture-tool-invalid-schema";
        else if (msgs.size() == 3) out.prompt = history ? "fixture-tool-followup"
                                                         : "fixture-tool-invalid-history";
        else if (options.tool_choice == ChatToolChoice::None) out.prompt = "fixture-tool-none";
        else if (options.parallel_tool_calls) out.prompt = "fixture-tool-parallel";
        else out.prompt = "fixture-tool-call";
        out.parse_chat_output = true;
        out.parse_tool_calls = options.tool_choice != ChatToolChoice::None;
        return out;
    }
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
        if (prompt == "fixture-tool-call" && s.chat_format.parse_tool_calls) {
            r.tokens = 4;
            r.finish_reason = "tool_calls";
            r.reasoning_content = "Checking weather.";
            r.tool_calls.push_back({"call_fixture_weather", "lookup_weather", "{\"city\":\"Paris\"}"});
            return r;
        }
        if (prompt == "fixture-tool-parallel" && s.chat_format.parse_tool_calls) {
            r.tokens = 8;
            r.finish_reason = "tool_calls";
            r.tool_calls.push_back({"call_fixture_weather", "lookup_weather", "{\"city\":\"Paris\"}"});
            r.tool_calls.push_back({"call_fixture_weather_2", "lookup_weather", "{\"city\":\"Lyon\"}"});
            return r;
        }
        if (prompt == "fixture-tool-followup") {
            r.tokens = 1;
            r.finish_reason = "stop";
            r.text = "It is 18 C in Paris.";
            if (s.on_token && !s.on_token(r.text)) {
                r.finish_reason = "cancelled";
                r.ok = false;
            }
            return r;
        }
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
