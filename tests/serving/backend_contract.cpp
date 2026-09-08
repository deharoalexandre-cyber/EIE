// Explicit real-model gate. Never run while the production resident owns the GPU.
#include "backends/cpu_backend.cpp"
#include "nlohmann/json.hpp"
#include <stdexcept>

static void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
class ProbeBackend : public eie::CpuBackend {
public:
    int promptCount(const std::string& p) { return int(common_tokenize(ctx_, p, true, true).size()); }
    void loadTemplateOnly(const char * path) {
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        model_ = llama_model_load_from_file(path, params);
        check(model_ != nullptr, "vocabulary-only load failed");
        tmpls_ = common_chat_templates_init(model_, "");
    }
};

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--api-only") {
            check(eie::mapKvType("f16") == GGML_TYPE_F16, "F16 mapping changed");
            auto * sampler = eie::penaltySampler(llama_sampler_init_penalties, 1024);
            check(sampler != nullptr, "penalty sampler API mismatch");
            llama_sampler_free(sampler);
            common_chat_params rendered;
            rendered.prompt = "question<assistant><think>";
            rendered.supports_thinking = true;
            rendered.thinking_start_tag = "<think>";
            rendered.thinking_end_tags = {"</think>"};
            check(eie::answerOnlyPrompt(rendered) == rendered.prompt + "</think>", "open reasoning prefix not closed");
            rendered.prompt += "</think>";
            check(eie::answerOnlyPrompt(rendered) == rendered.prompt, "closed reasoning prefix changed");
            rendered.prompt = "quoted <think> in user text<assistant>";
            check(eie::answerOnlyPrompt(rendered) == rendered.prompt, "user text changed");
            for (const auto * name : {"turbo2", "turbo3", "turbo4"}) {
                const auto type = eie::mapKvType(name);
                std::cout << name << " -> " << ggml_type_name(type) << '\n';
            }
            std::cout << "PASS runtime API compatibility (no model/GPU)\n";
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--template-only") {
            ProbeBackend backend;
            backend.loadTemplateOnly(argv[2]);
            const auto prompt = backend.formatChat({{"user", "Combien font 17 + 25 ? Reponds uniquement avec le nombre."}});
            check(!prompt.empty(), "empty chat prompt");
            const std::string open = "<think>";
            check(prompt.size() < open.size() || prompt.compare(prompt.size() - open.size(), open.size(), open) != 0,
                  "answer-only template still opens reasoning");
            std::cout << nlohmann::json({{"result", "pass"}, {"prompt", prompt}, {"weights_loaded", false}}).dump() << '\n';
            return 0;
        }
        check(argc == 3, "usage: serving-model-contract MODEL_GGUF EWS_SLOTS (0 for normal loading)");
        ProbeBackend b;
        b.init(0);
        eie::ModelParams p;
        p.path = argv[1]; p.alias = "contract"; p.ews_slots = std::stoi(argv[2]);
        p.kv.type_k = p.kv.type_v = "f16"; p.kv.n_ctx = 512;
        check(b.load(p), "model load failed");
        const auto prompt = b.formatChat({{"user", "Write one short sentence about a blue bicycle."}});
        check(!prompt.empty(), "native prompt template missing");
        const int expected_prompt = b.promptCount(prompt);
        eie::SamplingParams s;
        s.temperature = 0; s.max_tokens = 32; s.one_shot = true;
        const auto reference = b.chat(prompt, s);
        check(reference.ok && !reference.text.empty(), "reference failed");
        check(reference.prompt_tokens == expected_prompt, "usage differs from actual tokenizer");
        std::string wire;
        s.on_token = [&](const auto& text) { wire += text; return true; };
        const auto streamed = b.chat(prompt, s);
        check(streamed.ok && wire == reference.text && streamed.text == reference.text, "SSE helper output differs");
        check(streamed.tokens == reference.tokens && streamed.prompt_tokens == expected_prompt, "stream usage differs");
        const auto word = reference.text.find_first_not_of(" \r\n\t");
        check(word != std::string::npos, "reference is whitespace only");
        const auto stop = reference.text.substr(0, reference.text.find_first_of(" \r\n\t", word));
        check(!stop.empty(), "reference has no usable stop");
        s.stop = {stop}; s.one_shot = false; wire.clear();
        const auto stopped = b.chat(prompt, s);
        check(stopped.ok && stopped.finish_reason == "stop" && stopped.text.empty() && wire.empty(), "stop leaked");
        s.stop.clear(); s.on_token = {};
        const auto recovery = b.chat(prompt, s);
        check(recovery.ok && recovery.text == reference.text, "persistent KV recovery after stop failed");
        check(recovery.prompt_tokens == expected_prompt && recovery.tokens == reference.tokens, "recovery usage differs");
        s.one_shot = true;
        const auto one_shot = b.chat(prompt, s);
        check(one_shot.ok && one_shot.text == reference.text, "one-shot changed output");
        s.one_shot = false;
        const auto reused = b.chat(prompt, s);
        check(reused.ok && reused.text == reference.text && reused.reused_tokens > 0, "one-shot damaged persistent KV");
        s.on_token = [](const auto&) { return false; };
        const auto cancelled = b.chat(prompt, s);
        check(!cancelled.ok && cancelled.finish_reason == "cancelled", "callback cancellation failed");
        s.on_token = {};
        const auto after_cancel = b.chat(prompt, s);
        check(after_cancel.ok && after_cancel.text == reference.text, "recovery after cancellation failed");
        s.truncate_prompt = false; s.max_tokens = 512;
        const auto overflow = b.chat(prompt, s);
        check(!overflow.ok && overflow.error == "context_length_exceeded", "overflow error missing");
        s.max_tokens = 32;
        const auto after_overflow = b.chat(prompt, s);
        check(after_overflow.ok && after_overflow.text == reference.text, "recovery after overflow failed");
        std::cout << nlohmann::json({{"result", "pass"}, {"slots", p.ews_slots},
            {"prompt_tokens", expected_prompt}, {"completion_tokens", reference.tokens},
            {"text", reference.text}, {"reused_tokens", reused.reused_tokens},
            {"ews", b.streamingStats()}}).dump() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "MODEL CONTRACT FAILED: " << e.what() << '\n';
        return 1;
    }
}
