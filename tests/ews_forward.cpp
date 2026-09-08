// Matched, one-token reference/candidate forward test for EIE's actual runtime.
#include "backends/expert_stream.h"
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <cmath>

using json = nlohmann::ordered_json;
static void check(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
static bool reference_boundary(ggml_tensor * t, bool ask, void *) {
    return !ask || strncmp(t->name, "ffn_moe_topk-", 13) == 0;
}
int main(int argc, char ** argv) {
    try {
        check(argc >= 6 && argc <= 8, "usage: ews-forward MODEL PROMPT OUTPUT_PREFIX SLOTS PREDICT [gemma-gpu|glm-cpu|glm-gpu] [GPU_LAYERS]");
        const std::string profile = argc >= 7 ? argv[6] : "gemma-gpu";
        const bool glm = profile == "glm-cpu" || profile == "glm-gpu";
        check(glm || profile == "gemma-gpu", "unknown test profile");
        const int gpu_layers = argc == 8 ? std::stoi(argv[7]) : (glm ? 20 : 99);
        const int slots = std::stoi(argv[4]), predict = std::stoi(argv[5]);
        check(predict > 0 && predict <= 512, "invalid prediction count");
        std::string prefix = argv[3];
        check(!std::filesystem::exists(prefix + ".json") && !std::filesystem::exists(prefix + ".logits.bin"), "output exists");
        std::ifstream input(argv[2], std::ios::binary);
        check(bool(input), "prompt missing");
        std::string prompt((std::istreambuf_iterator<char>(input)), {});
        llama_backend_init();
        std::unique_ptr<eie::ExpertStream> stream;
        if (slots) stream = std::make_unique<eie::ExpertStream>(argv[1], slots);
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers; mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.use_extra_bufts = false; mp.ews_n_slots = slots;
        const llama_model_tensor_buft_override cpu_experts[] = {
            {"\\.ffn_.*_exps\\.weight", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
        if (glm) {
            if (profile == "glm-cpu") mp.tensor_buft_overrides = cpu_experts;
            if (!slots) mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        }
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        check(bool(model), "model load failed");
        if (stream) stream->bind(model.get());
        auto cp = llama_context_default_params();
        cp.n_ctx = 512; cp.n_batch = cp.n_ubatch = 1;
        cp.n_threads = cp.n_threads_batch = 4;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (glm) {
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.n_threads = cp.n_threads_batch = 8;
        }
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        if (stream) stream->configure(cp);
        else cp.cb_eval = reference_boundary;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        check(bool(ctx), "context load failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        int n = -llama_tokenize(vocab, prompt.data(), int(prompt.size()), nullptr, 0, true, true);
        check(n > 0 && n + predict < 512, "prompt does not fit test context");
        std::vector<llama_token> tokens(size_t(n), 0), generated;
        check(llama_tokenize(vocab, prompt.data(), int(prompt.size()), tokens.data(), n, true, true) == n, "tokenization failed");
        auto decode = [&](llama_token & t) {
            check(llama_decode(ctx.get(), llama_batch_get_one(&t, 1)) == 0, "decode failed");
            if (stream) check(stream->error().empty(), stream->error().c_str());
        };
        const auto start = std::chrono::steady_clock::now();
        for (auto & t : tokens) decode(t);
        const auto prefilled = std::chrono::steady_clock::now();
        std::ofstream logits(prefix + ".logits.bin", std::ios::binary);
        check(bool(logits), "cannot create logits output");
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < predict; ++i) {
            const float * values = llama_get_logits_ith(ctx.get(), -1);
            check(values != nullptr, "missing logits");
            for (int j = 0; j < n_vocab; ++j) check(std::isfinite(values[j]), "nonfinite logits");
            logits.write(reinterpret_cast<const char *>(values), n_vocab * sizeof(float));
            check(bool(logits), "logit output failed");
            llama_token next = llama_token(std::max_element(values, values + n_vocab) - values);
            generated.push_back(next);
            if (i + 1 < predict) decode(next);
        }
        const auto end = std::chrono::steady_clock::now();
        json report{{"slots", slots}, {"prompt_tokens", n}, {"generated_token_ids", generated},
            {"profile", profile}, {"gpu_layers", gpu_layers}, {"cpu_moe", profile == "glm-cpu"},
            {"vocab", n_vocab}, {"predict", predict}, {"kv", "f16/f16"}, {"router_boundary_both_arms", true},
            {"prefill_seconds", std::chrono::duration<double>(prefilled - start).count()},
            {"decode_seconds_including_logit_write", std::chrono::duration<double>(end - prefilled).count()}};
        if (stream) {
            auto s = stream->stats();
            report["ews"] = {{"callbacks", s.callbacks}, {"hits", s.hits}, {"misses", s.misses},
                {"payload_bytes", s.payload_bytes}, {"read_bytes", s.read_bytes},
                {"host_payload_bytes", s.host_payload_bytes}, {"device_payload_bytes", s.device_payload_bytes},
                {"host_expert_bytes", s.host_expert_bytes}, {"device_expert_bytes", s.device_expert_bytes},
                {"logical_expert_bytes", s.logical_expert_bytes}, {"physical_expert_bytes", s.physical_expert_bytes}};
        }
        std::ofstream out(prefix + ".json"); out << report.dump(2) << '\n';
        check(bool(out), "cannot write report");
        std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
