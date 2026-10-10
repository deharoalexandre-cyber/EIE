// EIE — Backend Implementations (CPU + CUDA + HIP)
// Apache License 2.0
//
// Built with the llama.cpp submodule (EIE_HAS_LLAMA), the CPU backend runs
// real GGUF inference. Without it, it falls back to placeholder responses
// so the scheduler/API layers can still be exercised standalone.

#include "compute_backend.h"
#include "text_output.h"
#include <iostream>
#include <chrono>

#ifdef EIE_HAS_LLAMA
#include "llama.h"
#include "common.h"
#include "chat.h"
#include "sampling.h"
#include "expert_stream.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <mutex>
#include <thread>
#include <algorithm>
#include <cmath>
#include <utility>
#include <atomic>
#include <ctime>
#endif

namespace eie {

// ═══════════════════════════════════════════════
// CPU Backend (also base for CUDA/HIP — same API)
// ═══════════════════════════════════════════════

#ifdef EIE_HAS_LLAMA

static void ensureBackendInit() {
    static std::once_flag once;
    std::call_once(once, [] { llama_backend_init(); });
}

// Map EIE KV type names to ggml types. TurboQuant KV types come from the
// llama-cpp-turboquant fork; on plain CPU builds the safe choices are
// f16 / q8_0 / q4_0.
static ggml_type mapKvType(const std::string& t) {
#if defined(__APPLE__) && defined(__aarch64__)
    // Apple Silicon : le cache KV vit sur Metal, qui n'a pas les noyaux
    // TurboQuant du fork (tq3_1s & co. font échouer le contexte). f16 d'office.
    if (t.rfind("turbo", 0) == 0) {
        std::cerr << "[KV] '" << t << "' non pris en charge par Metal, cache KV en f16" << std::endl;
        return GGML_TYPE_F16;
    }
#endif
    if (t == "f32")    return GGML_TYPE_F32;
    if (t == "f16")    return GGML_TYPE_F16;
    if (t == "q8_0")   return GGML_TYPE_Q8_0;
    if (t == "q4_0")   return GGML_TYPE_Q4_0;
    if (t == "turbo2" || t == "turbo3" || t == "turbo4") {
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            const auto type = static_cast<ggml_type>(i);
            const char * name = ggml_type_name(type);
            if (name && t == name) return type;
        }
    }
    std::cerr << "[KV] unknown type '" << t << "', using f16" << std::endl;
    return GGML_TYPE_F16;
}

// The GLM runtime adds vocabulary size to the penalty sampler API.
static llama_sampler * penaltySampler(llama_sampler * (*init)(int32_t, float, float, float),
                                      int32_t, float repeat, float present) {
    return init(256, repeat, 0.0f, present);
}
static llama_sampler * penaltySampler(llama_sampler * (*init)(int32_t, int32_t, float, float, float),
                                      int32_t n_vocab, float repeat, float present) {
    return init(n_vocab, 256, repeat, 0.0f, present);
}

static std::string answerOnlyPrompt(common_chat_params rendered) {
    // Some templates ignore enable_thinking=false and leave reasoning open.
    const auto & start = rendered.thinking_start_tag;
    if (rendered.supports_thinking && !start.empty() && !rendered.thinking_end_tags.empty() &&
        rendered.prompt.size() >= start.size() &&
        rendered.prompt.compare(rendered.prompt.size() - start.size(), start.size(), start) == 0) {
        rendered.prompt += rendered.thinking_end_tags.front();
    }
    return rendered.prompt;
}

class CpuBackend : public ComputeBackend {
protected:
    KvConfig kv_;
    std::string path_;
    int threads_ = 4;
    int gpu_id_ = 0;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    common_chat_templates_ptr tmpls_;
    std::mutex infer_mutex_;
    std::unique_ptr<ExpertStream> expert_stream_;
    bool trace_default_ = false;
    bool embed_mode_ = false;
    // KV-reuse : tokens (prompt + génération) encore présents dans le cache KV.
    // Le préfixe commun avec la requête suivante n'est pas re-préfillé.
    std::vector<llama_token> cache_tokens_;

    // (Re)crée le contexte en mode génération ou embedding (mêmes poids).
    llama_context* makeContext(bool embeddings) {
        if (embeddings && expert_stream_) return nullptr;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = kv_.n_ctx;
        cp.n_threads = threads_;
        cp.n_threads_batch = threads_;
        if (embeddings) {
            cp.embeddings = true;
            cp.n_batch = cp.n_ubatch = 2048; // encodeurs : la séquence en un seul ubatch
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        } else {
            cp.flash_attn_type = kv_.flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO
                                                : LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.type_k = mapKvType(kv_.type_k);
            cp.type_v = mapKvType(kv_.type_v);
        }
        if (expert_stream_) expert_stream_->configure(cp);
        return llama_init_from_model(model_, cp);
    }

    bool switchMode(bool embeddings) {
        if (embed_mode_ == embeddings && ctx_) return true;
        llama_context* nctx = makeContext(embeddings);
        if (!nctx) return false;
        if (ctx_) llama_free(ctx_);
        ctx_ = nctx;
        embed_mode_ = embeddings;
        cache_tokens_.clear(); // nouveau contexte : cache KV perdu
        return true;
    }

public:
    ~CpuBackend() override { unload(); }

    BackendType type() const override { return BackendType::CPU; }
    std::string name() const override { return "CPU"; }

    bool init(int gpu_id) override {
        gpu_id_ = gpu_id;
        ensureBackendInit();
        std::cout << "[" << name() << "] GPU " << gpu_id << " init" << std::endl;
        return true;
    }

    bool load(const ModelParams& p) override {
        ensureBackendInit();
        path_ = p.path;
        trace_default_ = p.ews_trace;
        alias = p.alias;
        threads_ = p.n_threads > 0 ? p.n_threads
                 : (int)std::max(1u, std::thread::hardware_concurrency() / 2);
        kv_ = p.kv;

        llama_model_params mp = llama_model_default_params();
        const llama_model_tensor_buft_override cpu_experts[] = {
            {"\\.ffn_.*_exps\\.weight", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
        if (p.cpu_moe) mp.tensor_buft_overrides = cpu_experts;
        // Mac Intel : jamais d'offload — ggml choisirait Metal, qui produit du
        // bruit sur les iGPU non-Apple-Silicon. Ailleurs (Apple Silicon inclus,
        // Metal fiable), on respecte n_gpu_layers demandé.
#if defined(__APPLE__) && defined(__x86_64__)
        mp.n_gpu_layers = 0;
#else
        mp.n_gpu_layers = p.n_gpu_layers;
#endif
        if (p.ews_slots != 0) {
#ifndef EIE_HAS_EWS
            std::cerr << "EWS is unavailable with the PrismML llama.cpp runtime" << std::endl;
            return false;
#else
            try { expert_stream_ = std::make_unique<ExpertStream>(path_, p.ews_slots); }
            catch (const std::exception & e) { std::cerr << e.what() << std::endl; return false; }
            mp.ews_n_slots = p.ews_slots;
            mp.load_mode = LLAMA_LOAD_MODE_NONE;
            mp.use_extra_bufts = false;
#endif
        }
        model_ = llama_model_load_from_file(path_.c_str(), mp);
        if (!model_) {
            std::cerr << "[" << name() << "] failed to load model: " << path_ << std::endl;
            return false;
        }
        if (expert_stream_) {
            try { expert_stream_->bind(model_); }
            catch (const std::exception & e) { std::cerr << e.what() << std::endl; unload(); return false; }
        }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = kv_.n_ctx;
        cp.n_threads = threads_;
        cp.n_threads_batch = threads_;
        cp.flash_attn_type = kv_.flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO
                                            : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = mapKvType(kv_.type_k);
        cp.type_v = mapKvType(kv_.type_v);

        if (expert_stream_) expert_stream_->configure(cp);
        ctx_ = llama_init_from_model(model_, cp);
        if (!ctx_ && (cp.type_k != GGML_TYPE_F16 || cp.type_v != GGML_TYPE_F16)) {
            std::cerr << "[" << name() << "] context init failed with kv="
                      << kv_.type_k << "/" << kv_.type_v << ", retrying f16" << std::endl;
            cp.type_k = GGML_TYPE_F16;
            cp.type_v = GGML_TYPE_F16;
            kv_.type_k = kv_.type_v = "f16";
            ctx_ = llama_init_from_model(model_, cp);
        }
        if (!ctx_) {
            llama_model_free(model_);
            model_ = nullptr;
            return false;
        }

        // Template de chat natif du modèle (métadonnées GGUF)
        try {
            tmpls_ = common_chat_templates_init(model_, "");
        } catch (const std::exception& e) {
            std::cerr << "[" << name() << "] no usable chat template: " << e.what() << std::endl;
        }

        loaded = true;
        std::cout << "[" << name() << "] loaded: " << alias
                  << " kv=" << kv_.type_k << "/" << kv_.type_v
                  << " ctx=" << kv_.n_ctx
                  << " threads=" << threads_ << std::endl;
        return true;
    }

    std::string formatChat(const std::vector<ChatMessage>& msgs) override {
        return formatChatWithOptions(msgs, {});
    }

    std::string formatChatWithOptions(const std::vector<ChatMessage>& msgs,
                                      const ChatFormatOptions& options) override {
        return formatChatRequest(msgs, options).prompt;
    }

    ChatPrompt formatChatRequest(const std::vector<ChatMessage>& msgs,
                                 const ChatFormatOptions& options) override {
        ChatPrompt out;
        const bool wants_tools = !options.tools.empty() && options.tool_choice != ChatToolChoice::None;
        if (!tmpls_ || msgs.empty()) {
            if (wants_tools) out.error = "tool calling requires a model chat template and messages";
            return out;
        }
        if (wants_tools) {
            const auto caps = common_chat_templates_get_caps(tmpls_.get());
            if (!caps.at("supports_tools") || !caps.at("supports_tool_calls")) {
                out.error = "model chat template cannot describe and parse tool calls";
                return out;
            }
        }
        common_chat_templates_inputs in;
        for (const auto& m : msgs) {
            common_chat_msg cm;
            cm.role = m.role;
            cm.content = m.content;
            cm.reasoning_content = m.reasoning_content;
            cm.tool_name = m.name;
            cm.tool_call_id = m.tool_call_id;
            for (const auto& call : m.tool_calls)
                cm.tool_calls.push_back({call.name, call.arguments, call.id});
            in.messages.push_back(std::move(cm));
        }
        for (const auto& tool : options.tools)
            in.tools.push_back({tool.name, tool.description, tool.parameters});
        in.tool_choice = options.tool_choice == ChatToolChoice::Required ? COMMON_CHAT_TOOL_CHOICE_REQUIRED :
                         options.tool_choice == ChatToolChoice::None ? COMMON_CHAT_TOOL_CHOICE_NONE :
                         COMMON_CHAT_TOOL_CHOICE_AUTO;
        in.parallel_tool_calls = options.parallel_tool_calls;
        in.add_generation_prompt = true;
        in.enable_thinking = options.enable_thinking;
        if (!options.tools.empty()) in.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        if (options.reasoning_effort == "low" || options.reasoning_effort == "medium" ||
            options.reasoning_effort == "high" || options.reasoning_effort == "xhigh")
            in.chat_template_kwargs["reasoning_effort"] = "\"" + options.reasoning_effort + "\"";
        try {
            auto rendered = common_chat_templates_apply(tmpls_.get(), in);
            if (wants_tools && (rendered.format == COMMON_CHAT_FORMAT_CONTENT_ONLY || rendered.parser.empty())) {
                out.error = "model chat template does not support structured tool calls";
                return out;
            }
            if (wants_tools && options.tool_choice == ChatToolChoice::Required && rendered.grammar.empty()) {
                out.error = "model chat template cannot enforce tool_choice=required";
                return out;
            }
            out.prompt = options.enable_thinking ? rendered.prompt : answerOnlyPrompt(rendered);
            out.parse_chat_output = !options.tools.empty() && !rendered.parser.empty();
            out.parse_tool_calls = wants_tools;
            out.parser_format = static_cast<int>(rendered.format);
            out.parser = std::move(rendered.parser);
            out.generation_prompt = rendered.generation_prompt + out.prompt.substr(rendered.prompt.size());
            out.grammar = std::move(rendered.grammar);
            out.grammar_lazy = rendered.grammar_lazy;
            for (const auto& trigger : rendered.grammar_triggers)
                out.grammar_triggers.push_back({static_cast<int>(trigger.type), trigger.value, static_cast<int>(trigger.token)});
            out.preserved_tokens = std::move(rendered.preserved_tokens);
            out.additional_stops = std::move(rendered.additional_stops);
            return out;
        } catch (const std::exception& e) {
            std::cerr << "[" << name() << "] chat template failed: " << e.what() << std::endl;
            if (wants_tools) out.error = std::string("chat template failed: ") + e.what();
            return out;
        }
    }

    InferenceResult chat(const std::string& prompt, const SamplingParams& s) override {
        auto t0 = std::chrono::steady_clock::now();
        InferenceResult r;
        r.model = alias;

        if (!ctx_ || !model_) {
            r.ok = false;
            r.error = "model not loaded";
            return r;
        }

        bool trace_cancelled = false;
        auto cancelled = [&] { return s.should_continue && !s.should_continue(); };
        auto mark_cancelled = [&] {
            trace_cancelled = true;
            r.ok = false;
            r.error = "request cancelled";
            r.finish_reason = "cancelled";
            r.latency_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        };
        if (cancelled()) { mark_cancelled(); return r; }
        std::lock_guard<std::mutex> lock(infer_mutex_);
        // A request may have disconnected while waiting for another inference.
        if (cancelled()) { mark_cancelled(); return r; }

        if (expert_stream_) expert_stream_->beginTrace(trace_default_ || s.ews_trace);
        struct TraceEnd {
            ExpertStream* stream;
            InferenceResult& result;
            bool& cancelled;
            int exceptions = std::uncaught_exceptions();
            ~TraceEnd() {
                if (stream) stream->endTrace(std::uncaught_exceptions() > exceptions ? "exception" :
                    cancelled ? "cancelled" : result.ok ? "complete" : "error");
            }
        } trace_end{expert_stream_.get(), r, trace_cancelled};

        if (expert_stream_ && !expert_stream_->error().empty()) {
            r.ok = false; r.error = expert_stream_->error(); return r;
        }
        if (!switchMode(false)) {
            r.ok = false;
            r.error = "context switch failed";
            return r;
        }

        // One-shot : contexte éphémère, le cache KV du chat reste intact
        // (fix « bug n°1 » d'Elyne Mobile, version serveur).
        llama_context* run_ctx = ctx_;
        if (s.one_shot) {
            run_ctx = makeContext(false);
            if (!run_ctx) {
                r.ok = false;
                r.error = "one-shot context failed";
                return r;
            }
        }

        auto tokens = common_tokenize(run_ctx, prompt, true, true);
        int n_ctx = (int)llama_n_ctx(run_ctx);
        bool truncated = false;
        if (!s.truncate_prompt && (int64_t)tokens.size() + s.max_tokens > n_ctx) {
            if (s.one_shot) llama_free(run_ctx);
            r.ok = false;
            r.error = "context_length_exceeded";
            return r;
        }
        if (s.truncate_prompt && (int)tokens.size() >= n_ctx - 8) {
            // keep the tail of the prompt if it does not fit
            tokens.erase(tokens.begin(), tokens.end() - (n_ctx - 8));
            truncated = true;
        }
        if (tokens.empty()) {
            if (s.one_shot) llama_free(run_ctx);
            r.ok = false;
            r.error = "empty prompt";
            return r;
        }
        r.prompt_tokens = (int)tokens.size();

        // KV-reuse (D12 du mobile, porté serveur) : le préfixe commun avec la
        // requête précédente reste dans le cache KV ; seul le delta est préfillé.
        // Jamais en one-shot : contexte vierge, cache du chat non consulté.
        size_t prefix = 0;
        if (!truncated && !s.one_shot) {
            while (prefix < cache_tokens_.size() && prefix < tokens.size() - 1 &&
                   cache_tokens_[prefix] == tokens[prefix]) prefix++;
        }
        auto* mem = llama_get_memory(run_ctx);
        if (prefix > 0 && llama_memory_seq_rm(mem, 0, (llama_pos)prefix, -1)) {
            // cache conservé jusqu'à `prefix` (seq_rm peut échouer sur SWA -> repli)
        } else {
            llama_memory_clear(mem, true);
            prefix = 0;
        }

        const llama_vocab* vocab = llama_model_get_vocab(model_);
        const int chunk = expert_stream_ ? 1 : 512;
        llama_batch batch = llama_batch_init(chunk, 0, 1);

        // Prompt evaluation (chunked), à partir du préfixe réutilisé
        bool decode_ok = true;
        bool was_cancelled = false;
        if (expert_stream_) expert_stream_->tracePhase(RoutingPhase::Prefill);
        for (size_t off = prefix; off < tokens.size(); off += chunk) {
            if (cancelled()) { was_cancelled = true; break; }
            size_t end = std::min(tokens.size(), off + chunk);
            common_batch_clear(batch);
            for (size_t i = off; i < end; i++) {
                common_batch_add(batch, tokens[i], (llama_pos)i, {0},
                                 i == tokens.size() - 1);
            }
            if (llama_decode(run_ctx, batch) != 0 || (expert_stream_ && !expert_stream_->error().empty())) {
                decode_ok = false; break;
            }
        }
        r.reused_tokens = s.one_shot ? -1 : (int)prefix;
        if (was_cancelled || !decode_ok) {
            llama_batch_free(batch);
            if (s.one_shot) llama_free(run_ctx);
            if (was_cancelled) {
                if (!s.one_shot) {
                    llama_memory_clear(mem, true);
                    cache_tokens_.clear();
                }
                mark_cancelled();
                return r;
            }
            r.ok = false;
            r.error = expert_stream_ && !expert_stream_->error().empty() ? expert_stream_->error() : "prompt decode failed";
            if (!s.one_shot) cache_tokens_.clear();
            return r;
        }

        // Tool templates supply a grammar (lazy for auto, strict for required).
        // Use llama.cpp's shared sampler here so its trigger and prefill behavior
        // matches llama-server. Plain chat keeps EIE's historical sampler chain.
        common_sampler_ptr tool_sampler;
        llama_sampler* smpl = nullptr;
        if (s.chat_format.parse_tool_calls && !s.chat_format.grammar.empty()) {
            common_params_sampling params;
            params.top_k = s.top_k;
            params.top_p = s.top_p;
            params.min_p = s.min_p;
            params.temp = s.temperature;
            params.penalty_last_n = 256;
            params.penalty_repeat = s.repetition_penalty;
            params.penalty_present = s.presence_penalty;
            params.min_keep = 1;
            params.samplers = {COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_K,
                               COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_MIN_P,
                               COMMON_SAMPLER_TYPE_TEMPERATURE};
            params.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, s.chat_format.grammar};
            params.grammar_lazy = s.chat_format.grammar_lazy;
            params.generation_prompt = s.chat_format.generation_prompt;
            for (const auto& token : s.chat_format.preserved_tokens) {
                auto ids = common_tokenize(vocab, token, false, true);
                if (ids.size() == 1) params.preserved_tokens.insert(ids.front());
            }
            for (const auto& trigger : s.chat_format.grammar_triggers) {
                auto type = static_cast<common_grammar_trigger_type>(trigger.type);
                if (type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    auto ids = common_tokenize(vocab, trigger.value, false, true);
                    if (ids.size() == 1 && params.preserved_tokens.count(ids.front())) {
                        params.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
                                                          trigger.value, ids.front()});
                        continue;
                    }
                }
                params.grammar_triggers.push_back({type, trigger.value,
                    static_cast<llama_token>(trigger.token)});
            }
            try {
                tool_sampler.reset(common_sampler_init(model_, params));
            } catch (const std::exception& e) {
                llama_batch_free(batch);
                if (s.one_shot) llama_free(run_ctx);
                if (!s.one_shot) cache_tokens_.clear();
                r.ok = false;
                r.error = std::string("tool grammar initialization failed: ") + e.what();
                return r;
            }
            if (!tool_sampler) {
                llama_batch_free(batch);
                if (s.one_shot) llama_free(run_ctx);
                if (!s.one_shot) cache_tokens_.clear();
                r.ok = false;
                r.error = "tool grammar initialization failed";
                return r;
            }
        } else {
            smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
            // Request overrides preserve the historical EIE repetition default.
            llama_sampler_chain_add(smpl, penaltySampler(llama_sampler_init_penalties,
                llama_vocab_n_tokens(vocab), s.repetition_penalty, s.presence_penalty));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(s.top_k));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(s.top_p, 1));
            if (s.min_p > 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(s.min_p, 1));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(s.temperature));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        }

        // Generation loop
        TextOutput output(s.stop, s.on_token);
        if (!s.one_shot) cache_tokens_ = tokens; // les tokens générés s'y ajoutent
        int n_past = (int)tokens.size();
        r.finish_reason = "length";
        if (expert_stream_) expert_stream_->tracePhase(RoutingPhase::Decode);
        for (int i = 0; i < s.max_tokens && n_past < n_ctx; i++) {
            if (cancelled()) { was_cancelled = true; break; }
            llama_token id = tool_sampler ? common_sampler_sample(tool_sampler.get(), run_ctx, -1)
                                          : llama_sampler_sample(smpl, run_ctx, -1);
            if (tool_sampler) common_sampler_accept(tool_sampler.get(), id, true);
            ++r.tokens; // Include a sampled EOG or token containing a stop sequence.
            if (llama_vocab_is_eog(vocab, id)) { r.finish_reason = "stop"; break; }
            std::string piece = common_token_to_piece(run_ctx, id);
            if (!output.push(piece)) {
                was_cancelled = output.cancelled();
                if (output.stopped()) r.finish_reason = "stop";
                break;
            }
            common_batch_clear(batch);
            common_batch_add(batch, id, n_past++, {0}, true);
            if (llama_decode(run_ctx, batch) != 0 || (expert_stream_ && !expert_stream_->error().empty())) {
                decode_ok = false; break;
            }
            if (!s.one_shot) cache_tokens_.push_back(id); // Only tokens actually decoded into KV.
        }
        if (decode_ok && !was_cancelled && !output.finish()) was_cancelled = true;

        if (smpl) llama_sampler_free(smpl);
        llama_batch_free(batch);
        if (s.one_shot) llama_free(run_ctx);

        r.text = output.text();
        r.ok = decode_ok;
        if (was_cancelled) {
            // Discard partial persistent KV; a one-shot leaves the chat intact.
            if (!s.one_shot) {
                llama_memory_clear(mem, true);
                cache_tokens_.clear();
            }
            mark_cancelled();
        }
        if (!decode_ok) {
            r.error = expert_stream_ && !expert_stream_->error().empty() ? expert_stream_->error() : "generation decode failed";
            if (!s.one_shot) cache_tokens_.clear();
        }

        if (r.ok && s.chat_format.parse_chat_output && r.finish_reason != "cancelled") {
            try {
                common_chat_parser_params parser_params;
                parser_params.format = static_cast<common_chat_format>(s.chat_format.parser_format);
                parser_params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
                parser_params.generation_prompt = s.chat_format.generation_prompt;
                parser_params.parser.load(s.chat_format.parser);
                auto parsed = common_chat_parse(r.text, r.finish_reason == "length", parser_params);
                r.text = parsed.content;
                r.reasoning_content = parsed.reasoning_content;
                if (r.finish_reason != "length" && s.chat_format.parse_tool_calls) {
                    static std::atomic<uint64_t> next_call_id{0};
                    for (const auto& call : parsed.tool_calls) {
                        std::string id = call.id;
                        if (id.empty()) id = "call_eie_" + std::to_string(std::time(nullptr)) + "_" +
                                               std::to_string(++next_call_id);
                        r.tool_calls.push_back({id, call.name, call.arguments});
                    }
                    if (!r.tool_calls.empty()) r.finish_reason = "tool_calls";
                }
            } catch (const std::exception& e) {
                r.ok = false;
                r.error = std::string("tool response parse failed: ") + e.what();
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        r.latency_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        return r;
    }

    std::vector<float> embed(const std::string& text) override {
        if (!model_) return {};
        std::lock_guard<std::mutex> lock(infer_mutex_);
        if (!switchMode(true)) return {};

        llama_memory_clear(llama_get_memory(ctx_), true);
        auto tokens = common_tokenize(ctx_, text, true, true);
        if (tokens.empty()) return {};
        if ((int)tokens.size() > 2048) tokens.resize(2048);

        llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
        common_batch_clear(batch);
        for (size_t i = 0; i < tokens.size(); i++)
            common_batch_add(batch, tokens[i], (llama_pos)i, {0}, true);
        int rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0) return {};

        const float* v = llama_get_embeddings_seq(ctx_, 0);
        if (!v) return {};
        int n = llama_model_n_embd(model_);
        std::vector<float> out(v, v + n);
        // Normalisation L2 (cosinus = produit scalaire côté client)
        float norm = 0;
        for (float x : out) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 0) for (float& x : out) x /= norm;
        return out;
    }

    VramStatus vram() override {
        VramStatus result{gpu_id_, 0, 0, 0};
        if (type() == BackendType::CPU) return result;
        int index = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            if (index++ != gpu_id_) continue;
            ggml_backend_dev_memory(dev, &result.free_bytes, &result.total_bytes);
            result.used_bytes = result.total_bytes - result.free_bytes;
            break;
        }
        return result;
    }

    RoutingHistogram streamingRouting() override {
        std::lock_guard<std::mutex> lock(infer_mutex_);
        return expert_stream_ ? expert_stream_->routing() : RoutingHistogram{};
    }

    std::map<std::string, uint64_t> streamingStats() override {
        std::lock_guard<std::mutex> lock(infer_mutex_);
        if (!expert_stream_) return {};
        auto s = expert_stream_->stats();
        return {{"callbacks", s.callbacks}, {"hits", s.hits}, {"misses", s.misses},
            {"payload_bytes", s.payload_bytes}, {"read_bytes", s.read_bytes},
            {"host_payload_bytes", s.host_payload_bytes}, {"device_payload_bytes", s.device_payload_bytes},
            {"host_expert_bytes", s.host_expert_bytes}, {"device_expert_bytes", s.device_expert_bytes},
            {"logical_expert_bytes", s.logical_expert_bytes}, {"physical_expert_bytes", s.physical_expert_bytes}};
    }

    HealthStatus health() override {
        return HealthStatus{loaded, 0, loaded ? "OK" : "not loaded"};
    }

    void unload() override {
        tmpls_.reset();
        if (ctx_)   { llama_free(ctx_); ctx_ = nullptr; }
        if (model_) { llama_model_free(model_); model_ = nullptr; }
        expert_stream_.reset();
        cache_tokens_.clear();
        if (loaded) std::cout << "[" << name() << "] unloaded: " << alias << std::endl;
        loaded = false;
    }

    bool adaptKv(const KvConfig& kv) override {
        std::cout << "[" << name() << "] adapt KV " << kv_.type_v << " -> " << kv.type_v << std::endl;
        // Reallocate the context with the new KV quantization (weights stay loaded)
        std::lock_guard<std::mutex> lock(infer_mutex_);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = kv.n_ctx;
        cp.n_threads = threads_;
        cp.n_threads_batch = threads_;
        cp.flash_attn_type = kv.flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO
                                           : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = mapKvType(kv.type_k);
        cp.type_v = mapKvType(kv.type_v);
        if (expert_stream_) expert_stream_->configure(cp);
        llama_context* nctx = llama_init_from_model(model_, cp);
        if (!nctx) return false;
        if (ctx_) llama_free(ctx_);
        ctx_ = nctx;
        kv_ = kv;
        embed_mode_ = false;
        cache_tokens_.clear();
        return true;
    }
};

#else // !EIE_HAS_LLAMA — placeholder backend (standalone scheduler/API testing)

class CpuBackend : public ComputeBackend {
protected:
    KvConfig kv_;
    std::string path_;
    int threads_ = 4;
    int gpu_id_ = 0;

public:
    BackendType type() const override { return BackendType::CPU; }
    std::string name() const override { return "CPU"; }

    bool init(int gpu_id) override {
        gpu_id_ = gpu_id;
        std::cout << "[" << name() << "] GPU " << gpu_id << " init" << std::endl;
        return true;
    }

    bool load(const ModelParams& p) override {
        path_ = p.path;
        alias = p.alias;
        threads_ = p.n_threads;
        kv_ = p.kv;
        loaded = true;
        if (p.ews_slots != 0) {
            loaded = false;
            std::cerr << "EWS requires the patched llama.cpp runtime, not the placeholder backend" << std::endl;
            return false;
        }
        std::cout << "[" << name() << "] loaded (placeholder): " << alias
                  << " kv=" << kv_.type_k << "/" << kv_.type_v
                  << " ctx=" << kv_.n_ctx << std::endl;
        return true;
    }

    InferenceResult chat(const std::string& prompt, const SamplingParams& s) override {
        auto t0 = std::chrono::steady_clock::now();
        InferenceResult r;
        r.model = alias;
        r.text = "[" + alias + " response to: " + prompt.substr(0, 50) + "...]";
        r.tokens = 42;
        r.ok = true;
        auto t1 = std::chrono::steady_clock::now();
        r.latency_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        return r;
    }

    VramStatus vram() override { return VramStatus{gpu_id_, 0, 0, 0}; }

    HealthStatus health() override {
        return HealthStatus{loaded, 0, loaded ? "OK" : "not loaded"};
    }

    void unload() override {
        loaded = false;
        std::cout << "[" << name() << "] unloaded: " << alias << std::endl;
    }

    bool adaptKv(const KvConfig& kv) override {
        std::cout << "[" << name() << "] adapt KV " << kv_.type_v << " -> " << kv.type_v << std::endl;
        kv_ = kv;
        return true;
    }
};

#endif // EIE_HAS_LLAMA

// ═══════════════════════════════════════════════
// CUDA Backend — inherits CPU, overrides GPU-specific
// ═══════════════════════════════════════════════

class CudaBackend : public CpuBackend {
public:
    BackendType type() const override { return BackendType::CUDA; }
    std::string name() const override { return "CUDA"; }

    bool init(int gpu_id) override {
        gpu_id_ = gpu_id;
        // cudaSetDevice(gpu_id);
        // cudaDeviceProp props; cudaGetDeviceProperties(&props, gpu_id);
        // std::cout << "[CUDA] " << props.name << " " << props.totalGlobalMem/(1<<30) << " GB" << std::endl;
        std::cout << "[CUDA] GPU " << gpu_id << " initialized" << std::endl;
        return CpuBackend::init(gpu_id);
    }

    VramStatus vram() override {
#ifdef EIE_HAS_LLAMA
        return CpuBackend::vram();
#else
        VramStatus s;
        s.gpu_id = gpu_id_;
        // size_t free, total;
        // cudaMemGetInfo(&free, &total);
        // s.total_bytes = total; s.free_bytes = free; s.used_bytes = total - free;
        s.total_bytes = 16ULL << 30;  // placeholder 16 GB
        s.free_bytes = 8ULL << 30;
        s.used_bytes = s.total_bytes - s.free_bytes;
        return s;
#endif
    }
};

// ═══════════════════════════════════════════════
// HIP/ROCm Backend — same pattern
// ═══════════════════════════════════════════════

class HipBackend : public CpuBackend {
public:
    BackendType type() const override { return BackendType::HIP; }
    std::string name() const override { return "ROCm"; }

    bool init(int gpu_id) override {
        gpu_id_ = gpu_id;
        // hipSetDevice(gpu_id);
        std::cout << "[ROCm] GPU " << gpu_id << " initialized" << std::endl;
        return CpuBackend::init(gpu_id);
    }

    VramStatus vram() override {
#ifdef EIE_HAS_LLAMA
        return CpuBackend::vram();
#else
        VramStatus s;
        s.gpu_id = gpu_id_;
        // hipMemGetInfo(&s.free_bytes, &s.total_bytes);
        s.total_bytes = 48ULL << 30;  // placeholder 48 GB (W7900)
        s.free_bytes = 40ULL << 30;
        s.used_bytes = s.total_bytes - s.free_bytes;
        return s;
#endif
    }
};

// ═══════════════════════════════════════════════
// Auto-detection
// ═══════════════════════════════════════════════

std::unique_ptr<ComputeBackend> detectBackend(int gpu_id) {
#ifdef GGML_USE_CUDA
    {
        auto b = std::make_unique<CudaBackend>();
        if (b->init(gpu_id)) return b;
    }
#endif
#ifdef GGML_USE_HIP
    {
        auto b = std::make_unique<HipBackend>();
        if (b->init(gpu_id)) return b;
    }
#endif
    auto b = std::make_unique<CpuBackend>();
    b->init(0);
    return b;
}

} // namespace eie
