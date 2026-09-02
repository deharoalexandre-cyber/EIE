// EIE — Backend Implementations (CPU + CUDA + HIP)
// Apache License 2.0
//
// Built with the llama.cpp submodule (EIE_HAS_LLAMA), the CPU backend runs
// real GGUF inference. Without it, it falls back to placeholder responses
// so the scheduler/API layers can still be exercised standalone.

#include "compute_backend.h"
#include <iostream>
#include <chrono>

#ifdef EIE_HAS_LLAMA
#include "llama.h"
#include "common.h"
#include "chat.h"
#include <mutex>
#include <thread>
#include <algorithm>
#include <cmath>
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
    if (t == "f32")    return GGML_TYPE_F32;
    if (t == "f16")    return GGML_TYPE_F16;
    if (t == "q8_0")   return GGML_TYPE_Q8_0;
    if (t == "q4_0")   return GGML_TYPE_Q4_0;
    if (t == "turbo2") return GGML_TYPE_TQ2_0;
    if (t == "turbo3") return GGML_TYPE_TQ3_1S;
    if (t == "turbo4") return GGML_TYPE_TQ4_1S;
    std::cerr << "[KV] unknown type '" << t << "', using f16" << std::endl;
    return GGML_TYPE_F16;
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
    bool embed_mode_ = false;
    // KV-reuse : tokens (prompt + génération) encore présents dans le cache KV.
    // Le préfixe commun avec la requête suivante n'est pas re-préfillé.
    std::vector<llama_token> cache_tokens_;

    // (Re)crée le contexte en mode génération ou embedding (mêmes poids).
    llama_context* makeContext(bool embeddings) {
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
        path_ = p.path;
        alias = p.alias;
        threads_ = p.n_threads > 0 ? p.n_threads
                 : (int)std::max(1u, std::thread::hardware_concurrency() / 2);
        kv_ = p.kv;

        llama_model_params mp = llama_model_default_params();
        // CPU backend: never offload. On Intel Macs ggml would otherwise pick
        // the Metal backend, which produces garbage on non-Apple-Silicon GPUs.
        mp.n_gpu_layers = (type() == BackendType::CPU) ? 0 : p.n_gpu_layers;
        model_ = llama_model_load_from_file(path_.c_str(), mp);
        if (!model_) {
            std::cerr << "[" << name() << "] failed to load model: " << path_ << std::endl;
            return false;
        }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = kv_.n_ctx;
        cp.n_threads = threads_;
        cp.n_threads_batch = threads_;
        cp.flash_attn_type = kv_.flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO
                                            : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = mapKvType(kv_.type_k);
        cp.type_v = mapKvType(kv_.type_v);

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
        if (!tmpls_ || msgs.empty()) return "";
        common_chat_templates_inputs in;
        for (auto& m : msgs) {
            common_chat_msg cm;
            cm.role = m.role;
            cm.content = m.content;
            in.messages.push_back(cm);
        }
        in.add_generation_prompt = true;
        // Pas de canal de "réflexion" par défaut : la réponse est la réponse.
        in.enable_thinking = false;
        try {
            return common_chat_templates_apply(tmpls_.get(), in).prompt;
        } catch (const std::exception& e) {
            std::cerr << "[" << name() << "] chat template failed: " << e.what() << std::endl;
            return "";
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

        std::lock_guard<std::mutex> lock(infer_mutex_);

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
        if ((int)tokens.size() >= n_ctx - 8) {
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
        const int chunk = 512;
        llama_batch batch = llama_batch_init(chunk, 0, 1);

        // Prompt evaluation (chunked), à partir du préfixe réutilisé
        bool decode_ok = true;
        for (size_t off = prefix; off < tokens.size(); off += chunk) {
            size_t end = std::min(tokens.size(), off + chunk);
            common_batch_clear(batch);
            for (size_t i = off; i < end; i++) {
                common_batch_add(batch, tokens[i], (llama_pos)i, {0},
                                 i == tokens.size() - 1);
            }
            if (llama_decode(run_ctx, batch) != 0) { decode_ok = false; break; }
        }
        r.reused_tokens = s.one_shot ? -1 : (int)prefix;
        if (!decode_ok) {
            llama_batch_free(batch);
            if (s.one_shot) llama_free(run_ctx);
            r.ok = false;
            r.error = "prompt decode failed";
            return r;
        }

        // Sampler chain
        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        // Pénalité de répétition : les petits modèles bouclent vite en complétion brute
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(256, 1.15f, 0.0f, 0.0f));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(s.top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(s.top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(s.temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        // Generation loop
        std::string output;
        if (!s.one_shot) cache_tokens_ = tokens; // les tokens générés s'y ajoutent
        int n_past = (int)tokens.size();
        for (int i = 0; i < s.max_tokens && n_past < n_ctx; i++) {
            llama_token id = llama_sampler_sample(smpl, run_ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) break;
            std::string piece = common_token_to_piece(run_ctx, id);
            output += piece;
            if (!s.one_shot) cache_tokens_.push_back(id);
            r.tokens++;
            if (s.on_token && s.stop.empty() && !s.on_token(piece)) break; // flux (sans stop-sequences)
            // Séquences d'arrêt : coupe dès qu'une apparaît en fin de sortie
            bool stopped = false;
            for (auto& st : s.stop) {
                auto pos = output.rfind(st);
                if (pos != std::string::npos && pos + st.size() >= output.size()) {
                    output.erase(pos);
                    stopped = true;
                    break;
                }
            }
            if (stopped) break;
            common_batch_clear(batch);
            common_batch_add(batch, id, n_past++, {0}, true);
            if (llama_decode(run_ctx, batch) != 0) break;
        }

        llama_sampler_free(smpl);
        llama_batch_free(batch);
        if (s.one_shot) llama_free(run_ctx);

        r.text = output;
        r.ok = true;

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

    VramStatus vram() override { return VramStatus{gpu_id_, 0, 0, 0}; }

    HealthStatus health() override {
        return HealthStatus{loaded, 0, loaded ? "OK" : "not loaded"};
    }

    void unload() override {
        tmpls_.reset();
        if (ctx_)   { llama_free(ctx_); ctx_ = nullptr; }
        if (model_) { llama_model_free(model_); model_ = nullptr; }
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
        llama_context* nctx = llama_init_from_model(model_, cp);
        if (!nctx) return false;
        if (ctx_) llama_free(ctx_);
        ctx_ = nctx;
        kv_ = kv;
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
        return true;
    }

    VramStatus vram() override {
        VramStatus s;
        s.gpu_id = gpu_id_;
        // size_t free, total;
        // cudaMemGetInfo(&free, &total);
        // s.total_bytes = total; s.free_bytes = free; s.used_bytes = total - free;
        s.total_bytes = 16ULL << 30;  // placeholder 16 GB
        s.free_bytes = 8ULL << 30;
        s.used_bytes = s.total_bytes - s.free_bytes;
        return s;
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
        return true;
    }

    VramStatus vram() override {
        VramStatus s;
        s.gpu_id = gpu_id_;
        // hipMemGetInfo(&s.free_bytes, &s.total_bytes);
        s.total_bytes = 48ULL << 30;  // placeholder 48 GB (W7900)
        s.free_bytes = 40ULL << 30;
        s.used_bytes = s.total_bytes - s.free_bytes;
        return s;
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
