// EIE — Backend Implementations (CPU + CUDA + HIP)
// Apache License 2.0
//
// In production: replace placeholder inference with llama.cpp calls:
//   #include "llama.h"
//   llama_model_load_from_file(), llama_init_from_model(),
//   llama_decode(), llama_sampling(), llama_free()

#include "compute_backend.h"
#include <iostream>
#include <chrono>

namespace eie {

// ═══════════════════════════════════════════════
// CPU Backend (also base for CUDA/HIP — same API)
// ═══════════════════════════════════════════════

class CpuBackend : public ComputeBackend {
protected:
    KvConfig kv_;
    std::string path_;
    int threads_ = 4;
    int gpu_id_ = 0;
    // llama_model * model_ = nullptr;
    // llama_context * ctx_ = nullptr;

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

        // ── llama.cpp integration point ──
        // llama_model_params mp = llama_model_default_params();
        // mp.n_gpu_layers = p.n_gpu_layers;
        // model_ = llama_model_load_from_file(path_.c_str(), mp);
        // if (!model_) return false;
        //
        // llama_context_params cp = llama_context_default_params();
        // cp.n_ctx = kv_.n_ctx;
        // cp.flash_attn = kv_.flash_attn;
        //
        // Map KV types:
        //   "f16"    -> GGML_TYPE_F16
        //   "q8_0"   -> GGML_TYPE_Q8_0
        //   "q4_0"   -> GGML_TYPE_Q4_0
        //   "turbo2" -> GGML_TYPE_TQ2_0
        //   "turbo3" -> GGML_TYPE_TQ3_0
        //   "turbo4" -> GGML_TYPE_TQ4_0
        //
        // cp.type_k = map_kv_type(kv_.type_k);
        // cp.type_v = map_kv_type(kv_.type_v);
        // ctx_ = llama_init_from_model(model_, cp);
        // if (!ctx_) { llama_model_free(model_); return false; }

        loaded = true;
        std::cout << "[" << name() << "] loaded: " << alias
                  << " kv=" << kv_.type_k << "/" << kv_.type_v
                  << " ctx=" << kv_.n_ctx << std::endl;
        return true;
    }

    InferenceResult chat(const std::string& prompt, const SamplingParams& s) override {
        auto t0 = std::chrono::steady_clock::now();
        InferenceResult r;
        r.model = alias;

        // ── llama.cpp inference loop ──
        // 1. Tokenize prompt:
        //    auto tokens = llama_tokenize(model_, prompt.c_str(), true);
        //
        // 2. Create batch and decode:
        //    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        //    for (size_t i = 0; i < tokens.size(); i++)
        //        llama_batch_add(batch, tokens[i], i, {0}, false);
        //    batch.logits[batch.n_tokens - 1] = true;
        //    llama_decode(ctx_, batch);
        //
        // 3. Sampling loop:
        //    auto sampler = llama_sampler_chain_init({});
        //    llama_sampler_chain_add(sampler, llama_sampler_init_temp(s.temperature));
        //    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(s.top_p, 1));
        //    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(s.top_k));
        //    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));
        //
        //    std::string output;
        //    for (int i = 0; i < s.max_tokens; i++) {
        //        llama_token id = llama_sampler_sample(sampler, ctx_, -1);
        //        if (llama_token_is_eog(model_, id)) break;
        //        output += llama_token_to_piece(ctx_, id);
        //        llama_batch_clear(batch);
        //        llama_batch_add(batch, id, tokens.size() + i, {0}, true);
        //        llama_decode(ctx_, batch);
        //        r.tokens++;
        //    }
        //    r.text = output;

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
        // llama_free(ctx_); ctx_ = nullptr;
        // llama_model_free(model_); model_ = nullptr;
        loaded = false;
        std::cout << "[" << name() << "] unloaded: " << alias << std::endl;
    }

    bool adaptKv(const KvConfig& kv) override {
        std::cout << "[" << name() << "] adapt KV " << kv_.type_v << " -> " << kv.type_v << std::endl;
        kv_ = kv;
        // In production: reallocate KV cache with new quantization
        // llama_kv_cache_clear(ctx_);
        // Reconfigure cache type without reloading weights
        return true;
    }
};

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
