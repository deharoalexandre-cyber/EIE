// EIE — Compute Backend Abstraction
// Apache License 2.0
#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <map>

namespace eie {

enum class BackendType { CUDA, HIP, CPU };

struct VramStatus {
    int gpu_id = 0;
    size_t total_bytes = 0, used_bytes = 0, free_bytes = 0;
    float util() const { return total_bytes > 0 ? (float)used_bytes / total_bytes : 0; }
};

struct HealthStatus {
    bool ok = true;
    float latency_ms = 0;
    std::string msg = "OK";
};

struct KvConfig {
    std::string type_k = "turbo3";
    std::string type_v = "turbo3";
    bool flash_attn = true;
    int n_ctx = 4096;
};

struct ModelParams {
    std::string path, alias;
    int n_gpu_layers = 99, n_threads = 2;
    KvConfig kv;
};

struct SamplingParams {
    float temperature = 0.7f, top_p = 0.9f;
    int top_k = 40, max_tokens = 2048;
    std::vector<std::string> stop; // arrêt de génération sur ces séquences
    // One-shot (consolidations, rêves, tâches annexes) : contexte ÉPHÉMÈRE,
    // le cache KV de la conversation principale est préservé.
    // (Fix « bug n°1 » d'Elyne Mobile, porté serveur : 2e contexte, KV du chat intact.)
    bool one_shot = false;
};

struct ChatMessage {
    std::string role;    // "system" / "user" / "assistant"
    std::string content;
};

struct InferenceResult {
    std::string model, text, error;
    int tokens = 0;
    int reused_tokens = 0; // préfixe KV réutilisé (0 = préfill complet)
    float latency_ms = 0;
    bool ok = true;
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    virtual BackendType type() const = 0;
    virtual std::string name() const = 0;
    virtual bool init(int gpu_id = 0) = 0;
    virtual bool load(const ModelParams& p) = 0;
    virtual InferenceResult chat(const std::string& prompt, const SamplingParams& s) = 0;
    /** Applique le template de chat natif du modèle (métadonnées GGUF).
     *  Renvoie "" si le backend ne sait pas ; l'appelant utilise alors un repli. */
    virtual std::string formatChat(const std::vector<ChatMessage>& msgs) { return ""; }
    /** Embedding L2-normalisé du texte (modèles encodeurs type bge-m3).
     *  Renvoie un vecteur vide si le backend ne sait pas. */
    virtual std::vector<float> embed(const std::string& text) { return {}; }
    virtual VramStatus vram() = 0;
    virtual HealthStatus health() = 0;
    virtual void unload() = 0;
    virtual bool adaptKv(const KvConfig& kv) { return false; }
    bool loaded = false;
    std::string alias;
};

std::unique_ptr<ComputeBackend> detectBackend(int gpu_id = 0);

} // namespace eie
