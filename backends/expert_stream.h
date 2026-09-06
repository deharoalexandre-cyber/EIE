// EIE - on-demand Gemma4 expert weights (Apache-2.0)
#pragma once
#include "llama.h"
#include <memory>
#include <string>
#include <cstdint>

namespace eie {
struct ExpertStreamStats {
    uint64_t callbacks = 0, hits = 0, misses = 0, payload_bytes = 0, read_bytes = 0;
    uint64_t logical_expert_bytes = 0, physical_expert_bytes = 0;
};

// One cache per loaded model, shared by its serialized inference contexts.
class ExpertStream {
public:
    ExpertStream(const std::string & path, int slots);
    ~ExpertStream();
    void bind(llama_model * model);
    void configure(llama_context_params & params);
    const std::string & error() const;
    ExpertStreamStats stats() const;
    static bool callback(ggml_tensor * tensor, bool ask, void * user);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
