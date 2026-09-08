// EIE - consumed expert streaming, ported from experiments/ews (Apache-2.0)
#include "expert_stream.h"
#include "llama-model.h"
#include "gguf.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace eie {
namespace {
void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct SlabFile {
    uint64_t bytes_read = 0;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    void * scratch = nullptr;
    size_t capacity = 0;
    explicit SlabFile(const std::string & path) {
        file = CreateFileW(std::filesystem::u8path(path).c_str(), GENERIC_READ,
            FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
        check(file != INVALID_HANDLE_VALUE, "EWS: cannot open GGUF for direct reads");
    }
    ~SlabFile() { if (file != INVALID_HANDLE_VALUE) CloseHandle(file); _aligned_free(scratch); }
    const uint8_t * read(uint64_t offset, size_t bytes) {
        const uint64_t first = offset & ~uint64_t(4095);
        const size_t needed = size_t(((offset + bytes + 4095) & ~uint64_t(4095)) - first);
        if (needed > capacity) {
            _aligned_free(scratch);
            scratch = _aligned_malloc(needed, 4096);
            check(scratch != nullptr, "EWS: read buffer allocation failed");
            capacity = needed;
        }
        LARGE_INTEGER where; where.QuadPart = first;
        check(SetFilePointerEx(file, where, nullptr, FILE_BEGIN), "EWS: GGUF seek failed");
        DWORD got = 0;
        check(needed <= MAXDWORD && ReadFile(file, scratch, DWORD(needed), &got, nullptr)
            && got >= bytes + offset - first, "EWS: GGUF slab read failed");
        bytes_read += got;
        return static_cast<uint8_t *>(scratch) + offset - first;
    }
#else
    // Portable correctness path; unlike Windows direct I/O, this uses the OS page cache.
    std::ifstream file;
    std::vector<uint8_t> scratch;
    explicit SlabFile(const std::string & path) : file(path, std::ios::binary) {
        check(bool(file), "EWS: cannot open GGUF");
    }
    const uint8_t * read(uint64_t offset, size_t bytes) {
        scratch.resize(bytes);
        file.clear(); file.seekg(offset); file.read(reinterpret_cast<char *>(scratch.data()), bytes);
        check(size_t(file.gcount()) == bytes, "EWS: GGUF slab read failed");
        bytes_read += bytes;
        return scratch.data();
    }
#endif
};

struct Weight {
    uint64_t offset = 0;
    size_t slab = 0, file = 0;
    ggml_type type = GGML_TYPE_F32;
    ggml_tensor * tensor = nullptr;
};
struct Layer {
    std::map<std::string, Weight> weights;
    std::vector<int> experts;
    std::vector<uint64_t> touched;
    uint64_t clock = 0;
    std::pair<int, bool> acquire(int id, const std::set<int> & active) {
        for (int s = 0; s < int(experts.size()); ++s) if (experts[s] == id) {
            touched[s] = ++clock; return {s, true};
        }
        int victim = -1;
        for (int s = 0; s < int(experts.size()); ++s) {
            if (experts[s] < 0) { victim = s; break; }
            if (!active.count(experts[s]) && (victim < 0 || touched[s] < touched[victim])) victim = s;
        }
        check(victim >= 0, "EWS: more simultaneous experts than physical slots");
        experts[victim] = id; touched[victim] = ++clock;
        return {victim, false};
    }
};

int integer(const gguf_context * meta, const std::string & key, int fallback = -1) {
    const auto i = gguf_find_key(meta, key.c_str());
    if (i < 0) return fallback;
    switch (gguf_get_kv_type(meta, i)) {
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(meta, i);
        case GGUF_TYPE_UINT32: return int(gguf_get_val_u32(meta, i));
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(meta, i);
        default: throw std::runtime_error("EWS: expected integer metadata: " + key);
    }
}

bool expert_weight(const char * kind) {
    return strcmp(kind, "ffn_gate_up_exps.weight") == 0 || strcmp(kind, "ffn_gate_exps.weight") == 0 ||
           strcmp(kind, "ffn_up_exps.weight") == 0 || strcmp(kind, "ffn_down_exps.weight") == 0;
}
}

struct ExpertStream::Impl {
    int slots;
    int expert_count = 0, top_k = 0;
    std::map<int, Layer> layers;
    std::vector<std::unique_ptr<SlabFile>> files;
    std::string error;
    ExpertStreamStats stats;
    Impl(const std::string & path, int n) : slots(n) {
        int count = 1, first_layer = 0, last_layer = 0;
        std::string arch;
        std::vector<char> split_prefix(path.size() + 32), split_path(path.size() + 32);
        for (int part = 0; part < count; ++part) {
            std::string current = path;
            if (part) {
                check(llama_split_path(split_path.data(), split_path.size(), split_prefix.data(), part, count) > 0,
                      "EWS: cannot form shard path");
                current = split_path.data();
            }
            ggml_context * raw_tensors = nullptr;
            std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(
                gguf_init_from_file(current.c_str(), {true, &raw_tensors}), gguf_free);
            std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(raw_tensors, ggml_free);
            check(bool(meta), "EWS: cannot read GGUF shard metadata");
            if (!part) {
                const auto key = gguf_find_key(meta.get(), "general.architecture");
                check(key >= 0, "EWS: missing architecture");
                arch = gguf_get_val_str(meta.get(), key);
                check(arch == "gemma4" || arch == "glm5next", "EWS: unsupported architecture");
                expert_count = integer(meta.get(), arch + ".expert_count");
                top_k = integer(meta.get(), arch + ".expert_used_count");
                first_layer = integer(meta.get(), arch + ".leading_dense_block_count", 0);
                last_layer = integer(meta.get(), arch + ".block_count") - integer(meta.get(), arch + ".nextn_predict_layers", 0);
                check(top_k > 0 && slots >= top_k && slots < expert_count, "EWS: need top-k <= slots < expert count");
                count = integer(meta.get(), "split.count", 1);
                check(count >= 1 && integer(meta.get(), "split.no", 0) == 0, "EWS: supply the first GGUF shard");
                if (count > 1) check(llama_split_prefix(split_prefix.data(), split_prefix.size(), path.c_str(), 0, count) > 0,
                                     "EWS: invalid split GGUF filename");
            }
            check(integer(meta.get(), "split.no", 0) == part && integer(meta.get(), "split.count", 1) == count,
                  "EWS: inconsistent shard index");
            files.push_back(std::make_unique<SlabFile>(current));
            for (auto * t = tensors ? ggml_get_first_tensor(tensors.get()) : nullptr; t; t = ggml_get_next_tensor(tensors.get(), t)) {
                int id = -1; char kind[80] = {};
                if (sscanf(t->name, "blk.%d.%79s", &id, kind) != 2) continue;
                if (!expert_weight(kind) || id < first_layer || id >= last_layer) continue;
                check(t->ne[2] == expert_count && t->ne[3] == 1, "EWS: expert count mismatch");
                auto & layer = layers[id];
                layer.experts.assign(slots, -1); layer.touched.assign(slots, 0);
                check(!layer.weights.count(kind), "EWS: duplicate expert tensor");
                auto & w = layer.weights[kind];
                w.slab = t->nb[2];
                w.file = size_t(part); w.type = t->type;
                w.offset = gguf_get_data_offset(meta.get()) + gguf_get_tensor_offset(meta.get(), gguf_find_tensor(meta.get(), t->name));
                stats.logical_expert_bytes += ggml_nbytes(t);
            }
        }
        check(last_layer > first_layer && layers.size() == size_t(last_layer - first_layer), "EWS: missing routed layer");
        for (const auto & p : layers) {
            const auto & w = p.second.weights;
            const bool fused = w.size() == 2 && w.count("ffn_gate_up_exps.weight");
            const bool separate = w.size() == 3 && w.count("ffn_gate_exps.weight") && w.count("ffn_up_exps.weight");
            check(w.count("ffn_down_exps.weight") && (fused || separate), "EWS: missing expert projection");
        }
    }
    void upload(Weight & w, int expert, int slot) {
        auto & file = *files.at(w.file);
        const uint64_t before = file.bytes_read;
        const auto * data = file.read(w.offset + expert * w.slab, w.slab);
        ggml_backend_tensor_set(w.tensor, data, size_t(slot) * w.slab, w.slab);
        stats.payload_bytes += w.slab;
        if (ggml_backend_buffer_is_host(w.tensor->buffer)) stats.host_payload_bytes += w.slab;
        else stats.device_payload_bytes += w.slab;
        stats.read_bytes += file.bytes_read - before;
    }
};

ExpertStream::ExpertStream(const std::string & path, int slots) : impl_(new Impl(path, slots)) {}
ExpertStream::~ExpertStream() = default;
const std::string & ExpertStream::error() const { return impl_->error; }
ExpertStreamStats ExpertStream::stats() const { return impl_->stats; }

void ExpertStream::bind(llama_model * model) {
    auto & s = *impl_;
    for (const auto & p : model->tensors_by_name) {
        int id = -1; char kind[80] = {};
        if (sscanf(p.first.c_str(), "blk.%d.%79s", &id, kind) != 2) continue;
        if (!expert_weight(kind)) continue;
        auto & w = s.layers.at(id).weights.at(kind);
        w.tensor = p.second;
        check(w.tensor->ne[2] == s.slots && w.tensor->nb[2] == w.slab && w.tensor->type == w.type,
              "EWS: runtime slot layout mismatch");
        s.stats.physical_expert_bytes += ggml_nbytes(w.tensor);
    }
    for (auto & p : s.layers) for (auto & w : p.second.weights) check(w.second.tensor != nullptr, "EWS: unbound weights");
}

void ExpertStream::configure(llama_context_params & p) {
    p.n_batch = p.n_ubatch = 1;
    p.cb_eval = callback; p.cb_eval_user_data = this;
}

bool ExpertStream::callback(ggml_tensor * t, bool ask, void * user) {
    const bool boundary = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    const char * prefix = "ffn_moe_ews_slots-";
    const bool remap = strncmp(t->name, prefix, strlen(prefix)) == 0;
    if (ask) return boundary || remap;
    if (!remap) return true;
    auto & s = *static_cast<ExpertStream *>(user)->impl_;
    if (!s.error.empty()) return false;
    try {
        check(t->type == GGML_TYPE_I32 && t->ne[0] == s.top_k && t->ne[1] == 1, "EWS: expected one-token top-k routing");
        std::vector<int32_t> ids(size_t(s.top_k));
        ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));
        const std::set<int> active(ids.begin(), ids.end());
        check(active.size() == ids.size(), "EWS: duplicate route IDs");
        auto & layer = s.layers.at(std::stoi(t->name + strlen(prefix)));
        ++s.stats.callbacks;
        for (auto & id : ids) {
            check(id >= 0 && id < s.expert_count, "EWS: expert ID out of range");
            auto slot = layer.acquire(id, active);
            if (slot.second) ++s.stats.hits;
            else {
                ++s.stats.misses;
                for (auto & w : layer.weights) s.upload(w.second, id, slot.first);
            }
            id = slot.first;
        }
        ggml_backend_tensor_set(t, ids.data(), 0, ids.size() * sizeof(int32_t));
        return true;
    } catch (const std::exception & e) { s.error = e.what(); return false; }
}
}
