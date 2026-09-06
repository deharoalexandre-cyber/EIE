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

struct Weight { uint64_t offset = 0; size_t slab = 0; ggml_tensor * tensor = nullptr; };
struct Layer {
    Weight gate_up, down;
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
}

struct ExpertStream::Impl {
    int slots;
    std::map<int, Layer> layers;
    SlabFile file;
    std::string error;
    ExpertStreamStats stats;
    Impl(const std::string & path, int n) : slots(n), file(path) {
        check(slots >= 8 && slots < 128, "EWS: Gemma4 requires 8..127 slots (0 disables EWS)");
        ggml_context * tensors = nullptr;
        auto * meta = gguf_init_from_file(path.c_str(), {true, &tensors});
        check(meta && tensors, "EWS: cannot read GGUF metadata");
        try {
            for (auto * t = ggml_get_first_tensor(tensors); t; t = ggml_get_next_tensor(tensors, t)) {
                int id = -1; char kind[80] = {};
                if (sscanf(t->name, "blk.%d.%79s", &id, kind) != 2) continue;
                bool gu = strcmp(kind, "ffn_gate_up_exps.weight") == 0;
                bool dn = strcmp(kind, "ffn_down_exps.weight") == 0;
                if (!gu && !dn) continue;
                check(t->ne[2] == 128 && t->ne[3] == 1, "EWS: expected Gemma4 128-expert weights");
                auto & layer = layers[id];
                layer.experts.assign(slots, -1); layer.touched.assign(slots, 0);
                auto & w = gu ? layer.gate_up : layer.down;
                w.slab = t->nb[2];
                w.offset = gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, gguf_find_tensor(meta, t->name));
                stats.logical_expert_bytes += ggml_nbytes(t);
            }
            check(layers.size() == 30, "EWS: this port supports Gemma4 26B-A4B's 30 MoE layers");
            for (const auto & p : layers) check(p.second.gate_up.slab && p.second.down.slab, "EWS: missing expert tensor");
        } catch (...) { gguf_free(meta); ggml_free(tensors); throw; }
        gguf_free(meta); ggml_free(tensors);
    }
    void upload(Weight & w, int expert, int slot) {
        const auto * data = file.read(w.offset + expert * w.slab, w.slab);
        ggml_backend_tensor_set(w.tensor, data, size_t(slot) * w.slab, w.slab);
        stats.payload_bytes += w.slab;
        stats.read_bytes = file.bytes_read;
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
        bool gu = strcmp(kind, "ffn_gate_up_exps.weight") == 0;
        bool dn = strcmp(kind, "ffn_down_exps.weight") == 0;
        if (!gu && !dn) continue;
        auto & w = gu ? s.layers.at(id).gate_up : s.layers.at(id).down;
        w.tensor = p.second;
        check(w.tensor->ne[2] == s.slots && w.tensor->nb[2] == w.slab, "EWS: runtime slot layout mismatch");
        s.stats.physical_expert_bytes += ggml_nbytes(w.tensor);
    }
    for (auto & p : s.layers) check(p.second.gate_up.tensor && p.second.down.tensor, "EWS: unbound weights");
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
        check(t->type == GGML_TYPE_I32 && t->ne[0] == 8 && t->ne[1] == 1, "EWS: expected top-8 one-token routing");
        std::vector<int32_t> ids(8);
        ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));
        const std::set<int> active(ids.begin(), ids.end());
        check(active.size() == ids.size(), "EWS: duplicate route IDs");
        auto & layer = s.layers.at(std::stoi(t->name + strlen(prefix)));
        ++s.stats.callbacks;
        for (auto & id : ids) {
            check(id >= 0 && id < 128, "EWS: expert ID out of range");
            auto slot = layer.acquire(id, active);
            if (slot.second) ++s.stats.hits;
            else { ++s.stats.misses; s.upload(layer.gate_up, id, slot.first); s.upload(layer.down, id, slot.first); }
            id = slot.first;
        }
        ggml_backend_tensor_set(t, ids.data(), 0, ids.size() * sizeof(int32_t));
        return true;
    } catch (const std::exception & e) { s.error = e.what(); return false; }
}
}
