// Test the production reader and LRU with small split GGUFs; no model or GPU.
#include "backends/expert_stream.cpp"
#include "core/config.h"
#include "ggml-cpu.h"
#include <chrono>
#include <iostream>

static int checks = 0;
static void expect(bool ok, const char * message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}

static std::string fixture(const std::filesystem::path & dir, bool glm, bool missing_up = false) {
    const std::string arch = glm ? "glm5next" : "gemma4";
    const int experts = glm ? 288 : 128, parts = glm ? 3 : 1;
    const auto prefix = (dir / (arch + (missing_up ? "-missing" : ""))).string();
    std::string first;
    for (int part = 0; part < parts; ++part) {
        char name[4096];
        if (glm) llama_split_path(name, sizeof(name), prefix.c_str(), part, parts);
        else snprintf(name, sizeof(name), "%s.gguf", prefix.c_str());
        if (!part) first = name;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_empty(), gguf_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(ggml_init({1 << 20, nullptr, false}), ggml_free);
        gguf_set_val_u16(meta.get(), "split.count", parts);
        gguf_set_val_u16(meta.get(), "split.no", part);
        if (!part) {
            gguf_set_val_str(meta.get(), "general.architecture", arch.c_str());
            gguf_set_val_u32(meta.get(), (arch + ".block_count").c_str(), glm ? 3 : 1);
            gguf_set_val_u32(meta.get(), (arch + ".leading_dense_block_count").c_str(), glm ? 1 : 0);
            gguf_set_val_u32(meta.get(), (arch + ".nextn_predict_layers").c_str(), glm ? 1 : 0);
            gguf_set_val_u32(meta.get(), (arch + ".expert_count").c_str(), experts);
            gguf_set_val_u32(meta.get(), (arch + ".expert_used_count").c_str(), 8);
        }
        std::vector<std::string> kinds;
        if (!glm) kinds = {"gate_up", "down"};
        else if (part == 1) kinds = missing_up ? std::vector<std::string>{"gate"} : std::vector<std::string>{"gate", "up"};
        else if (part == 2) kinds = {"down"};
        for (const auto & kind : kinds) {
            auto * t = ggml_new_tensor_3d(tensors.get(), GGML_TYPE_F32, 8, kind == "gate_up" ? 8 : 4, experts);
            const auto tensor_name = "blk." + std::to_string(glm ? 1 : 0) + ".ffn_" + kind + "_exps.weight";
            ggml_set_name(t, tensor_name.c_str());
            for (size_t i = 0; i < ggml_nbytes(t) / sizeof(float); ++i) static_cast<float *>(t->data)[i] = float(i);
            gguf_add_tensor(meta.get(), t);
        }
        expect(gguf_write_to_file(meta.get(), name, false), "fixture write");
    }
    return first;
}

// Real production callback + GGUF reads + physical tensor writes, with tiny
// synthetic expert tensors. This is NOT a trained GLM inference test.
static std::vector<uint8_t> callback_gate(const std::string& path, bool trace) {
    eie::ExpertStream stream(path, 8);
    struct TensorModel : llama_model_base {
        TensorModel() : llama_model_base(llama_model_default_params()) {}
        void load_arch_hparams(llama_model_loader&) override {}
        void load_arch_tensors(llama_model_loader&) override {}
        std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params&) const override { return {}; }
    } model;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1 << 20, nullptr, true}), ggml_free);
    for (const auto* kind : {"gate", "up", "down"}) {
        auto* t = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 4, 8);
        std::string name = std::string("blk.1.ffn_") + kind + "_exps.weight";
        ggml_set_name(t, name.c_str());
        model.tensors_by_name.push_back({name, t});
    }
    auto* ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 8, 1);
    ggml_set_name(ids, "ffn_moe_ews_slots-1");
    auto* backend = ggml_backend_cpu_init();
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
    expect(buffer != nullptr, "CPU fixture buffer");
    stream.bind(&model);
    stream.beginTrace(trace);
    std::vector<uint8_t> result;
    for (int round = 0; round < 3; ++round) {
        if (round) stream.tracePhase(eie::RoutingPhase::Decode);
        std::vector<int32_t> logical{280, 281, 282, 283, 284, 285, 286, round == 2 ? 7 : 287};
        ggml_backend_tensor_set(ids, logical.data(), 0, 32);
        expect(eie::ExpertStream::callback(ids, true, &stream), "ask sees remap boundary");
        expect(eie::ExpertStream::callback(ids, false, &stream), "production callback failed");
        std::vector<int32_t> physical(8);
        ggml_backend_tensor_get(ids, physical.data(), 0, 32);
        for (int index = 0; index < 8; ++index) {
            expect(physical[index] >= 0 && physical[index] < 8, "physical slot out of range");
            for (auto& weight : model.tensors_by_name) {
                float values[32];
                ggml_backend_tensor_get(weight.second, values, physical[index] * sizeof(values), sizeof(values));
                for (int i = 0; i < 32; ++i)
                    expect(values[i] == float(logical[index] * 32 + i), "logical expert payload changed");
                auto* bytes = reinterpret_cast<uint8_t*>(values);
                result.insert(result.end(), bytes, bytes + sizeof(values));
            }
        }
    }
    stream.endTrace("complete");
    auto stats = stream.stats();
    expect(stats.callbacks == 3 && stats.hits == 15 && stats.misses == 9, "actual callback counters");
    const auto routing = stream.routing();
    if (trace) {
        const auto& phases = routing.layers.at(1).phases;
        expect(phases[0].callbacks == 1 && phases[1].callbacks == 2, "prefill/decode split");
        expect(phases[0].experts[287].misses == 1 && phases[1].experts[287].hits == 1, "high logical ID lost");
        expect(phases[1].experts[7].misses == 1, "eviction missing from histogram");
        expect(routing.layers.at(1).expert_weight_bytes == 384, "per-expert bytes");
        uint64_t hits = 0, misses = 0;
        for (const auto& phase : phases) for (const auto& access : phase.experts) {
            hits += access.hits; misses += access.misses;
        }
        expect(hits == stats.hits && misses == stats.misses, "histogram/callback conservation");
    } else expect(routing.layers.empty(), "disabled trace allocated counters");
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return result;
}

int main(int argc, char ** argv) {
    try {
        expect(argc == 2, "supply a new scratch directory");
        const auto dir = std::filesystem::absolute(argv[1]);
        expect(std::filesystem::create_directory(dir), "scratch directory already exists");
        const auto config_path = dir / "placement.yaml";
        {
            std::ofstream config(config_path);
            config << "models:\n  glm: first-shard.gguf\n  resident: resident.gguf\n"
                      "ews_slots:\n  glm: 8\ngpu_layers:\n  glm: 20\n  resident: 0\n"
                      "ews_trace:\n  glm: true\n  resident: false\n"
                      "cpu_moe:\n  glm: true\n  resident: false\nthreads:\n  glm: 8\n"
                      "port: 18280\n";
        }
        const auto config = eie::loadConfig(config_path.string());
        expect(config.models.at("glm") == "first-shard.gguf", "split model path lost");
        expect(config.ews_slots.at("glm") == 8, "slot config lost");
        expect(config.ews_trace.at("glm") && !config.ews_trace.at("resident"), "trace preset map lost");
        expect(config.gpu_layers.at("glm") == 20 && config.gpu_layers.at("resident") == 0, "per-model placement lost");
        expect(config.cpu_moe.at("glm") && !config.cpu_moe.at("resident"), "per-model CPU expert toggle lost");
        expect(config.threads.at("glm") == 8 && !config.threads.count("resident"), "per-model threads/default changed");
        expect(config.port == 18280, "placement section consumed subsequent global setting");
        const eie::ModelParams defaults;
        expect(defaults.n_gpu_layers == 99 && defaults.n_threads == 0 && !defaults.cpu_moe && !defaults.ews_slots && !defaults.ews_trace,
               "normal model defaults changed");
        const auto gemma = fixture(dir, false), glm = fixture(dir, true), missing = fixture(dir, true, true);
        eie::ExpertStream gemma_stream(gemma, 8), glm_stream(glm, 8);
        expect(gemma_stream.stats().logical_expert_bytes == 128 * (8 * 8 + 8 * 4) * 4, "fused Gemma inventory");
        expect(glm_stream.stats().logical_expert_bytes == 288 * 3 * 8 * 4 * 4, "split GLM inventory");
        const auto untraced = callback_gate(glm, false);
        const auto traced = callback_gate(glm, true);
        expect(untraced == traced, "tracing changed consumed expert tensor bytes");
        bool rejected = false;
        try { eie::ExpertStream incomplete(missing, 8); } catch (const std::runtime_error &) { rejected = true; }
        expect(rejected, "missing projection must be reported");
        eie::Layer layer;
        layer.experts.assign(8, -1); layer.touched.assign(8, 0);
        const std::set<int> initial{0, 1, 2, 3, 4, 5, 286, 287};
        for (int id : initial) expect(!layer.acquire(id, initial).second, "initial miss");
        for (int id : initial) expect(layer.acquire(id, initial).second, "high-ID cache hit");
        const std::set<int> next{7, 1, 2, 3, 4, 5, 286, 287};
        expect(!layer.acquire(7, next).second, "eviction miss");
        for (int id : next) expect(layer.acquire(id, next).second, "active expert evicted");

        // Read the final bytes of a real shard through the same aligned I/O path.
        const auto file_path = dir / "glm5next-00003-of-00003.gguf";
        const auto length = std::filesystem::file_size(file_path);
        std::ifstream expected_file(file_path, std::ios::binary);
        std::vector<char> expected(137);
        expected_file.seekg(length - expected.size()); expected_file.read(expected.data(), expected.size());
        eie::SlabFile file(file_path.string());
        const auto * actual = file.read(length - expected.size(), expected.size());
        expect(memcmp(actual, expected.data(), expected.size()) == 0, "unaligned EOF slab differs");
        expect(file.bytes_read >= expected.size(), "aligned I/O accounting");
        std::cout << "PASS " << checks << " EWS assertions; fixtures retained at " << dir << '\n';
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
