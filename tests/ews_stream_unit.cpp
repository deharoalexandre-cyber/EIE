// Test the production reader and LRU with small split GGUFs; no model or GPU.
#include "backends/expert_stream.cpp"
#include "core/config.h"
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
                      "cpu_moe:\n  glm: true\n  resident: false\nthreads:\n  glm: 8\n"
                      "port: 18280\n";
        }
        const auto config = eie::loadConfig(config_path.string());
        expect(config.models.at("glm") == "first-shard.gguf", "split model path lost");
        expect(config.ews_slots.at("glm") == 8, "slot config lost");
        expect(config.gpu_layers.at("glm") == 20 && config.gpu_layers.at("resident") == 0, "per-model placement lost");
        expect(config.cpu_moe.at("glm") && !config.cpu_moe.at("resident"), "per-model CPU expert toggle lost");
        expect(config.threads.at("glm") == 8 && !config.threads.count("resident"), "per-model threads/default changed");
        expect(config.port == 18280, "placement section consumed subsequent global setting");
        const eie::ModelParams defaults;
        expect(defaults.n_gpu_layers == 99 && defaults.n_threads == 2 && !defaults.cpu_moe && !defaults.ews_slots,
               "normal model defaults changed");
        const auto gemma = fixture(dir, false), glm = fixture(dir, true), missing = fixture(dir, true, true);
        eie::ExpertStream gemma_stream(gemma, 8), glm_stream(glm, 8);
        expect(gemma_stream.stats().logical_expert_bytes == 128 * (8 * 8 + 8 * 4) * 4, "fused Gemma inventory");
        expect(glm_stream.stats().logical_expert_bytes == 288 * 3 * 8 * 4 * 4, "split GLM inventory");
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
