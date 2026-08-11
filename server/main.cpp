// EIE — Main Entry Point
// Apache License 2.0

#include "../core/config.h"
#include "../core/scheduling.h"
#include "../core/model_manager.h"
#include "../core/vram_manager.h"
#include "../monitoring/monitoring.h"
#include "api.h"
#include <iostream>
#include <csignal>

static volatile bool running = true;
void sighandler(int) { running = false; }

int main(int argc, char** argv) {
    std::cout << R"(
╔══════════════════════════════════════╗
║  EIE — Elyne Inference Engine v1.0   ║
║  Apache License 2.0                  ║
╚══════════════════════════════════════╝
)" << std::endl;

    // Parse args
    std::string config_path = "presets/generic.yaml";
    std::vector<std::string> cli_models;
    std::string cli_models_dir, cli_host;
    int cli_port = 0, cli_ctx = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc)
            config_path = argv[++i];
        else if ((arg == "--model" || arg == "-m") && i + 1 < argc)
            cli_models.push_back(argv[++i]);
        else if (arg == "--models-dir" && i + 1 < argc)
            cli_models_dir = argv[++i];
        else if (arg == "--port" && i + 1 < argc)
            cli_port = std::stoi(argv[++i]);
        else if (arg == "--host" && i + 1 < argc)
            cli_host = argv[++i];
        else if (arg == "--ctx" && i + 1 < argc)
            cli_ctx = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: eie-server [--config <path>] [-m <model.gguf>]...\n"
                      << "                  [--models-dir <dir>] [--host <h>] [--port <p>] [--ctx <n>]\n";
            return 0;
        }
    }

    // Load config (CLI overrides)
    auto cfg = eie::loadConfig(config_path);
    if (!cli_models_dir.empty()) { cfg.model_dir = cli_models_dir; cfg.auto_discover = true; }
    if (!cli_host.empty()) cfg.host = cli_host;
    if (cli_port > 0) cfg.port = cli_port;
    if (cli_ctx > 0) cfg.default_kv.n_ctx = cli_ctx;

    // Initialize policy engine
    auto strategy = eie::createStrategy(cfg.strategy);
    strategy->groups = cfg.groups;
    std::cout << "[EIE] Strategy: " << strategy->name()
              << " (" << cfg.groups.size() << " groups)" << std::endl;

    // Initialize managers
    eie::ModelManager models;
    eie::VramManager vram(cfg.vram);
    eie::GroupScheduler scheduler(strategy.get());
    eie::Metrics metrics;
    eie::AuditTrail audit(cfg.audit_enabled, cfg.audit_path);

    // Discover models
    if (cfg.auto_discover) {
        models.discover(cfg.model_dir);
    }
    for (auto& [alias, path] : cfg.models) {
        models.reg(alias, path);
    }
    for (auto& path : cli_models) {
        auto stem = std::filesystem::path(path).stem().string();
        models.reg(stem, path);
        cfg.preload.push_back(stem);
    }

    // Pre-load pinned models from groups
    for (auto& [gname, group] : cfg.groups) {
        for (auto& alias : group.models) {
            eie::KvConfig kv = group.kv_override.type_k.empty()
                ? cfg.default_kv : group.kv_override;
            if (models.load(alias, kv)) {
                std::cout << "[EIE] Pre-loaded: " << alias
                          << " (group: " << gname << ")" << std::endl;
            } else {
                std::cerr << "[EIE] WARN: Failed to load: " << alias << std::endl;
            }
        }
    }

    // Preload models requested via config `preload:` or CLI -m
    std::vector<std::string> to_preload;
    for (auto& alias : cfg.preload) {
        if (alias == "all") {
            auto avail = models.available();
            to_preload.insert(to_preload.end(), avail.begin(), avail.end());
        } else {
            to_preload.push_back(alias);
        }
    }
    for (auto& alias : to_preload) {
        if (models.get(alias)) continue; // already loaded
        if (models.load(alias, cfg.default_kv)) {
            std::cout << "[EIE] Pre-loaded: " << alias << std::endl;
        } else {
            std::cerr << "[EIE] WARN: Failed to load: " << alias << std::endl;
        }
    }

    // VRAM report
    vram.report();

    // Start server
    std::cout << "\n[EIE] Starting server on " << cfg.host << ":"
              << cfg.port << std::endl;
    signal(SIGINT, sighandler);
    signal(SIGTERM, sighandler);

    eie::startServer(cfg, models, scheduler, strategy.get(), metrics, audit);

    std::cout << "\n[EIE] Shutdown." << std::endl;
    return 0;
}
