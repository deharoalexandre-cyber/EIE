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
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc)
            config_path = argv[++i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: eie-server [--config <path>]\n";
            return 0;
        }
    }

    // Load config
    auto cfg = eie::loadConfig(config_path);

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
