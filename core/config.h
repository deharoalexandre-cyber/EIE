// EIE — Configuration
// Apache License 2.0
#pragma once
#include "scheduling.h"
#include "vram_manager.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace eie {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string auth_token;
    std::string strategy = "generic";
    std::string model_dir = "/models";
    bool auto_discover = true;
    KvConfig default_kv;
    VramConfig vram;
    std::map<std::string, GroupConfig> groups;
    std::map<std::string, std::string> models; // alias -> path
    bool audit_enabled = false;
    std::string audit_path = "/var/log/eie/audit.chain";
    std::string log_level = "info";
};

// Simple key:value parser — replace with yaml-cpp for production
inline ServerConfig loadConfig(const std::string& path) {
    ServerConfig cfg;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[Config] cannot open: " << path << ", using defaults" << std::endl;
        return cfg;
    }
    std::string line;
    while (std::getline(f, line)) {
        line.erase(0, line.find_first_not_of(" \t"));
        if (line.empty() || line[0] == '#') continue;
        auto c = line.find('#');
        if (c != std::string::npos) line = line.substr(0, c);
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        std::string val = line.substr(colon + 1);
        key.erase(key.find_last_not_of(" \t") + 1);
        val.erase(0, val.find_first_not_of(" \t"));
        val.erase(val.find_last_not_of(" \t\r\n") + 1);

        if (key == "host") cfg.host = val;
        else if (key == "port") cfg.port = std::stoi(val);
        else if (key == "auth_token") cfg.auth_token = val;
        else if (key == "strategy") cfg.strategy = val;
        else if (key == "model_dir") cfg.model_dir = val;
        else if (key == "auto_discover") cfg.auto_discover = (val == "true");
        else if (key == "type_k") cfg.default_kv.type_k = val;
        else if (key == "type_v") cfg.default_kv.type_v = val;
        else if (key == "n_ctx") cfg.default_kv.n_ctx = std::stoi(val);
        else if (key == "flash_attn") cfg.default_kv.flash_attn = (val == "true");
        else if (key == "audit_enabled") cfg.audit_enabled = (val == "true");
        else if (key == "audit_path") cfg.audit_path = val;
        else if (key == "reserve_mb") cfg.vram.reserve_mb = std::stoul(val);
        else if (key == "log_level") cfg.log_level = val;
    }
    std::cout << "[Config] loaded: strategy=" << cfg.strategy
              << " port=" << cfg.port << std::endl;
    return cfg;
}

} // namespace eie
