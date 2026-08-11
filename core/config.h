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
    std::vector<std::string> preload;          // aliases to load at boot ("all" = every discovered model)
    bool audit_enabled = false;
    std::string audit_path = "/var/log/eie/audit.chain";
    std::string log_level = "info";
};

// YAML minimal : clés `key: value` au niveau racine, liste `groups:`
// (blocs `- name: ...`) et carte `models:` (alias: chemin).
// Couvre les presets fournis — pas un parseur YAML complet.
inline ServerConfig loadConfig(const std::string& path) {
    ServerConfig cfg;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[Config] cannot open: " << path << ", using defaults" << std::endl;
        return cfg;
    }

    enum class Section { TOP, GROUPS, MODELS };
    Section section = Section::TOP;
    GroupConfig cur;
    bool has_cur = false;

    auto flush = [&]() {
        if (has_cur && !cur.name.empty()) cfg.groups[cur.name] = cur;
        cur = GroupConfig{};
        has_cur = false;
    };
    auto parseList = [](std::string v) {
        std::vector<std::string> out;
        if (!v.empty() && v.front() == '[') v.erase(0, 1);
        if (!v.empty() && v.back() == ']') v.pop_back();
        std::stringstream ps(v);
        std::string item;
        while (std::getline(ps, item, ',')) {
            item.erase(0, item.find_first_not_of(" \t"));
            item.erase(item.find_last_not_of(" \t") + 1);
            if (!item.empty()) out.push_back(item);
        }
        return out;
    };

    std::string line;
    while (std::getline(f, line)) {
        auto c = line.find('#');
        if (c != std::string::npos) line = line.substr(0, c);
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty()) continue;

        size_t indent = line.find_first_not_of(" \t");
        std::string body = line.substr(indent);

        if (indent == 0 && section != Section::TOP) {
            if (section == Section::GROUPS) flush();
            section = Section::TOP;
        }

        bool item_start = body.rfind("- ", 0) == 0;
        if (item_start) body.erase(0, 2);

        auto colon = body.find(':');
        if (colon == std::string::npos) continue;
        std::string key = body.substr(0, colon);
        std::string val = body.substr(colon + 1);
        key.erase(key.find_last_not_of(" \t") + 1);
        val.erase(0, val.find_first_not_of(" \t"));
        val.erase(val.find_last_not_of(" \t") + 1);

        if (indent == 0) {
            if (key == "groups" && val.empty()) { section = Section::GROUPS; continue; }
            if (key == "models" && val.empty()) { section = Section::MODELS; continue; }

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
            else if (key == "preload") {
                for (auto& item : parseList(val)) cfg.preload.push_back(item);
            }
            continue;
        }

        if (section == Section::GROUPS) {
            if (item_start) { flush(); has_cur = true; }
            if (!has_cur) continue;
            if (key == "name") cur.name = val;
            else if (key == "models") cur.models = parseList(val);
            else if (key == "required_responses" || key == "required") cur.required = std::stoi(val);
            else if (key == "type") cur.type = val;
            else if (key == "pinned") cur.pinned = (val == "true");
            else if (key == "fallback") cur.fallback = val;
            else if (key == "replace_with" || key == "replacement") cur.replacement = val;
            else if (key == "max_latency_ms") cur.max_latency_ms = std::stof(val);
        } else if (section == Section::MODELS) {
            if (!key.empty() && !val.empty()) cfg.models[key] = val;
        }
    }
    if (section == Section::GROUPS) flush();

    std::cout << "[Config] loaded: strategy=" << cfg.strategy
              << " port=" << cfg.port
              << " groups=" << cfg.groups.size() << std::endl;
    return cfg;
}

} // namespace eie
