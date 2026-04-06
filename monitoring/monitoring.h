// EIE — Monitoring: Audit Trail + Health + Metrics
// Apache License 2.0
#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <ctime>
#include <map>
#include <cstdint>

namespace eie {

// ═══════════════════════════════════════════
// Audit Trail (hash-chain)
// ═══════════════════════════════════════════

class AuditTrail {
    bool enabled_;
    std::string path_;
    std::ofstream file_;
    std::string last_hash_ = std::string(64, '0');
    size_t count_ = 0;

    std::string hash(const std::string& s) {
        // FNV-1a based hash for MVP — replace with SHA-256 in production
        uint64_t h = 0xcbf29ce484222325ULL;
        for (char c : s) { h ^= (uint8_t)c; h *= 0x100000001b3ULL; }
        char buf[65];
        snprintf(buf, sizeof(buf), "%016llx%016llx%016llx%016llx",
            (unsigned long long)h, (unsigned long long)(h ^ 0xdeadbeefcafeULL),
            (unsigned long long)(h * 31), (unsigned long long)(h * 37));
        return std::string(buf);
    }

public:
    AuditTrail(bool enabled = false, const std::string& path = "")
        : enabled_(enabled), path_(path) {
        if (enabled_ && !path_.empty()) file_.open(path_, std::ios::app);
    }

    void record(const std::string& group, const std::string& decision,
                const std::string& prompt_hash) {
        if (!enabled_) return;
        std::ostringstream ss;
        ss << std::time(nullptr) << "|" << group << "|" << decision
           << "|" << prompt_hash << "|" << last_hash_;
        std::string h = hash(ss.str());
        last_hash_ = h;
        count_++;
        if (file_.is_open()) {
            file_ << h << " " << group << " " << decision << "\n";
            file_.flush();
        }
    }

    size_t size() const { return count_; }
};

// ═══════════════════════════════════════════
// Metrics (Prometheus-compatible)
// ═══════════════════════════════════════════

class Metrics {
    struct ModelM { int64_t reqs = 0; float lat_max = 0; int64_t tokens = 0; };
    struct GroupM { int64_t execs = 0, partials = 0; float lat_max = 0; };
    std::map<std::string, ModelM> models_;
    std::map<std::string, GroupM> groups_;
    int64_t start_time_ = std::time(nullptr);

public:
    void recordModel(const std::string& m, float lat, int tokens = 0) {
        auto& x = models_[m];
        x.reqs++;
        x.lat_max = std::max(x.lat_max, lat);
        x.tokens += tokens;
    }

    void recordGroup(const std::string& g, bool complete, float lat) {
        auto& x = groups_[g];
        x.execs++;
        if (!complete) x.partials++;
        x.lat_max = std::max(x.lat_max, lat);
    }

    std::string prometheus() {
        std::ostringstream ss;
        for (auto& [n, m] : models_) {
            ss << "eie_model_requests_total{model=\"" << n << "\"} " << m.reqs << "\n";
            ss << "eie_model_latency_max_ms{model=\"" << n << "\"} " << m.lat_max << "\n";
            ss << "eie_model_tokens_total{model=\"" << n << "\"} " << m.tokens << "\n";
        }
        for (auto& [n, g] : groups_) {
            ss << "eie_group_executions_total{group=\"" << n << "\"} " << g.execs << "\n";
            ss << "eie_group_partial_count{group=\"" << n << "\"} " << g.partials << "\n";
            ss << "eie_group_latency_max_ms{group=\"" << n << "\"} " << g.lat_max << "\n";
        }
        ss << "eie_uptime_seconds " << (std::time(nullptr) - start_time_) << "\n";
        ss << "eie_models_loaded " << models_.size() << "\n";
        return ss.str();
    }

    std::string healthJson() {
        return "{\"status\":\"ok\",\"uptime\":" +
               std::to_string(std::time(nullptr) - start_time_) +
               ",\"models\":" + std::to_string(models_.size()) + "}";
    }
};

} // namespace eie
