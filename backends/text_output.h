// Incremental stop matching shared by streamed and buffered generation.
// Only a possible stop prefix / incomplete UTF-8 character is held back.
#pragma once
#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace eie {

class TextOutput {
    std::vector<std::string> stops_;
    std::function<bool(const std::string&)> emit_;
    std::string pending_, text_;
    bool stopped_ = false, cancelled_ = false, finished_ = false;

    // Emit valid UTF-8 even when a model token is just part of a character.
    // Invalid bytes and an incomplete character at finalization become U+FFFD.
    bool publish(size_t limit, bool final) {
        std::string chunk;
        size_t i = 0;
        while (i < limit) {
            const auto c = static_cast<unsigned char>(pending_[i]);
            size_t n = c < 0x80 ? 1 : (c >= 0xC2 && c <= 0xDF ? 2 :
                       (c >= 0xE0 && c <= 0xEF ? 3 : (c >= 0xF0 && c <= 0xF4 ? 4 : 0)));
            bool valid = n != 0;
            for (size_t j = 1; valid && j < n && i + j < limit; ++j) {
                const auto b = static_cast<unsigned char>(pending_[i+j]);
                valid = b >= 0x80 && b <= 0xBF;
                if (j == 1) valid = valid && !(c == 0xE0 && b < 0xA0)
                    && !(c == 0xED && b >= 0xA0) && !(c == 0xF0 && b < 0x90)
                    && !(c == 0xF4 && b >= 0x90);
            }
            if (valid && i + n > limit) {
                if (!final) break;
                chunk += "\xEF\xBF\xBD";
                i = limit;
            } else if (valid) {
                chunk.append(pending_, i, n);
                i += n;
            } else {
                chunk += "\xEF\xBF\xBD";
                ++i;
            }
        }
        pending_.erase(0, i);
        text_ += chunk;
        if (!chunk.empty() && emit_ && !emit_(chunk)) cancelled_ = true;
        return !cancelled_;
    }

public:
    TextOutput(const std::vector<std::string>& stops,
               std::function<bool(const std::string&)> emit = {}) : emit_(std::move(emit)) {
        for (const auto& s : stops) if (!s.empty()) stops_.push_back(s);
    }

    bool push(const std::string& piece) {
        if (stopped_ || cancelled_ || finished_) return false;
        pending_ += piece;
        size_t first = std::string::npos;
        size_t first_end = std::string::npos;
        for (const auto& s : stops_) {
            const auto pos = pending_.find(s);
            if (pos != std::string::npos && (pos + s.size() < first_end ||
                (pos + s.size() == first_end && pos < first))) {
                first = pos;
                first_end = pos + s.size();
            }
        }
        if (first != std::string::npos) {
            stopped_ = true;
            publish(first, true);
            pending_.clear();
            return false;
        }
        size_t keep = 0;
        for (const auto& s : stops_)
            for (size_t n = 1; n < s.size() && n <= pending_.size(); ++n)
                if (pending_.compare(pending_.size() - n, n, s, 0, n) == 0) keep = std::max(keep, n);
        return publish(pending_.size() - keep, false);
    }

    bool finish() {
        if (finished_ || stopped_ || cancelled_) return !cancelled_;
        finished_ = true;
        return publish(pending_.size(), true);
    }
    bool stopped() const { return stopped_; }
    bool cancelled() const { return cancelled_; }
    const std::string& text() const { return text_; }
};

} // namespace eie
