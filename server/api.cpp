// EIE — API Server Implementation
// Layer 1: OpenAI-compatible | Layer 2: Generic extensions
// Apache License 2.0
//
// Uses cpp-httplib (bundled with llama.cpp) for HTTP.
// Build with llama.cpp submodule to get httplib.h.
// Without it, this file provides the route stubs for integration.

#include "api.h"
#include <iostream>
#include <sstream>
#include <ctime>
#include <csignal>
#include <thread>
#include <chrono>
#include <set>
#include <algorithm>

// httplib + nlohmann/json come from the llama.cpp vendor tree;
// HAS_HTTPLIB is defined by CMake when the submodule is present.
#if defined(HAS_HTTPLIB) && HAS_HTTPLIB
#include "httplib.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

namespace eie {

#if defined(HAS_HTTPLIB) && HAS_HTTPLIB
// Un alias "de type embedding" (bge, embed, mmproj) n'est jamais choisi comme
// modèle de chat par défaut, et inversement sert de repli pour /v1/embeddings.
static bool looksLikeEmbedder(const std::string& alias) {
    std::string a = alias;
    for (auto& c : a) c = (char)tolower(c);
    return a.find("bge") != std::string::npos ||
           a.find("embed") != std::string::npos ||
           a.find("mmproj") != std::string::npos;
}

// Flatten an OpenAI-style messages[] array into a single prompt.
static std::string promptFromMessages(const json& body) {
    std::ostringstream ss;
    if (body.contains("messages") && body["messages"].is_array()) {
        for (auto& m : body["messages"]) {
            std::string role = m.value("role", "user");
            std::string content = m.contains("content") && m["content"].is_string()
                ? m["content"].get<std::string>() : "";
            if (role == "system")      ss << content << "\n\n";
            else if (role == "user")   ss << "User: " << content << "\n";
            else                       ss << "Assistant: " << content << "\n";
        }
        ss << "Assistant:";
    } else {
        ss << body.value("prompt", "");
    }
    return ss.str();
}

// Keep assistant tool calls and tool results in the history rendered by the
// model's native template. A tool reply commonly has null/empty content.
static std::vector<ChatMessage> messagesFromJson(const json& body) {
    std::vector<ChatMessage> msgs;
    if (!body.contains("messages")) return msgs;
    if (!body["messages"].is_array()) throw std::invalid_argument("messages must be an array");
    for (const auto& m : body["messages"]) {
        if (!m.is_object() || !m.value("role", json()).is_string())
            throw std::invalid_argument("each message needs a role");
        ChatMessage msg;
        msg.role = m.at("role").get<std::string>();
        if (msg.role != "system" && msg.role != "developer" && msg.role != "user" &&
            msg.role != "assistant" && msg.role != "tool")
            throw std::invalid_argument("unsupported message role");
        if (m.contains("content") && !m["content"].is_null()) {
            if (!m["content"].is_string()) throw std::invalid_argument("text message content must be a string or null");
            msg.content = m["content"].get<std::string>();
        }
        if (m.contains("name")) {
            if (!m["name"].is_string()) throw std::invalid_argument("message name must be a string");
            msg.name = m["name"].get<std::string>();
        }
        if (m.contains("reasoning_content") && !m["reasoning_content"].is_null()) {
            if (msg.role != "assistant" || !m["reasoning_content"].is_string())
                throw std::invalid_argument("reasoning_content must be an assistant string");
            msg.reasoning_content = m["reasoning_content"].get<std::string>();
        }
        if (m.contains("tool_call_id")) {
            if (!m["tool_call_id"].is_string()) throw std::invalid_argument("tool_call_id must be a string");
            msg.tool_call_id = m["tool_call_id"].get<std::string>();
        }
        if (msg.role == "tool" && msg.tool_call_id.empty())
            throw std::invalid_argument("tool message requires tool_call_id");
        if (m.contains("tool_calls")) {
            if (msg.role != "assistant" || !m["tool_calls"].is_array())
                throw std::invalid_argument("tool_calls must be an assistant array");
            for (const auto& item : m["tool_calls"]) {
                if (!item.is_object() || item.value("type", "") != "function" ||
                    !item.contains("function") || !item["function"].is_object() ||
                    !item["function"].value("name", json()).is_string() ||
                    !item["function"].value("arguments", json()).is_string() ||
                    !item.value("id", json()).is_string())
                    throw std::invalid_argument("invalid assistant tool_calls entry");
                msg.tool_calls.push_back({item["id"].get<std::string>(),
                    item["function"]["name"].get<std::string>(),
                    item["function"]["arguments"].get<std::string>()});
            }
        }
        msgs.push_back(std::move(msg));
    }
    return msgs;
}

static ChatFormatOptions toolOptionsFromJson(const json& body) {
    ChatFormatOptions options;
    options.enable_thinking = body.value("enable_thinking", false);
    options.reasoning_effort = body.value("reasoning_effort", "");
    if (body.contains("chat_template_kwargs")) {
        const auto& kwargs = body["chat_template_kwargs"];
        if (!kwargs.is_object()) throw std::invalid_argument("chat_template_kwargs must be an object");
        if (kwargs.contains("enable_thinking")) {
            if (!kwargs["enable_thinking"].is_boolean())
                throw std::invalid_argument("chat_template_kwargs.enable_thinking must be a boolean");
            options.enable_thinking = kwargs["enable_thinking"].get<bool>();
        }
    }
    if (options.reasoning_effort == "none") options.enable_thinking = false;
    options.parallel_tool_calls = body.value("parallel_tool_calls", false);
    if (body.contains("tools")) {
        if (!body["tools"].is_array()) throw std::invalid_argument("tools must be an array");
        std::set<std::string> names;
        for (const auto& item : body["tools"]) {
            if (!item.is_object() || item.value("type", "") != "function" ||
                !item.contains("function") || !item["function"].is_object())
                throw std::invalid_argument("only function tools are supported");
            const auto& fn = item["function"];
            if (!fn.value("name", json()).is_string() || fn["name"].get<std::string>().empty())
                throw std::invalid_argument("tool function requires a name");
            if (fn.contains("description") && !fn["description"].is_string())
                throw std::invalid_argument("tool description must be a string");
            if (fn.contains("parameters") && !fn["parameters"].is_object())
                throw std::invalid_argument("tool parameters must be a JSON object");
            ChatToolDefinition tool;
            tool.name = fn["name"].get<std::string>();
            if (!names.insert(tool.name).second) throw std::invalid_argument("duplicate tool function name");
            tool.description = fn.value("description", "");
            tool.parameters = fn.value("parameters", json{{"type", "object"}, {"properties", json::object()}}).dump();
            options.tools.push_back(std::move(tool));
        }
    }
    if (body.contains("tool_choice")) {
        const auto& choice = body["tool_choice"];
        if (choice.is_string()) {
            const auto value = choice.get<std::string>();
            if (value == "auto") options.tool_choice = ChatToolChoice::Auto;
            else if (value == "required") options.tool_choice = ChatToolChoice::Required;
            else if (value == "none") options.tool_choice = ChatToolChoice::None;
            else throw std::invalid_argument("tool_choice must be auto, required, none, or a named function");
        } else if (choice.is_object()) {
            if (choice.value("type", "") != "function" || !choice.contains("function") ||
                !choice["function"].is_object() || !choice["function"].value("name", json()).is_string())
                throw std::invalid_argument("invalid named tool_choice");
            const auto name = choice["function"]["name"].get<std::string>();
            auto it = std::find_if(options.tools.begin(), options.tools.end(),
                [&](const ChatToolDefinition& tool) { return tool.name == name; });
            if (it == options.tools.end()) throw std::invalid_argument("named tool_choice is not in tools");
            ChatToolDefinition selected = *it;
            options.tools = {std::move(selected)};
            options.tool_choice = ChatToolChoice::Required;
        } else {
            throw std::invalid_argument("invalid tool_choice");
        }
    }
    if (options.tool_choice == ChatToolChoice::Required && options.tools.empty())
        throw std::invalid_argument("tool_choice=required needs at least one tool");
    return options;
}

static SamplingParams samplingFromJson(const json& body) {
    SamplingParams sp;
    sp.temperature = body.value("temperature", sp.temperature);
    sp.top_p       = body.value("top_p", sp.top_p);
    sp.min_p       = body.value("min_p", sp.min_p);
    sp.repetition_penalty = body.value("repetition_penalty", body.value("repeat_penalty", sp.repetition_penalty));
    sp.presence_penalty   = body.value("presence_penalty", sp.presence_penalty);
    sp.top_k       = body.value("top_k", sp.top_k);
    sp.max_tokens  = body.value("max_tokens", 256);
    sp.one_shot    = body.value("one_shot", false);
    sp.ews_trace   = body.value("ews_trace", false);
    sp.truncate_prompt = body.value("truncate_prompt", true);
    if (body.contains("stop")) {
        if (body["stop"].is_string()) sp.stop.push_back(body["stop"]);
        else if (body["stop"].is_array())
            for (auto& s : body["stop"])
                if (s.is_string()) sp.stop.push_back(s);
    }
    return sp;
}
#endif

// ═══════════════════════════════════════════
// JSON Helpers
// ═══════════════════════════════════════════

static std::string escapeJson(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    const char* hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[(static_cast<unsigned char>(c) >> 4) & 15];
                    out += hex[static_cast<unsigned char>(c) & 15];
                } else out += c;
        }
    }
    return out;
}

static std::string chatCompletionJson(const InferenceResult& r, const std::string& model) {
    std::ostringstream ss;
    ss << "{\"id\":\"eie-" << std::time(nullptr) << "\","
       << "\"object\":\"chat.completion\","
       << "\"model\":\"" << escapeJson(model) << "\","
       << "\"choices\":[{\"index\":0,"
       << "\"message\":{\"role\":\"assistant\",\"content\":\"" << escapeJson(r.text) << "\"";
    if (!r.reasoning_content.empty())
        ss << ",\"reasoning_content\":\"" << escapeJson(r.reasoning_content) << "\"";
    if (!r.tool_calls.empty()) {
        ss << ",\"tool_calls\":[";
        for (size_t i = 0; i < r.tool_calls.size(); ++i) {
            if (i) ss << ',';
            const auto& call = r.tool_calls[i];
            ss << "{\"id\":\"" << escapeJson(call.id) << "\",\"type\":\"function\","
               << "\"function\":{\"name\":\"" << escapeJson(call.name)
               << "\",\"arguments\":\"" << escapeJson(call.arguments) << "\"}}";
        }
        ss << ']';
    }
    ss << "},\"finish_reason\":\"" << escapeJson(r.finish_reason) << "\"}],"
       << "\"usage\":{\"prompt_tokens\":" << r.prompt_tokens << ",\"completion_tokens\":" << r.tokens
       << ",\"total_tokens\":" << (int64_t(r.prompt_tokens) + r.tokens)
       << ",\"prompt_tokens_details\":{\"cached_tokens\":" << std::max(0, r.reused_tokens) << "}}}";
    return ss.str();
}

static std::string groupResultJson(const GroupResult& r) {
    std::ostringstream ss;
    ss << "{\"group\":\"" << escapeJson(r.group) << "\",\"responses\":[";
    for (size_t i = 0; i < r.responses.size(); i++) {
        if (i > 0) ss << ",";
        ss << "{\"model\":\"" << escapeJson(r.responses[i].model) << "\","
           << "\"content\":\"" << escapeJson(r.responses[i].text) << "\","
           << "\"error\":\"" << escapeJson(r.responses[i].error) << "\","
           << "\"latency_ms\":" << r.responses[i].latency_ms << ","
           << "\"ok\":" << (r.responses[i].ok ? "true" : "false") << "}";
    }
    ss << "],\"attempts\":[";
    for (size_t i = 0; i < r.attempts.size(); ++i) {
        if (i) ss << ',';
        const auto& a = r.attempts[i];
        ss << "{\"requested_model\":\"" << escapeJson(a.requested_model)
           << "\",\"action\":\"" << a.action << "\",\"model\":\"" << escapeJson(a.result.model)
           << "\",\"ok\":" << (a.result.ok ? "true" : "false")
           << ",\"error\":\"" << escapeJson(a.result.error) << "\",\"content\":\"" << escapeJson(a.result.text)
           << "\",\"finish_reason\":\"" << escapeJson(a.result.finish_reason) << "\"}";
    }
    ss << "],\"completed\":" << r.completed
       << ",\"required\":" << r.required
       << ",\"status\":\"" << r.status << "\""
       << ",\"latency_ms\":" << r.latency_ms << "}";
    return ss.str();
}

static std::string modelsJson(const std::vector<std::string>& list) {
    std::ostringstream ss;
    ss << "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < list.size(); i++) {
        if (i > 0) ss << ",";
        ss << "{\"id\":\"" << list[i] << "\",\"object\":\"model\",\"owned_by\":\"local\"}";
    }
    ss << "]}";
    return ss.str();
}

// ═══════════════════════════════════════════
// Server Implementation
// ═══════════════════════════════════════════

void startServer(const ServerConfig& cfg, ModelManager& models,
                 GroupScheduler& scheduler, PolicyStrategy* policy,
                 Metrics& metrics, AuditTrail& audit) {

#if defined(HAS_HTTPLIB) && HAS_HTTPLIB
    // Arrêt propre : SIGTERM/SIGINT doivent STOPPER le serveur, pas juste
    // lever un drapeau que personne ne lit (les processus survivaient aux
    // kills et s'accumulaient). Le handler appelle svr.stop() -> listen()
    // rend la main -> shutdown normal.
    static httplib::Server svr;
    static auto stopHandler = [](int) { svr.stop(); };
    signal(SIGINT, stopHandler);
    signal(SIGTERM, stopHandler);

    // Optional existing preset credential. An empty token preserves local use.
    // Check before application handlers, including streaming and administration.
    auto authorize = [&cfg](const httplib::Request& req, httplib::Response& res) {
        if (cfg.auth_token.empty()) return httplib::Server::HandlerResponse::Unhandled;
        bool authorized = false;
        if (req.get_header_value_count("Authorization") == 1) {
            const auto value = req.get_header_value("Authorization");
            const auto separator = value.find(' ');
            std::string scheme = value.substr(0, separator);
            for (char& c : scheme) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (scheme == "bearer" && separator != std::string::npos) {
                const auto token = value.substr(separator + 1);
                unsigned difference = token.size() == cfg.auth_token.size() ? 0 : 1;
                for (size_t i = 0; i < cfg.auth_token.size(); ++i)
                    difference |= static_cast<unsigned char>(cfg.auth_token[i]) ^
                        (i < token.size() ? static_cast<unsigned char>(token[i]) : 0);
                authorized = difference == 0;
            }
        }
        if (authorized) return httplib::Server::HandlerResponse::Unhandled;
        res.status = 401;
        res.set_header("WWW-Authenticate", "Bearer");
        res.set_content("{\"error\":{\"type\":\"authentication_error\",\"message\":\"A valid Authorization: Bearer token is required.\"}}", "application/json");
        return httplib::Server::HandlerResponse::Handled;
    };
    svr.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        // Let httplib consume POST bodies before replying, avoiding a Windows
        // reset if the body arrives after an early 401. No application work yet.
        return req.method == "POST" ? httplib::Server::HandlerResponse::Unhandled : authorize(req, res);
    });
    auto post = [&](const std::string& path, httplib::Server::Handler handler) {
        svr.Post(path, [&, handler](const httplib::Request& req, httplib::Response& res) {
            if (authorize(req, res) == httplib::Server::HandlerResponse::Unhandled) handler(req, res);
        });
    };

    // ════════════════════════════════════
    // LAYER 1 — OpenAI Compatible
    // ════════════════════════════════════

    // POST /v1/chat/completions
    post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid JSON\"}", "application/json");
            return;
        }
        std::string model;
        SamplingParams sp;
        ChatFormatOptions format;
        std::vector<ChatMessage> messages;
        bool strict_model = false, stream = false;
        try {
            model = body.value("model", "default");
            sp = samplingFromJson(body);
            format = toolOptionsFromJson(body);
            messages = messagesFromJson(body);
            strict_model = body.value("strict_model", false);
            stream = body.value("stream", false);
            if (!format.tools.empty() && format.tool_choice != ChatToolChoice::None && messages.empty())
                throw std::invalid_argument("tool calling requires messages");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", {{"message", e.what()}, {"type", "invalid_request_error"}}}}).dump(), "application/json");
            return;
        }

        auto* backend = models.get(model);
        if (!backend && !strict_model) {
            // repli : l'unique modèle de GÉNÉRATION chargé (les embedders sont exclus)
            std::vector<std::string> gen;
            for (auto& a : models.loaded())
                if (!looksLikeEmbedder(a)) gen.push_back(a);
            if (gen.size() == 1) { model = gen[0]; backend = models.get(model); }
        }

        // Template de chat natif du modèle pour messages[] ; repli générique sinon.
        if (!backend) {
            res.status = 404;
            res.set_content("{\"error\":\"model not found\"}", "application/json");
            return;
        }
        // Keep model-specific formatting, grammar and parser together per request.
        ChatPrompt rendered = backend->formatChatRequest(messages, format);
        if (!rendered.error.empty()) {
            res.status = 400;
            res.set_content(json({{"error", {{"message", rendered.error}, {"type", "invalid_request_error"}}}}).dump(), "application/json");
            return;
        }
        const bool wants_tools = !format.tools.empty() && format.tool_choice != ChatToolChoice::None;
        if (rendered.prompt.empty() && wants_tools) {
            res.status = 400;
            res.set_content(json({{"error", {{"message", "model chat template cannot render tool calls"},
                {"type", "invalid_request_error"}}}}).dump(), "application/json");
            return;
        }
        std::string prompt = rendered.prompt.empty() ? promptFromMessages(body) : rendered.prompt;
        sp.chat_format = std::move(rendered);
        sp.stop.insert(sp.stop.end(), sp.chat_format.additional_stops.begin(), sp.chat_format.additional_stops.end());

        if (stream) {
            // SSE : chunks OpenAI (chat.completion.chunk) au fil de la génération.
            std::string prompt_copy = prompt, model_copy = model;
            SamplingParams sp_copy = sp;
            res.set_header("Cache-Control", "no-cache");
            res.set_chunked_content_provider("text/event-stream",
                [&, backend, prompt_copy, model_copy, sp_copy](size_t, httplib::DataSink& sink) mutable {
                    long id = (long)std::time(nullptr);
                    auto emit = [&](const json& j) {
                        std::string line = "data: " + j.dump() + "\n\n";
                        return sink.write(line.data(), line.size());
                    };
                    sp_copy.should_continue = [&sink] { return sink.is_writable(); };
                    if (!sp_copy.chat_format.parse_chat_output) sp_copy.on_token = [&](const std::string& piece) {
                        return emit({{"id", "eie-" + std::to_string(id)},
                                     {"object", "chat.completion.chunk"},
                                     {"model", model_copy},
                                     {"choices", {{{"index", 0},
                                                   {"delta", {{"content", piece}}},
                                                   {"finish_reason", nullptr}}}}});
                    };
                    auto result = backend->chat(prompt_copy, sp_copy);
                    // The sink only lives for this provider invocation.
                    sp_copy.on_token = {};
                    sp_copy.should_continue = {};
                    if (result.finish_reason == "cancelled") return false;
                    if (!result.ok) {
                        emit({{"error", {{"message", result.error}, {"type", "inference_error"},
                              {"code", result.error == "context_length_exceeded" ? "context_length_exceeded" : "inference_error"}}}});
                        sink.done();
                        return true;
                    }
                    std::cout << "[KV] reused=" << result.reused_tokens
                              << " gen=" << result.tokens
                              << " total_ms=" << (int)result.latency_ms << " (stream)" << std::endl;
                    metrics.recordModel(model_copy, result.latency_ms, result.tokens);
                    if (sp_copy.chat_format.parse_chat_output) {
                        // Classify complete output before emitting deltas: model control
                        // tokens must not leak as delta.content.
                        if (!result.reasoning_content.empty()) emit({{"id", "eie-" + std::to_string(id)},
                            {"object", "chat.completion.chunk"}, {"model", model_copy},
                            {"choices", {{{"index", 0}, {"delta", {{"reasoning_content", result.reasoning_content}}},
                                          {"finish_reason", nullptr}}}}});
                        if (!result.text.empty()) emit({{"id", "eie-" + std::to_string(id)},
                            {"object", "chat.completion.chunk"}, {"model", model_copy},
                            {"choices", {{{"index", 0}, {"delta", {{"content", result.text}}},
                                          {"finish_reason", nullptr}}}}});
                        if (!result.tool_calls.empty()) {
                            json calls = json::array();
                            for (size_t i = 0; i < result.tool_calls.size(); ++i) {
                                const auto& call = result.tool_calls[i];
                                calls.push_back({{"index", i}, {"id", call.id}, {"type", "function"},
                                    {"function", {{"name", call.name}, {"arguments", call.arguments}}}});
                            }
                            emit({{"id", "eie-" + std::to_string(id)},
                                {"object", "chat.completion.chunk"}, {"model", model_copy},
                                {"choices", {{{"index", 0}, {"delta", {{"tool_calls", calls}}},
                                              {"finish_reason", nullptr}}}}});
                        }
                    }
                    emit({{"id", "eie-" + std::to_string(id)},
                          {"object", "chat.completion.chunk"},
                          {"model", model_copy},
                          {"choices", {{{"index", 0}, {"delta", json::object()},
                                        {"finish_reason", result.finish_reason}}}},
                          {"usage", {{"prompt_tokens", result.prompt_tokens},
                                     {"completion_tokens", result.tokens},
                                     {"total_tokens", int64_t(result.prompt_tokens) + result.tokens},
                                     {"prompt_tokens_details", {{"cached_tokens", std::max(0, result.reused_tokens)}}}}}});
                    std::string done = "data: [DONE]\n\n";
                    sink.write(done.data(), done.size());
                    sink.done();
                    return true;
                });
            return;
        }

        auto result = backend->chat(prompt, sp);
        if (!result.ok) {
            const bool overflow = result.error == "context_length_exceeded";
            res.status = overflow ? 400 : 500;
            res.set_content(json({{"error", {{"message", result.error},
                {"type", overflow ? "invalid_request_error" : "inference_error"},
                {"code", overflow ? "context_length_exceeded" : "inference_error"}}}}).dump(), "application/json");
            return;
        }
        std::cout << "[KV] reused=" << result.reused_tokens
                  << " gen=" << result.tokens
                  << " total_ms=" << (int)result.latency_ms << std::endl;
        metrics.recordModel(model, result.latency_ms, result.tokens);
        res.set_content(chatCompletionJson(result, model), "application/json");
    });

    // GET /v1/models
    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(modelsJson(models.loaded()), "application/json");
    });

    // GET /health
    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(metrics.healthJson(models.loaded().size()), "application/json");
    });

    // POST /v1/embeddings — OpenAI-compatible
    post("/v1/embeddings", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid JSON\"}", "application/json");
            return;
        }
        std::string model = body.value("model", "embedder");
        auto* backend = models.get(model);
        if (!backend) {
            // repli : le premier modèle chargé de type embedding
            for (auto& a : models.loaded())
                if (looksLikeEmbedder(a)) { model = a; backend = models.get(a); break; }
        }
        if (!backend) {
            res.status = 404;
            res.set_content("{\"error\":\"no embedding model loaded\"}", "application/json");
            return;
        }

        std::vector<std::string> inputs;
        if (body.contains("input")) {
            if (body["input"].is_string()) inputs.push_back(body["input"]);
            else if (body["input"].is_array())
                for (auto& s : body["input"])
                    if (s.is_string()) inputs.push_back(s);
        }
        if (inputs.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"missing input\"}", "application/json");
            return;
        }

        json data = json::array();
        for (size_t i = 0; i < inputs.size(); i++) {
            auto t0 = std::chrono::steady_clock::now();
            auto vec = backend->embed(inputs[i]);
            if (vec.empty()) {
                res.status = 500;
                res.set_content("{\"error\":\"embedding failed\"}", "application/json");
                return;
            }
            auto ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            metrics.recordModel(model, ms, 0);
            data.push_back({{"object", "embedding"}, {"index", i}, {"embedding", vec}});
        }
        json out = {{"object", "list"}, {"model", model}, {"data", data}};
        res.set_content(out.dump(), "application/json");
    });

    // ════════════════════════════════════
    // LAYER 2 — Generic Extensions
    // ════════════════════════════════════

    // POST /v1/batch/execute — parallel group execution
    post("/v1/batch/execute", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid JSON\"}", "application/json");
            return;
        }
        std::string group_name = body.value("group", "core");
        // Chaque backend du groupe templte les mêmes messages avec le
        // template natif de son propre GGUF ; repli générique sinon.
        auto msgs = messagesFromJson(body);
        std::string fallback = promptFromMessages(body);
        SamplingParams sp = samplingFromJson(body);

        auto it = policy->groups.find(group_name);
        if (it == policy->groups.end()) {
            res.status = 404;
            res.set_content("{\"error\":\"group not found\"}", "application/json");
            return;
        }

        auto backends = models.getAll();
        auto result = scheduler.exec(it->second, msgs, fallback, sp, backends);
        metrics.recordGroup(group_name, result.status == "complete", result.latency_ms);
        audit.record(group_name, result.status, "prompt_hash");
        res.set_content(groupResultJson(result), "application/json");
    });

    // POST /v1/chain/execute — sequential chain
    post("/v1/chain/execute", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid JSON\"}", "application/json");
            return;
        }
        std::string group_name = body.value("group", "chain");
        auto msgs = messagesFromJson(body);
        std::string fallback = promptFromMessages(body);
        SamplingParams sp = samplingFromJson(body);

        auto it = policy->groups.find(group_name);
        if (it == policy->groups.end()) {
            res.status = 404;
            res.set_content("{\"error\":\"group not found\"}", "application/json");
            return;
        }

        auto backends = models.getAll();
        auto result = scheduler.execSequential(it->second, msgs, fallback, sp, backends);
        metrics.recordGroup(group_name, result.status == "complete", result.latency_ms);
        res.set_content(groupResultJson(result), "application/json");
    });

    // GET /v1/admin/models/discover
    svr.Get("/v1/admin/models/discover", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(modelsJson(models.available()), "application/json");
    });

    // GET /v1/admin/vram/status
    svr.Get("/v1/admin/vram/status", [&](const httplib::Request&, httplib::Response& res) {
        json data = json::array();
        for (auto & [alias, backend] : models.getAll()) {
            auto v = backend->vram();
            data.push_back({{"model", alias}, {"gpu_id", v.gpu_id}, {"total_bytes", v.total_bytes},
                {"used_bytes", v.used_bytes}, {"free_bytes", v.free_bytes}});
        }
        res.set_content(json({{"devices_by_model", data}}).dump(), "application/json");
    });

    svr.Get("/v1/admin/ews/routing", [&](const httplib::Request&, httplib::Response& res) {
        json data = json::object();
        for (auto& [alias, backend] : models.getAll())
            data[alias] = json::parse(backend->streamingRouting().json());
        res.set_content(data.dump(), "application/json");
    });

    svr.Get("/v1/admin/ews/status", [&](const httplib::Request&, httplib::Response& res) {
        json data = json::object();
        for (auto & [alias, backend] : models.getAll()) data[alias] = backend->streamingStats();
        res.set_content(data.dump(), "application/json");
    });

    // GET /v1/admin/scheduling/status
    svr.Get("/v1/admin/scheduling/status", [&](const httplib::Request&, httplib::Response& res) {
        std::ostringstream ss;
        ss << "{\"strategy\":\"" << policy->name() << "\","
           << "\"groups\":" << policy->groups.size() << "}";
        res.set_content(ss.str(), "application/json");
    });

    // POST /v1/admin/config/reload
    post("/v1/admin/config/reload", [&](const httplib::Request&, httplib::Response& res) {
        // TODO: reload config from file
        res.set_content("{\"status\":\"reload not yet implemented\"}", "application/json");
    });

    // GET /metrics — Prometheus
    svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(metrics.prometheus(models.loaded().size()), "text/plain");
    });

    // GET /v1/admin/health/deep
    svr.Get("/v1/admin/health/deep", [&](const httplib::Request&, httplib::Response& res) {
        // TODO: test inference on each loaded model
        res.set_content(metrics.healthJson(models.loaded().size()), "application/json");
    });

    std::cout << "[EIE] Server listening on " << cfg.host << ":" << cfg.port << std::endl;
    svr.listen(cfg.host, cfg.port);

#else
    // No httplib — standalone mode for testing
    std::cout << "[EIE] httplib not available — running in standalone test mode" << std::endl;
    std::cout << "[EIE] Server would listen on " << cfg.host << ":" << cfg.port << std::endl;
    std::cout << std::endl;

    // Demo: execute a group if any is configured
    if (!policy->groups.empty()) {
        auto& [gname, group] = *policy->groups.begin();
        std::cout << "[EIE] Demo: executing group '" << gname << "' ("
                  << group.models.size() << " models, " << group.type << ")" << std::endl;

        auto backends = models.getAll();
        if (!backends.empty()) {
            SamplingParams sp;
            auto result = scheduler.exec(group, "Hello, this is a test prompt.", sp, backends);

            std::cout << "\n[EIE] Group result:" << std::endl;
            std::cout << "  Status: " << result.status << std::endl;
            std::cout << "  Completed: " << result.completed << "/" << result.required << std::endl;
            std::cout << "  Latency: " << result.latency_ms << " ms" << std::endl;
            for (auto& r : result.responses) {
                std::cout << "  [" << r.model << "] " << r.text << " (" << r.latency_ms << "ms)" << std::endl;
            }

            metrics.recordGroup(gname, result.status == "complete", result.latency_ms);
            audit.record(gname, result.status, "test_prompt_hash");
        }
    }

    // Print metrics
    std::cout << "\n[EIE] Metrics:\n" << metrics.prometheus(models.loaded().size()) << std::endl;
    std::cout << "[EIE] Health: " << metrics.healthJson(models.loaded().size()) << std::endl;
    std::cout << "[EIE] Audit records: " << audit.size() << std::endl;

    std::cout << "\n[EIE] Standalone test complete." << std::endl;
    std::cout << "[EIE] To enable HTTP server, build with llama.cpp submodule (provides httplib.h)" << std::endl;
    std::cout << "[EIE] Then define HAS_HTTPLIB before including api.cpp" << std::endl;
#endif
}

} // namespace eie
