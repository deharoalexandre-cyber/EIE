// EWS P1a.2 — décomposition sur VRAIE inférence (Gemma-4-A4B, llama.cpp API).
//
// Trois modes ablatifs (runs séparés, mêmes prompt/params, greedy argmax):
//   mode 0: inférence pure, pas de callback            -> T0 (router_compute + FFN inclus)
//   mode 1: cb_eval lit le top-k de chaque couche MoE  -> T1-T0 = host_visibility (graph break + sync + D2H)
//   mode 2: + SLRU (pin 75%/SLRU 25%, calibré sur trace brûlée) + fetch RÉEL
//           chunké 512 KiB (NVMe->SHA->H2D) avec attente expert_ready DANS le
//           callback                                   -> T2-T1 = routing-dependent stall réel
//
// Le FFN consomme les poids résidents du modèle; les octets streamés font le
// trajet complet vers une arène VRAM partagée — timing équivalent au moteur
// final (les deux copies sont en VRAM), sémantique SLRU par couche préservée.
// Fail-closed conservé (digest complet du slab avant liberation du wait).
//
// Usage: ews_p1a2 <gguf> <prompt.txt> <trace_calib.tsv> <mode 0|1|2> [n_predict] [n_gpu_layers]

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>

#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <deque>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#pragma comment(lib, "bcrypt.lib")

static double now_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct Sha256 {
    BCRYPT_ALG_HANDLE alg = nullptr;
    Sha256() { BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0); }
    ~Sha256() { if (alg) BCryptCloseAlgorithmProvider(alg, 0); }
};

struct LayerMeta {
    size_t   gu_slab = 0, dn_slab = 0;
    uint64_t gu_off = 0, dn_off = 0;
    ggml_tensor * gu_arena = nullptr;   // arene physique PARTAGEE entre couches
    ggml_tensor * dn_arena = nullptr;   // (timing H2D identique, VRAM bornee)
};

// ---- SLRU (identique p1a) ----
struct Slru {
    std::set<int32_t>  pinned;
    std::deque<int32_t> probation, protected_;
    std::map<int32_t, int32_t> slot_of;
    std::vector<int32_t> free_slots;
    int n_prob = 0, n_prot = 0;
    const std::set<int32_t> * in_use = nullptr;

    int32_t lookup(int32_t e) { auto it = slot_of.find(e); return it == slot_of.end() ? -1 : it->second; }
    void touch(int32_t e) {
        if (pinned.count(e)) return;
        auto inP = std::find(protected_.begin(), protected_.end(), e);
        if (inP != protected_.end()) { protected_.erase(inP); protected_.push_back(e); return; }
        auto inQ = std::find(probation.begin(), probation.end(), e);
        if (inQ != probation.end()) {
            probation.erase(inQ);
            if (n_prot > 0) {
                protected_.push_back(e);
                if ((int) protected_.size() > n_prot) {
                    int32_t dem = protected_.front(); protected_.pop_front();
                    probation.push_back(dem);
                    if ((int) probation.size() > n_prob) free_slots.push_back(evict_from(probation));
                }
            } else probation.push_back(e);
        }
    }
    int32_t try_admit(int32_t e) {
        int32_t slot;
        if (!free_slots.empty()) { slot = free_slots.back(); free_slots.pop_back(); }
        else { slot = evict_any(); if (slot < 0) return -1; }
        probation.push_back(e);
        slot_of[e] = slot;
        return slot;
    }
    int32_t evict_any() {
        for (auto * q : { &probation, &protected_ })
            for (auto it = q->begin(); it != q->end(); ++it)
                if (!in_use || !in_use->count(*it)) {
                    int32_t v = *it; q->erase(it);
                    int32_t s = slot_of[v]; slot_of.erase(v);
                    return s;
                }
        return -1;
    }
    int32_t evict_from(std::deque<int32_t> & q) {
        int32_t v = q.front(); q.pop_front();
        int32_t s = slot_of[v]; slot_of.erase(v);
        return s;
    }
};

// ---- pool de fetch: SHA PAR CHUNK (spec GPT / format model.ews.idx v1) ----
// Granularite de tache = CHUNK (512 KiB). Chaque chunk: lecture overlapped,
// SHA-256 independant, verification contre la table (clef layer+expert+chunk,
// modele fixe par processus — binding model/version au chargement de l'index
// signe dans le vrai format), H2D. Les workers parallelisent les chunks d'un
// meme miss. Le slot ne passe READY (pending[batch]==0, donc fin du wait)
// qu'apres validation de TOUS les chunks: fail-closed inchange.
struct FetchPool {
    static const uint64_t ALIGN = 4096;
    struct Task {
        int l; int32_t e; int32_t slot; int batch; int chunk_idx;
        uint64_t f0; size_t asz, pay_delta, pay_n;
        ggml_tensor * dst; size_t dst_off;
    };
    std::mutex m;
    std::condition_variable cv_task, cv_done;
    std::deque<Task> q;
    std::map<int, int> pending;
    std::map<int, std::chrono::steady_clock::time_point> batch_t0;
    bool stop = false;
    std::vector<std::thread> workers;
    std::atomic<int64_t> ns_ready{0};
    std::atomic<int64_t> n_tasks{0};
    std::mutex h2d_mutex, dig_mutex;
    const char * model_path = nullptr;
    std::map<int, LayerMeta> * layers = nullptr;
    std::map<int64_t, std::vector<uint8_t>> * digests = nullptr;   // clef (l,e,chunk)
    size_t chunk_bytes = 512 * 1024;
    bool no_h2d = false;   // attribution seulement: octets lus+hashes, pas uploades

    static int64_t dkey(int l, int32_t e, int c) { return ((int64_t) l << 40) | ((int64_t)(uint32_t) e << 16) | c; }

    void start(int n) { for (int i = 0; i < n; ++i) workers.emplace_back([this] { run(); }); }
    // decoupe un miss d'expert en taches-chunks partageant un batch
    void submit_expert(int l, int32_t e, int32_t slot, int batch) {
        LayerMeta & lm = (*layers)[l];
        std::vector<Task> ts;
        int ci = 0;
        auto cut = [&](uint64_t off, size_t n, ggml_tensor * dst, size_t base) {
            const uint64_t a0 = off & ~(ALIGN - 1), a1 = (off + n + ALIGN - 1) & ~(ALIGN - 1);
            for (uint64_t c = a0; c < a1; c += chunk_bytes) {
                const size_t asz = (size_t) std::min<uint64_t>(chunk_bytes, a1 - c);
                const uint64_t p0 = std::max<uint64_t>(c, off), p1 = std::min<uint64_t>(c + asz, off + n);
                ts.push_back({ l, e, slot, batch, ci++, c, asz, (size_t)(p0 - c), (size_t)(p1 - p0), dst, base + (size_t)(p0 - off) });
            }
        };
        cut(lm.gu_off + (uint64_t) e * lm.gu_slab, lm.gu_slab, lm.gu_arena, (size_t) slot * lm.gu_slab);
        cut(lm.dn_off + (uint64_t) e * lm.dn_slab, lm.dn_slab, lm.dn_arena, (size_t) slot * lm.dn_slab);
        {
            std::lock_guard<std::mutex> lk(m);
            pending[batch] = (int) ts.size();
            batch_t0[batch] = std::chrono::steady_clock::now();
            for (auto & t : ts) q.push_back(t);
        }
        cv_task.notify_all();
    }
    double wait_batch(int batch) {
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [&] { auto it = pending.find(batch); return it == pending.end() || it->second == 0; });
        pending.erase(batch);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    void shutdown() {
        { std::lock_guard<std::mutex> lk(m); stop = true; }
        cv_task.notify_all();
        for (auto & w : workers) w.join();
    }
    void run() {
        HANDLE hf = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        Sha256 sha;
        uint8_t * buf = (uint8_t *) _aligned_malloc(chunk_bytes + ALIGN, ALIGN);
        HANDLE ev = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        for (;;) {
            Task t;
            {
                std::unique_lock<std::mutex> lk(m);
                cv_task.wait(lk, [&] { return stop || !q.empty(); });
                if (stop && q.empty()) break;
                t = q.front(); q.pop_front();
            }
            // lecture du chunk (overlapped, handle propre au worker)
            OVERLAPPED ov = {};
            ov.Offset = (DWORD)(t.f0 & 0xFFFFFFFF);
            ov.OffsetHigh = (DWORD)(t.f0 >> 32);
            ResetEvent(ev);
            ov.hEvent = ev;
            if (!ReadFile(hf, buf, (DWORD) t.asz, nullptr, &ov) && GetLastError() != ERROR_IO_PENDING) { fprintf(stderr, "ReadFile KO\n"); exit(1); }
            WaitForSingleObject(ev, INFINITE);
            DWORD got = 0;
            GetOverlappedResult(hf, &ov, &got, FALSE);
            // SHA-256 du chunk, verification contre la table (clef l+e+chunk)
            uint8_t dig[32];
            {
                BCRYPT_HASH_HANDLE h = nullptr;
                BCryptCreateHash(sha.alg, &h, nullptr, 0, nullptr, 0, 0);
                BCryptHashData(h, (PUCHAR)(buf + t.pay_delta), (ULONG) t.pay_n, 0);
                BCryptFinishHash(h, dig, 32, 0);
                BCryptDestroyHash(h);
            }
            {
                std::lock_guard<std::mutex> lk(dig_mutex);
                const int64_t key = dkey(t.l, t.e, t.chunk_idx);
                auto it = digests->find(key);
                if (it == digests->end()) (*digests)[key] = std::vector<uint8_t>(dig, dig + 32);
                else if (memcmp(it->second.data(), dig, 32) != 0) { fprintf(stderr, "SHA MISMATCH chunk — fail-closed\n"); exit(3); }
            }
            if (!no_h2d) {
                std::lock_guard<std::mutex> lk(h2d_mutex);
                ggml_backend_tensor_set(t.dst, buf + t.pay_delta, t.dst_off, t.pay_n);
            }
            {
                std::lock_guard<std::mutex> lk(m);
                if (--pending[t.batch] == 0) {
                    ns_ready += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - batch_t0[t.batch]).count();
                    n_tasks++;
                    batch_t0.erase(t.batch);
                    cv_done.notify_all();
                }
            }
        }
        CloseHandle(ev);
        _aligned_free(buf);
        CloseHandle(hf);
    }
};

// ---- etat global du callback ----
struct SimState {
    int mode = 0;
    bool no_wait = false;
    std::map<int, LayerMeta> layers;
    std::map<int, Slru> slru;
    std::map<int64_t, std::vector<uint8_t>> digests;
    FetchPool pool;
    int first_layer = -1;
    int batch_id = 0;
    // metriques decode-only
    double t_vis = 0, t_pol = 0, t_stall = 0;
    int64_t n_hit = 0, n_miss = 0, n_acc = 0;
    int n_tok = 0;
    uint64_t cold_bytes = 0;
    size_t slab_total = 0;
};

static bool eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    SimState * s = (SimState *) ud;
    const bool is_topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    if (ask) return is_topk;
    if (!is_topk) return true;
    if (t->ne[1] != 1) return true;   // prefill: pas de politique (decode-only)

    const int l = atoi(t->name + 13);
    if (s->first_layer < 0) s->first_layer = l;
    if (l == s->first_layer) s->n_tok++;

    // host_visibility: rendre le top-k GPU visible cote hote
    const double v0 = now_ms();
    int32_t ids[16] = {};
    const int k = (int) t->ne[0];
    ggml_backend_tensor_get(t, ids, 0, (size_t) k * sizeof(int32_t));
    s->t_vis += now_ms() - v0;

    if (s->mode < 2) return true;

    // policy + fetch reel + attente expert_ready (le stall s'injecte ICI, dans
    // la vraie inference, au vrai point de dependance causale)
    auto & sl = s->slru[l];
    const double p0 = now_ms();
    std::set<int32_t> in_use(ids, ids + k);
    sl.in_use = &in_use;
    std::vector<int> needed;
    for (int i = 0; i < k; ++i) {
        const int32_t e = ids[i];
        s->n_acc++;
        int32_t slot = sl.lookup(e);
        if (slot >= 0) { s->n_hit++; sl.touch(e); continue; }
        s->n_miss++;
        slot = sl.try_admit(e);
        if (slot < 0) { fprintf(stderr, "SLRU sature\n"); exit(4); }
        s->pool.submit_expert(l, e, slot, s->batch_id);   // eclate en taches-chunks
        needed.push_back(s->batch_id);
        s->batch_id++;
        s->cold_bytes += s->slab_total;
    }
    sl.in_use = nullptr;
    s->t_pol += now_ms() - p0;
    if (!s->no_wait) for (int b : needed) s->t_stall += s->pool.wait_batch(b);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <gguf> <prompt.txt> <trace_calib.tsv> <mode 0|1|2> [n_predict=256] [n_gpu=24]\n", argv[0]); return 1; }
    const char * model_path = argv[1];
    const char * prompt_path = argv[2];
    const char * trace_path = argv[3];
    const int    mode      = atoi(argv[4]);
    const int    n_predict = argc > 5 ? atoi(argv[5]) : 256;
    const int    n_gpu     = argc > 6 ? atoi(argv[6]) : 24;
    const int    n_sha     = argc > 7 ? atoi(argv[7]) : 3;   // attribution: contention CPU
    const int    attrib    = argc > 8 ? atoi(argv[8]) : 0;   // bit0: no_h2d (CUDA), bit1: no_wait (drain pipeline)
    const bool   no_h2d    = (attrib & 1) != 0;
    const bool   no_wait   = (attrib & 2) != 0;              // ATTRIBUTION SEULEMENT: poids non attendus

    printf("== EWS P1a.2 vraie inference, mode %d ==\n", mode);

    SimState sim;
    sim.mode = mode;
    sim.no_wait = no_wait;
    if (no_wait) printf("ATTRIBUTION: no_wait actif (fetches emis, jamais attendus — timing seulement)\n");

    // ---- meta GGUF (offsets des experts) + arene physique partagee
    ggml_backend_t arena_backend = nullptr;
    if (mode >= 2) {
        ggml_context * meta = nullptr;
        gguf_init_params gp = { true, &meta };
        gguf_context * g = gguf_init_from_file(model_path, gp);
        const uint64_t data_off = gguf_get_data_offset(g);
        char name[128];
        int64_t gu_ne0 = 0, gu_ne1 = 0, dn_ne0 = 0, dn_ne1 = 0;
        enum ggml_type gu_ty = GGML_TYPE_F32, dn_ty = GGML_TYPE_F32;
        for (int l = 0; l < 256; ++l) {
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_up_exps.weight", l);
            int64_t ti = gguf_find_tensor(g, name);
            if (ti < 0) continue;
            ggml_tensor * gu = ggml_get_tensor(meta, name);
            snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
            ggml_tensor * dn = ggml_get_tensor(meta, name);
            LayerMeta lm;
            lm.gu_slab = ggml_nbytes(gu) / gu->ne[2];
            lm.dn_slab = ggml_nbytes(dn) / dn->ne[2];
            lm.gu_off  = data_off + gguf_get_tensor_offset(g, gguf_find_tensor(g, gu->name));
            lm.dn_off  = data_off + gguf_get_tensor_offset(g, ti >= 0 ? gguf_find_tensor(g, dn->name) : 0);
            sim.layers[l] = lm;
            gu_ne0 = gu->ne[0]; gu_ne1 = gu->ne[1]; gu_ty = gu->type;
            dn_ne0 = dn->ne[0]; dn_ne1 = dn->ne[1]; dn_ty = dn->type;
        }
        sim.slab_total = sim.layers.begin()->second.gu_slab + sim.layers.begin()->second.dn_slab;
        const int n_slots = 32;
        arena_backend = ggml_backend_cuda_init(0);
        static ggml_context * actx = nullptr;
        ggml_init_params aip = { ggml_tensor_overhead() * 8, nullptr, true };
        actx = ggml_init(aip);
        ggml_tensor * gu_arena = ggml_new_tensor_3d(actx, gu_ty, gu_ne0, gu_ne1, n_slots);
        ggml_tensor * dn_arena = ggml_new_tensor_3d(actx, dn_ty, dn_ne0, dn_ne1, n_slots);
        ggml_backend_buffer_t abuf = ggml_backend_alloc_ctx_tensors(actx, arena_backend);
        printf("arene partagee: %.0f Mo (%d slots)\n", ggml_backend_buffer_get_size(abuf) / 1e6, n_slots);
        for (auto & kv : sim.layers) { kv.second.gu_arena = gu_arena; kv.second.dn_arena = dn_arena; }

        // ---- calibration SLRU depuis la trace brulee (1re moitie decode)
        std::map<int, std::vector<std::vector<int32_t>>> recs;
        int trace_predict = 0;
        FILE * f = fopen(trace_path, "rb");
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (line[0] == '#') { if (strstr(line, "n_predict=")) trace_predict = atoi(strstr(line, "n_predict=") + 10); continue; }
            char * p = line;
            strtol(p, &p, 10);
            int layer = (int) strtol(p, &p, 10);
            std::vector<int32_t> e;
            while (true) { char * q2; long v = strtol(p, &q2, 10); if (q2 == p) break; e.push_back((int32_t) v); p = q2; }
            recs[layer].push_back(e);
        }
        fclose(f);
        const int n_pin = 24, n_dyn = 8;
        for (auto & kv : recs) {
            auto seg = std::vector<std::vector<int32_t>>(kv.second.end() - std::min((int) kv.second.size(), trace_predict), kv.second.end());
            const int half = (int) seg.size() / 2;
            std::map<int32_t, int> freq;
            for (int i = 0; i < half; ++i) for (int32_t e : seg[i]) freq[e]++;
            std::vector<std::pair<int,int32_t>> order;
            for (auto & fe : freq) order.push_back({ -fe.second, fe.first });
            std::sort(order.begin(), order.end());
            Slru & sl = sim.slru[kv.first];
            sl.n_prob = n_dyn / 2; sl.n_prot = n_dyn - sl.n_prob;
            for (int s2 = 0; s2 < n_pin + n_dyn; ++s2) sl.free_slots.push_back(n_pin + n_dyn - 1 - s2);
            for (int i = 0; i < n_pin && i < (int) order.size(); ++i) {
                const int32_t e = order[i].second;
                sl.pinned.insert(e);
                sl.slot_of[e] = sl.free_slots.back(); sl.free_slots.pop_back();
            }
        }
        sim.pool.model_path = model_path;
        sim.pool.layers = &sim.layers;
        sim.pool.digests = &sim.digests;
        sim.pool.no_h2d = no_h2d;
        sim.pool.start(n_sha);
        printf("SLRU calibre (pin %d + dyn %d par couche), pool %d workers chunk 512 KiB%s\n",
               n_pin, n_dyn, n_sha, no_h2d ? ", H2D DESACTIVE (attribution)" : "");
        ggml_free(meta);
        gguf_free(g);
    }

    // ---- inference reelle
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    if (mode >= 1) { cp.cb_eval = eval_cb; cp.cb_eval_user_data = &sim; }
    llama_context * ctx = llama_init_from_model(model, cp);

    FILE * pf = fopen(prompt_path, "rb");
    fseek(pf, 0, SEEK_END); long pn = ftell(pf); fseek(pf, 0, SEEK_SET);
    std::string prompt((size_t) pn, '\0');
    fread(&prompt[0], 1, (size_t) pn, pf);
    fclose(pf);
    std::vector<llama_token> toks(prompt.size() + 16);
    const int n_prompt = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, true);
    toks.resize(n_prompt);
    fprintf(stderr, "prompt %d tokens\n", n_prompt);

    for (int i = 0; i < n_prompt; i += (int) cp.n_batch) {
        const int n = std::min((int) cp.n_batch, n_prompt - i);
        llama_batch b = llama_batch_get_one(toks.data() + i, n);
        llama_decode(ctx, b);
    }
    // decode greedy chronometre
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);
    llama_token cur = 0;
    for (int v = 1; v < n_vocab; ++v) if (logits[v] > logits[cur]) cur = v;
    const double d0 = now_ms();
    int n_gen = 0;
    for (; n_gen < n_predict; ++n_gen) {
        if (llama_vocab_is_eog(vocab, cur)) break;
        llama_batch b = llama_batch_get_one(&cur, 1);
        llama_decode(ctx, b);
        logits = llama_get_logits_ith(ctx, -1);
        llama_token best = 0;
        for (int v = 1; v < n_vocab; ++v) if (logits[v] > logits[best]) best = v;
        cur = best;
    }
    const double decode_ms = now_ms() - d0;

    printf("\n== P1a.2 mode %d ==\n", mode);
    printf("decode: %d tokens, %.1f ms/token, %.2f tok/s\n", n_gen, decode_ms / n_gen, n_gen * 1e3 / decode_ms);
    if (mode >= 1) printf("host_visibility: %.3f ms/token (%d tokens vus par le cb)\n", sim.t_vis / std::max(sim.n_tok, 1), sim.n_tok);
    if (mode >= 2) {
        // fermeture comptable: T_total = T_compute(?) + vis + pol + stall + OTHER
        const double per_tok = decode_ms / std::max(n_gen, 1);
        const double accounted = (sim.t_vis + sim.t_pol + sim.t_stall) / std::max(sim.n_tok, 1);
        printf("runtime_other:   %.2f ms/token (= %.2f total - %.2f compute_mode0_ref - %.2f vis+pol+stall) [compute ref a passer en externe]\n",
               per_tok - accounted - 22.5, per_tok, 22.5, accounted);
        printf("comptabilite:    total %.2f = vis+pol+stall %.2f + reste-avec-compute %.2f\n",
               per_tok, accounted, per_tok - accounted);
        printf("stall fetch:     %.2f ms/token | policy %.3f ms/token\n", sim.t_stall / std::max(sim.n_tok, 1), sim.t_pol / std::max(sim.n_tok, 1));
        printf("hit rate:        %.3f (%lld/%lld) | cold %.1f Mo/token | expert_ready %.2f ms (%lld miss)\n",
               (double) sim.n_hit / std::max<int64_t>(sim.n_acc, 1), (long long) sim.n_hit, (long long) sim.n_acc,
               sim.cold_bytes / 1e6 / std::max(sim.n_tok, 1),
               sim.pool.n_tasks ? sim.pool.ns_ready / 1e6 / sim.pool.n_tasks : 0.0, (long long) sim.pool.n_tasks);
        sim.pool.shutdown();
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
