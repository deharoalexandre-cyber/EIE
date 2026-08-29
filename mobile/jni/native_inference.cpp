// Moteur natif « EIE Mobile » — pont JNI réel (Soleau N.17, §3.4 / §3.5).
//
// Backend : fork llama.cpp-tq3 (turbo-tan) compilé arm64-v8a, libllama + libmtmd
// + libllama-common (liés en IMPORTED via CMake) + backends ggml chargés au
// runtime (CPU / OpenCL / Hexagon) par ggml_backend_load_all().
//
// Profil Baseline : KV cache q8_0/q4_0 (CPU). turbo3 (TQ3_0) reste gardé côté
// Kotlin (KvSelector) tant que les kernels NEON/NPU ne sont pas validés.
//
// Flux analyzeImage (calqué sur tools/mtmd/mtmd-cli.cpp) :
//   image -> mtmd_helper_bitmap_init_from_file
//   prompt + marqueur média -> mtmd_tokenize
//   mtmd_helper_eval_chunks (encode image + decode)
//   boucle de sampling -> texte (JSON §3.5)

#include <jni.h>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <android/log.h>
#include <sys/system_properties.h> // offload GPU réglable en live (debug.elyne.ngl)

#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <vector>
#include <cmath>
#include <chrono>
#include <atomic>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define LOG_TAG "EIEMobileNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// ===========================================================================
// TurboQuant TQ3_0 — kernel NEON (KV 3.5 bpw) + harnais d'auto-validation.
//
// Format (rétro-conçu de ggml-quants.c) : bloc de 32 valeurs = scale fp16 `d` +
// 12 octets d'indices 3-bit. Décode : dépack -> codebook Lloyd-Max 8 niveaux ->
// RHT inverse (WHT 32 pts + 1/√32 + signes) -> ×d.
//
// CORRECTION D'ABORD : on compare la déquantif NEON à une référence scalaire
// (copie EXACTE de l'upstream) sur des blocs aléatoires, sur device. On n'active
// le profil Turbo (KvSelector) qu'une fois `max_abs_err` ~ 0 confirmé en logcat.
// ===========================================================================
namespace tq3 {

constexpr int QK = 32;

// Codebook Lloyd-Max 8 niveaux (ggml-quants.c:2417).
const float CENTROIDS[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f,
};
// Masque de signes fixe du RHT (ggml-quants.c:2430).
const float SIGNS[32] = {
    +1.f,-1.f,+1.f,-1.f,+1.f,+1.f,-1.f,+1.f, -1.f,-1.f,+1.f,-1.f,+1.f,+1.f,-1.f,+1.f,
    -1.f,-1.f,+1.f,-1.f,+1.f,-1.f,-1.f,+1.f, -1.f,+1.f,+1.f,-1.f,+1.f,-1.f,-1.f,+1.f,
};

struct block { uint16_t d; uint8_t qs[12]; }; // == block_tq3_0

// Dépack 12 octets -> 32 indices 3-bit (entrelacement upstream, ggml-quants.c:3001).
inline void unpack(const uint8_t* qs, uint8_t* idx) {
    for (int g = 0; g < 4; g++) {
        const uint8_t* qp = qs + g * 3;
        uint8_t* o = idx + g * 8;
        o[0] =  qp[0]        & 7;
        o[1] = (qp[0] >> 3)  & 7;
        o[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
        o[3] = (qp[1] >> 1)  & 7;
        o[4] = (qp[1] >> 4)  & 7;
        o[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
        o[6] = (qp[2] >> 2)  & 7;
        o[7] = (qp[2] >> 5)  & 7;
    }
}

// Référence scalaire EXACTE (WHT 32 pts, étages 1->16) — vérité pour la validation.
void dequant_ref(const block& x, float* y) {
    uint8_t idx[32]; unpack(x.qs, idx);
    float w[32];
    for (int j = 0; j < 32; j++) w[j] = CENTROIDS[idx[j]];
    for (int step = 1; step < 32; step <<= 1)
        for (int i = 0; i < 32; i += step << 1)
            for (int j = i; j < i + step; j++) {
                float a = w[j], b = w[j + step];
                w[j] = a + b; w[j + step] = a - b;
            }
    const float d = ggml_fp16_to_fp32(x.d);
    const float nrm = 1.0f / sqrtf(32.0f);
    for (int j = 0; j < 32; j++) y[j] = w[j] * nrm * SIGNS[j] * d;
}

struct block_q8 { uint16_t d; int8_t qs[32]; }; // == block_q8_0

// vec_dot de référence (scalaire) : Σ_blocs déquant_tq3 · q8.qs · q8.d.
float vec_dot_ref(int nb, const block* x, const block_q8* y) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        float t[32]; dequant_ref(x[i], t);
        const float dy = ggml_fp16_to_fp32(y[i].d);
        for (int j = 0; j < 32; j++) sumf += t[j] * (float) y[i].qs[j] * dy;
    }
    return sumf;
}

#if defined(__ARM_NEON)
// WHT 32 pts en NEON (8× float32x4), étages 1->16 pour matcher le scalaire.
inline void wht32_neon(const float* in, float* out) {
    float32x4_t v[8];
    for (int k = 0; k < 8; k++) v[k] = vld1q_f32(in + 4 * k);
    // step=1 : papillon intra-vecteur lanes (0,1)(2,3)
    for (int k = 0; k < 8; k++) {
        float32x4_t r = vrev64q_f32(v[k]);
        float32x4_t s = vaddq_f32(v[k], r);
        float32x4_t d = vsubq_f32(v[k], r);
        v[k] = vtrnq_f32(s, d).val[0];               // [a+b, a-b, c+d, c-d]
    }
    // step=2 : lanes (0,2)(1,3)
    for (int k = 0; k < 8; k++) {
        float32x2_t lo = vget_low_f32(v[k]), hi = vget_high_f32(v[k]);
        v[k] = vcombine_f32(vadd_f32(lo, hi), vsub_f32(lo, hi));
    }
    // step=4 : vecteurs (k,k+1)
    for (int k = 0; k < 8; k += 2) {
        float32x4_t a = v[k], b = v[k + 1];
        v[k] = vaddq_f32(a, b); v[k + 1] = vsubq_f32(a, b);
    }
    // step=8 : vecteurs (k,k+2)
    for (int k : {0, 1, 4, 5}) {
        float32x4_t a = v[k], b = v[k + 2];
        v[k] = vaddq_f32(a, b); v[k + 2] = vsubq_f32(a, b);
    }
    // step=16 : vecteurs (k,k+4)
    for (int k = 0; k < 4; k++) {
        float32x4_t a = v[k], b = v[k + 4];
        v[k] = vaddq_f32(a, b); v[k + 4] = vsubq_f32(a, b);
    }
    for (int k = 0; k < 8; k++) vst1q_f32(out + 4 * k, v[k]);
}

void dequant_neon(const block& x, float* y) {
    uint8_t idx[32]; unpack(x.qs, idx);
    float rotated[32];
    for (int j = 0; j < 32; j++) rotated[j] = CENTROIDS[idx[j]];
    float w[32]; wht32_neon(rotated, w);
    const float d = ggml_fp16_to_fp32(x.d);
    const float nrm = 1.0f / sqrtf(32.0f);
    for (int j = 0; j < 32; j++) y[j] = w[j] * nrm * SIGNS[j] * d;
}

// vec_dot NEON : déquant NEON (validé bit-exact) + dot int8->float NEON.
// v1 « WHT par bloc » ; l'optim hoist-WHT (côté requête) viendra au niveau mul_mat.
float vec_dot_neon(int nb, const block* x, const block_q8* y) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        float t[32]; dequant_neon(x[i], t);
        const int8_t* qs = y[i].qs;
        const int8x16_t q0 = vld1q_s8(qs);
        const int8x16_t q1 = vld1q_s8(qs + 16);
        const int16x8_t q0l = vmovl_s8(vget_low_s8(q0));
        const int16x8_t q0h = vmovl_s8(vget_high_s8(q0));
        const int16x8_t q1l = vmovl_s8(vget_low_s8(q1));
        const int16x8_t q1h = vmovl_s8(vget_high_s8(q1));
        float32x4_t f[8];
        f[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0l)));
        f[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0l)));
        f[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0h)));
        f[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0h)));
        f[4] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1l)));
        f[5] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1l)));
        f[6] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1h)));
        f[7] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1h)));
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int k = 0; k < 8; k++) acc = vmlaq_f32(acc, vld1q_f32(t + 4 * k), f[k]);
        sumf += ggml_fp16_to_fp32(y[i].d) * vaddvq_f32(acc);
    }
    return sumf;
}
#endif // __ARM_NEON

// Compare NEON vs scalaire sur [nb] blocs aléatoires. Renvoie un JSON de diagnostic.
std::string selfcheck(int nb) {
#if defined(__ARM_NEON)
    unsigned s = 0x9E3779B9u;
    auto next = [&]() { s = s * 1664525u + 1013904223u; return s; };

    // (1) Déquantif : NEON vs scalaire — doit être bit-exact.
    float dq_max = 0.f;
    for (int b = 0; b < nb; b++) {
        block blk;
        blk.d = ggml_fp32_to_fp16(0.1f + 0.001f * (b % 100));
        for (int i = 0; i < 12; i++) blk.qs[i] = (uint8_t)(next() >> 13);
        float yref[32], yneon[32];
        dequant_ref(blk, yref);
        dequant_neon(blk, yneon);
        for (int j = 0; j < 32; j++) {
            float e = fabsf(yref[j] - yneon[j]);
            if (e > dq_max) dq_max = e;
        }
    }

    // (2) vec_dot : NEON vs scalaire — tolérance fp (ordre de sommation différent).
    const int NB = nb < 64 ? nb : 64;
    std::vector<block> xb(NB);
    std::vector<block_q8> yb(NB);
    for (int i = 0; i < NB; i++) {
        xb[i].d = ggml_fp32_to_fp16(0.05f + 0.002f * (i % 50));
        for (int k = 0; k < 12; k++) xb[i].qs[k] = (uint8_t)(next() >> 13);
        yb[i].d = ggml_fp32_to_fp16(0.03f + 0.001f * (i % 70));
        for (int k = 0; k < 32; k++) yb[i].qs[k] = (int8_t)(next() >> 24);
    }
    const float dref = vec_dot_ref(NB, xb.data(), yb.data());
    const float dneon = vec_dot_neon(NB, xb.data(), yb.data());
    const float dot_abs = fabsf(dref - dneon);
    const float dot_rel = dot_abs / (fabsf(dref) + 1e-9f);

    // (3) micro-bench kernel : NEON vs scalaire (vec_dot sur NB blocs).
    using clk = std::chrono::steady_clock;
    const int ITERS = 20000;
    volatile float sink = 0.f;
    const auto t0 = clk::now();
    for (int it = 0; it < ITERS; it++) sink += vec_dot_neon(NB, xb.data(), yb.data());
    const auto t1 = clk::now();
    for (int it = 0; it < ITERS; it++) sink += vec_dot_ref(NB, xb.data(), yb.data());
    const auto t2 = clk::now();
    const double neon_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
    const double ref_us  = std::chrono::duration<double, std::micro>(t2 - t1).count() / ITERS;

    char buf[384];
    snprintf(buf, sizeof(buf),
        "{\"tq3_selfcheck\":{\"neon\":true,\"dequant_max_abs_err\":%.3e,"
        "\"vecdot_abs_err\":%.3e,\"vecdot_rel_err\":%.3e,"
        "\"bench_blocks\":%d,\"neon_us\":%.3f,\"scalar_us\":%.3f,\"speedup\":%.2f}}",
        dq_max, dot_abs, dot_rel, NB, neon_us, ref_us, ref_us / (neon_us + 1e-9));
    return buf;
#else
    (void) nb;
    return "{\"tq3_selfcheck\":{\"neon\":false}}";
#endif
}

} // namespace tq3

constexpr int kMaxOutputTokens = 96; // explication courte (Verify) ; EOS coupe avant

bool g_runtime_init = false;

// Annulation d'inférence (« Stop »). Posé par cancel() depuis l'UI (autre thread),
// consulté par le callback d'abort de llama (interrompt llama_decode même en plein
// prefill) ET par les boucles de génération. Atomic = lecture/écriture inter-threads sûres.
std::atomic<bool> g_abort{false};
static bool abort_cb(void * /*data*/) { return g_abort.load(); }

// --- Accélération vision OpenCL/Adreno (Piste B, jalon Verify) -----------------
// On NE force PAS le GPU : on ne l'active que si un device OpenCL s'est bien
// enregistré au runtime (le loader SP-HAL de ggml-opencl peut échouer selon
// l'OEM). Sinon -> CPU strict, le verdict ONNX reste intact. Exposé via
// engineInfo() pour piloter le bench A/B.
bool        g_opencl_available  = false;
std::string g_opencl_device     = "";
std::string g_opencl_reason     = "not probed";
// Device de calcul EFFECTIF du dernier loadModel ("htp" | "gpu" | "cpu"). Exposé par
// engineInfo() pour que la couche Kotlin sache sur quoi elle tourne (ex. garde vision :
// le décode d'embeddings d'image ne passe pas sur le HTP). Indépendant des props.
std::string g_active_device     = "cpu";

void probe_opencl() {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            // Seul backend GPU compilé dans ce build = OpenCL (Adreno).
            const char *desc = ggml_backend_dev_description(dev);
            const char *name = ggml_backend_dev_name(dev);
            g_opencl_device    = desc ? desc : (name ? name : "GPU");
            g_opencl_available = true;
            g_opencl_reason    = "ok";
            LOGI("OpenCL device détecté: %s", g_opencl_device.c_str());
            return;
        }
    }
    g_opencl_reason = "aucun device OpenCL enregistré (SP-HAL/driver indisponible)";
    LOGI("OpenCL indisponible: %s", g_opencl_reason.c_str());
}

// Redirige les logs llama/ggml/mtmd vers logcat (sinon ils partent sur stderr
// et n'apparaissent pas — on a perdu du temps là-dessus).
void android_log_cb(ggml_log_level level, const char *text, void * /*ud*/) {
    int prio = ANDROID_LOG_INFO;
    if (level == GGML_LOG_LEVEL_ERROR) prio = ANDROID_LOG_ERROR;
    else if (level == GGML_LOG_LEVEL_WARN) prio = ANDROID_LOG_WARN;
    __android_log_print(prio, "EIE-llama", "%s", text);
}

// Initialise l'environnement natif AVANT tout appel llama/ggml/mtmd.
//  - ADSP_LIBRARY_PATH : indispensable pour que le CDSP trouve les skels HTP
//    (la dépendance libggml-hexagon.so est OBLIGATOIRE et s'auto-enregistre).
//  - GGML_HEXAGON_NDEV=0 : aucune session NPU en Baseline -> pas de FastRPC,
//    pas de device nul, exécution CPU propre. (Piste B remettra NDEV=1.)
void ensure_runtime(const char *lib_dir) {
    if (g_runtime_init) return;
    if (lib_dir && *lib_dir) {
        setenv("ADSP_LIBRARY_PATH", lib_dir, 1);
        setenv("LD_LIBRARY_PATH", lib_dir, 1);
    }
    // HTP/Hexagon : neutralisé par défaut (NDEV=0, jalon Verify), mais activable EN LIVE
    // pour le chantier NPU :  adb shell setprop debug.elyne.ndev 1  puis relance de l'app.
    // Nécessite libcdsprpc.so accessible (uses-native-library) + skel libggml-htp-vXX.so
    // dans le dossier des libs (ADSP_LIBRARY_PATH pointe dessus, cf. ci-dessus).
    char ndev_prop[PROP_VALUE_MAX] = {0};
    // DÉFAUT « optimisé flagship » : on TENTE le NPU Hexagon (ndev=1). Sur une puce sans
    // Hexagon (non-Qualcomm), le backend échoue proprement au chargement de libcdsprpc.so
    // -> repli CPU, sans crash. La prop debug.elyne.ndev reste prioritaire (0 = forcer CPU).
    const char *ndev = "1";
    if (__system_property_get("debug.elyne.ndev", ndev_prop) > 0 && ndev_prop[0]) ndev = ndev_prop;
    setenv("GGML_HEXAGON_NDEV", ndev, 1);
    if (ndev[0] != '0') LOGI("Hexagon NPU demande : GGML_HEXAGON_NDEV=%s", ndev);

    // DIAGNOSTIC/CONTOURNEMENT du crash VISION+NPU (SIGABRT ggml-hexagon flush_pending sur le
    // decode des embeddings d'image mtmd). Deux leviers du backend Hexagon, exposes en props :
    //  - debug.elyne.hexverbose N   -> GGML_HEXAGON_VERBOSE (trace les ops envoyees au DSP :
    //    la DERNIERE avant l'abort = l'op coupable).
    //  - debug.elyne.hexopfilter RE -> GGML_HEXAGON_OPFILTER (regex des ops que le backend NE
    //    revendique PAS : elles retombent sur CPU. Router l'op coupable ici = vision sans
    //    crash, le chat restant sur NPU pour tout le reste).
    char hexv_prop[PROP_VALUE_MAX] = {0};
    if (__system_property_get("debug.elyne.hexverbose", hexv_prop) > 0 && hexv_prop[0]) {
        setenv("GGML_HEXAGON_VERBOSE", hexv_prop, 1);
        LOGI("Hexagon VERBOSE=%s", hexv_prop);
    }
    char hexf_prop[PROP_VALUE_MAX] = {0};
    if (__system_property_get("debug.elyne.hexopfilter", hexf_prop) > 0 && hexf_prop[0]) {
        setenv("GGML_HEXAGON_OPFILTER", hexf_prop, 1);
        LOGI("Hexagon OPFILTER=%s (ces ops retombent sur CPU)", hexf_prop);
    }
    char hexb_prop[PROP_VALUE_MAX] = {0};
    if (__system_property_get("debug.elyne.hexopbatch", hexb_prop) > 0 && hexb_prop[0]) {
        setenv("GGML_HEXAGON_OPBATCH", hexb_prop, 1);
        LOGI("Hexagon OPBATCH=%s", hexb_prop);
    }
    // Callbacks de log posés AVANT load_all pour capturer dans logcat les
    // messages d'enregistrement des backends (dont ggml-opencl / loader SP-HAL).
    llama_log_set(android_log_cb, nullptr);
    mtmd_helper_log_set(android_log_cb, nullptr); // appelle aussi mtmd_log_set
    llama_backend_init();
    ggml_backend_load_all();
    // FILET (moteur multi-variantes GGML_BACKEND_DL) : load_all() scanne le dossier de
    // l'exécutable (/system/bin, vide sur Android) -> si RIEN ne s'est enregistré, on scanne
    // explicitement le dossier des libs de l'app (variantes libggml-cpu-android_*.so + OpenCL).
    // Ancien moteur (enregistrement statique via DT_NEEDED) : dev_count > 0 -> aucun rescan,
    // comportement historique intact.
    if (ggml_backend_dev_count() == 0 && lib_dir && *lib_dir) {
        ggml_backend_load_all_from_path(lib_dir);
    }
    probe_opencl(); // détecte si l'Adreno est joignable (SP-HAL)
    // Validation TQ3_0 NEON vs scalaire (correction d'abord) — résultat en logcat.
    LOGI("%s", tq3::selfcheck(512).c_str());
    g_runtime_init = true;
}

ggml_type map_kv_type(const std::string &id) {
    if (id == "f16")    return GGML_TYPE_F16;
    if (id == "q8_0")   return GGML_TYPE_Q8_0;
    if (id == "q4_0")   return GGML_TYPE_Q4_0;
    return GGML_TYPE_F16;
}

struct Session {
    common_init_result_ptr init;
    llama_model       *model  = nullptr;
    llama_context     *lctx   = nullptr;
    // SECOND contexte (même modèle, poids partagés) DÉDIÉ aux one-shots : consolidation
    // de profil, rêves, portes agenda/mail/web, panneau d'assistance. Sans lui, ces
    // générations décodaient dans `lctx` et écrasaient le KV du CHAT -> re-prefill complet
    // au tour suivant (bug n°1). Ici elles ont leur propre KV jetable ; le KV du chat
    // (`lctx`/`evaluated`) n'est JAMAIS touché. Coût ~140 Mo (KV 12 Mo + buffer calcul).
    llama_context     *lctx_os = nullptr;
    const llama_vocab *vocab  = nullptr;
    mtmd_context      *mctx   = nullptr;
    int                n_batch = 2048;
    int                n_threads = 4;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // KV reuse : tokens actuellement présents dans le cache KV (prompt + génération
    // du tour précédent). Permet de ne re-décoder que le suffixe nouveau (génération
    // incrémentale) au lieu de relire tout le fil à chaque tour. Un jeu par contexte.
    std::vector<llama_token> evaluated;      // chat (lctx)
    std::vector<llama_token> evaluated_os;   // one-shots (lctx_os)
};

std::string jstr(JNIEnv *env, jstring s) {
    if (!s) return {};
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    env->ReleaseStringUTFChars(s, c);
    return out;
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_elyne_trustid_inference_NativeInference_loadModel(
        JNIEnv *env, jobject /*thiz*/,
        jstring modelPath, jstring mmprojPath, jint contextSize,
        jstring kvType, jint threads, jboolean flashAttn) {

    std::string model_path        = jstr(env, modelPath);
    std::string mmproj_path       = jstr(env, mmprojPath);
    const std::string kv          = jstr(env, kvType);
    // Override EXPÉRIMENTAL du modèle (ex. test requant Q4_0 pour kernels Adreno/HTP) SANS
    // toucher au modèle par défaut :  adb shell setprop debug.elyne.model <chemin.gguf>
    // puis relance. Chaîne vide / prop absente = modèle normal.
    char mdl_prop[PROP_VALUE_MAX] = {0};
    if (__system_property_get("debug.elyne.model", mdl_prop) > 0 && mdl_prop[0]) {
        model_path = mdl_prop;
        LOGI("Override modele (debug.elyne.model) : %s", model_path.c_str());
    }
    // Override du projecteur vision (ex. mmproj Ministral avec le modele Ministral —
    // un mmproj etranger au modele echouerait/aberrerait) : debug.elyne.mmproj <chemin>.
    char mmp_prop[PROP_VALUE_MAX] = {0};
    if (__system_property_get("debug.elyne.mmproj", mmp_prop) > 0 && mmp_prop[0]) {
        mmproj_path = mmp_prop;
        LOGI("Override mmproj (debug.elyne.mmproj) : %s", mmproj_path.c_str());
    }

    ensure_runtime(nullptr); // au cas où initRuntime() n'aurait pas été appelé

    common_params params;
    params.model.path          = model_path;
    params.mmproj.path         = mmproj_path;
    params.n_ctx               = contextSize;
    // Baseline CPU par défaut (ngl=0). OFFLOAD GPU (Adreno/OpenCL) opt-in, réglable EN LIVE
    // sans rebuild :  adb shell setprop debug.elyne.ngl 16   puis relance de l'app.
    // Ne s'applique que si un device OpenCL s'est réellement enregistré (probe_opencl).
    // Prudence : modèle 3,4 Go + KV sur GPU vs ~4,5 Go libres -> commencer en offload PARTIEL.
    // NPU Hexagon réellement enregistré ? (Qualcomm : ndev=1 a fait apparaître un device HTP)
    bool htp_present = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (strstr(ggml_backend_dev_name(ggml_backend_dev_get(i)), "HTP")) { htp_present = true; break; }
    }
    // n_gpu_layers : prop prioritaire (debug) ; sinon DÉFAUT = offload complet quand un HTP
    // est là (« optimisé flagship »), CPU sinon (le décode CPU reste roi hors NPU).
    int ngl;
    char ngl_prop[PROP_VALUE_MAX] = {0};
    if (__system_property_get("debug.elyne.ngl", ngl_prop) > 0 && ngl_prop[0]) {
        ngl = atoi(ngl_prop);
    } else {
        ngl = htp_present ? 99 : 0;
    }
    const bool accel_ok = htp_present || g_opencl_available;
    params.n_gpu_layers        = (accel_ok && ngl > 0) ? ngl : 0;
    if (params.n_gpu_layers > 0) {
        LOGI("Offload actif : n_gpu_layers=%d (htp=%d opencl=%s)", params.n_gpu_layers,
             (int) htp_present, g_opencl_device.c_str());
    }
    params.cpuparams.n_threads = threads;
    params.flash_attn_type     = flashAttn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                           : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.cache_type_k        = map_kv_type(kv);
    params.cache_type_v        = map_kv_type(kv);
    if (params.n_gpu_layers > 0) {
        // OFFLOAD GPU : le KV quantifié (q8_0) + FA forcée ne sont pas exécutables par les
        // kernels OpenCL/Adreno -> le scheduler abort sur tenseur pré-alloué (observé :
        // ggml_backend_sched_backend_id_from_cur -> ggml_abort, 5 crashs identiques).
        // KV en f16 (coût mémoire mineur) + FA en AUTO (llama la coupe si non supportée).
        params.cache_type_k    = GGML_TYPE_F16;
        params.cache_type_v    = GGML_TYPE_F16;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        LOGI("Offload GPU : KV force en f16 + flash_attn AUTO (compat OpenCL)");
    }
    // SÉLECTEUR DE DEVICE (matrice silicium) : debug.elyne.dev = htp | gpu. Vide = auto.
    // Auto avec Adreno ET Hexagon enregistrés = llama SCINDE le modèle entre les deux ->
    // chaque token paie deux frontières de device (décode observé : 3 tok/s). Le filtre
    // force UN seul accélérateur. Liste terminée par nullptr (contrat llama_model_params).
    char dev_prop[PROP_VALUE_MAX] = {0};
    const bool dev_set = (__system_property_get("debug.elyne.dev", dev_prop) > 0 && dev_prop[0]);
    bool want_htp = false;
    bool do_filter = false;
    if (params.n_gpu_layers > 0 && dev_set) {
        want_htp = (strncmp(dev_prop, "htp", 3) == 0);   // override explicite (debug)
        do_filter = true;
    } else if (params.n_gpu_layers > 0 && htp_present) {
        want_htp = true;   // DÉFAUT flagship : NPU pur (évite le SPLIT Adreno+Hexagon = 3 tok/s)
        do_filter = true;
    }
    if (do_filter) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            const char *nm = ggml_backend_dev_name(d);
            const bool is_htp = (strstr(nm, "HTP") != nullptr);
            if (is_htp == want_htp) {
                params.devices.push_back(d);
                LOGI("Device choisi : %s", nm);
            }
        }
        if (!params.devices.empty()) params.devices.push_back(nullptr);
    }
    // Trace du device EFFECTIF (pour engineInfo -> garde vision côté Kotlin).
    g_active_device = (params.n_gpu_layers > 0)
                          ? (do_filter && want_htp && !params.devices.empty() ? "htp" : "gpu")
                          : "cpu";
    // ubatch réglable OPT-IN (debug.elyne.ubatch). Non posé = défaut llama (pas de surcoût vision).
    // (Le budget tokens image élevé + gros ubatch = 3 min sur NPU sans gain qualité -> désactivé par défaut.)
    {
        char ub[PROP_VALUE_MAX] = {0};
        if (__system_property_get("debug.elyne.ubatch", ub) > 0 && ub[0]) {
            int ubatch = atoi(ub);
            params.n_ubatch = ubatch;
            if (params.n_batch < ubatch) params.n_batch = ubatch; // contrat llama : n_batch >= n_ubatch
            LOGI("Vision : n_ubatch=%d n_batch=%d (prop)", params.n_ubatch, params.n_batch);
        }
    }
    params.warmup              = false;
    params.fit_params          = false; // INDISPENSABLE : sinon common_init échoue
                                        // (équivalent du -fit off du CLI)

    auto init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) {
        LOGE("common_init_from_params a échoué (model=%s)", model_path.c_str());
        return 0;
    }

    auto *s = new Session();
    s->init      = std::move(init);
    s->model     = s->init->model();
    s->lctx      = s->init->context();
    llama_set_abort_callback(s->lctx, abort_cb, nullptr); // « Stop » interrompt le decode
    s->vocab     = llama_model_get_vocab(s->model);
    s->n_batch   = params.n_batch;
    s->n_threads = threads;
    s->fa        = params.flash_attn_type;

    mtmd_context_params mp = mtmd_context_params_default();
    // Encode vision sur GPU OpenCL (Adreno) PAR DÉFAUT.
    // Historique : le fork de juin corrompait les embeddings sur Adreno (« couleurs seulement »).
    // Le rebase b9969 (12/07) a corrigé les kernels vision -> encode Adreno CORRECT et RAPIDE
    // (~6 s vs ~35-65 s CPU), stable à résolution standard (~266 tokens). Validé device : description
    // exacte + géométrie juste (montagnes en fond, plus d'eau hallucinée/inversion).
    // LIMITE b9969 : la vision GPU à HAUTE résolution (imgtokens élevé) plante (graph alloc / kernel FA
    // absent pour la tête vision) -> NE PAS pousser debug.elyne.imgtokens quand le GPU est actif.
    // Opt-out CPU (lent mais toute résolution) : debug.elyne.visgpu=0.
    bool vis_use_gpu = g_opencl_available;
    {
        char visg[PROP_VALUE_MAX] = {0};
        if (__system_property_get("debug.elyne.visgpu", visg) > 0 && visg[0] == '0') vis_use_gpu = false;
    }
    mp.use_gpu         = vis_use_gpu;
    mp.print_timings   = true; // bench A/B : timings d'encode dans logcat
    LOGI("mtmd use_gpu=%d (opencl=%s)", (int) mp.use_gpu, g_opencl_device.c_str());
    mp.n_threads       = threads;
    // FIX VISION GPU (12/07) : la flash-attention OpenCL n'a PAS de kernel pour la dimension de
    // tête de l'encodeur vision (SigLIP/MobileNet-V5). En AUTO/ENABLED, ggml_flash_attn_ext part
    // sur l'Adreno puis jette « map::at: key not found » dès que la résolution monte (>266 tokens)
    // -> « elle ne voit pas d'image ». L'attention STANDARD OpenCL marche pour n'importe quelle
    // tête. On désactive donc la FA pour l'encode vision UNIQUEMENT ; la FA du modèle texte
    // (s->fa = params.flash_attn_type) reste active sur NPU/GPU. C'est ce qui débloque l'encode
    // Adreno haute-résolution (le vrai livrable du rebase).
    mp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    mp.warmup          = false;
    // Plafond tokens image réglable OPT-IN (debug.elyne.imgtokens). Non posé = défaut métadonnées (~280).
    {
        char itk[PROP_VALUE_MAX] = {0};
        if (__system_property_get("debug.elyne.imgtokens", itk) > 0 && itk[0]) {
            int img_tokens = atoi(itk);
            mp.image_min_tokens = img_tokens;
            mp.image_max_tokens = img_tokens;
            LOGI("Vision : image_tokens=%d (prop)", img_tokens);
        }
    }
    s->mctx = mtmd_init_from_file(mmproj_path.c_str(), s->model, mp);
    if (!s->mctx) {
        LOGE("mtmd_init_from_file a échoué (mmproj=%s)", mmproj_path.c_str());
        delete s; // libère aussi model+context via init
        return 0;
    }

    // SECOND CONTEXTE (one-shots) — même modèle, mêmes réglages KV/FA/device que le chat
    // (il suit le placement device du modèle -> HTP si le chat y est). Échec = non fatal :
    // les one-shots retombent sur le contexte primaire (ancien comportement).
    {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = (uint32_t) contextSize;
        cp.n_batch         = params.n_batch;
        if (params.n_ubatch > 0) cp.n_ubatch = params.n_ubatch;
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
        cp.flash_attn_type = params.flash_attn_type;
        cp.type_k          = params.cache_type_k;
        cp.type_v          = params.cache_type_v;
        s->lctx_os = llama_init_from_model(s->model, cp);
        if (s->lctx_os) {
            llama_set_abort_callback(s->lctx_os, abort_cb, nullptr);
            LOGI("Second contexte (one-shots) prêt : le KV du chat est désormais inviolable.");
        } else {
            LOGE("Second contexte indisponible -> one-shots sur le contexte primaire (fallback).");
        }
    }

    LOGI("loadModel OK : kv=%s ctx=%d threads=%d fa=%d", kv.c_str(),
         contextSize, threads, (int) flashAttn);
    return reinterpret_cast<jlong>(s);
}

JNIEXPORT jstring JNICALL
Java_com_elyne_trustid_inference_NativeInference_analyzeImage(
        JNIEnv *env, jobject /*thiz*/,
        jlong handle, jstring imagePath, jstring prompt) {

    auto *s = reinterpret_cast<Session *>(handle);
    auto err = [&](const char *msg) -> jstring {
        std::string j = std::string("{\"synthetic_score\":0.5,\"quality_score\":0.0,")
                      + "\"signals\":[],\"explanation\":\"" + msg + "\"}";
        return env->NewStringUTF(j.c_str());
    };
    if (!s) return err("handle invalide");

    const std::string image_path = jstr(env, imagePath);
    const std::string user_prompt = jstr(env, prompt);

    // Contexte neuf à chaque analyse (chargement on-demand, mono-image).
    llama_memory_clear(llama_get_memory(s->lctx), true);
    s->evaluated.clear(); // invalide le KV chat (anti-contamination vision/chat)

    // 1) Image -> bitmap (décodage jpg/png par le helper).
    auto wrap = mtmd_helper_bitmap_init_from_file(s->mctx, image_path.c_str(), false);
    if (!wrap.bitmap) return err("image illisible");

    // 2) Prompt avec marqueur média (format de tour de CE modèle : <|turn>…<turn|>).
    //    NB: parse_special=true => les tokens spéciaux sont interprétés.
    std::string text = std::string("<|turn>user\n")
                     + mtmd_default_marker() + "\n"
                     + user_prompt
                     + "<turn|>\n<|turn>model\n";
    mtmd_input_text it;
    it.text          = text.c_str();
    it.add_special   = true;
    it.parse_special = true;

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    const mtmd_bitmap *bitmaps[1] = { wrap.bitmap };
    int32_t tok = mtmd_tokenize(s->mctx, chunks, &it, bitmaps, 1);
    mtmd_bitmap_free(wrap.bitmap); // données copiées par tokenize
    if (tok != 0) {
        mtmd_input_chunks_free(chunks);
        return err("tokenisation impossible");
    }

    // 3) Eval des chunks (encode image + decode).
    llama_pos n_past = 0, new_n_past = 0;
    int32_t ev = mtmd_helper_eval_chunks(s->mctx, s->lctx, chunks, n_past,
                                         /*seq_id*/ 0, s->n_batch,
                                         /*logits_last*/ true, &new_n_past);
    mtmd_input_chunks_free(chunks);
    if (ev != 0) return err("evaluation impossible");
    n_past = new_n_past;

    // 4) Génération (sampler frais pour un état propre par analyse).
    common_params_sampling sp;
    sp.temp = 0.2f; // réponse stable/structurée (JSON)
    common_sampler *smpl = common_sampler_init(s->model, sp);
    if (!smpl) return err("sampler init échoué");

    std::string out;
    llama_batch batch = llama_batch_init(1, 0, 1);
    for (int i = 0; i < kMaxOutputTokens; i++) {
        llama_token id = common_sampler_sample(smpl, s->lctx, -1);
        common_sampler_accept(smpl, id, true);
        if (llama_vocab_is_eog(s->vocab, id)) break;
        out += common_token_to_piece(s->lctx, id);
        common_batch_clear(batch);
        common_batch_add(batch, id, n_past++, {0}, true);
        if (llama_decode(s->lctx, batch)) break;
    }
    llama_batch_free(batch);
    common_sampler_free(smpl);

    // Diagnostic : sortie brute du modèle (tronquée par logcat).
    LOGI("analyzeImage RAW OUTPUT >>>\n%s\n<<< END", out.c_str());

    if (out.empty()) return err("generation vide");
    return env->NewStringUTF(out.c_str());
}

// String describeImage(long handle, String imagePath, String prompt, int maxTokens)
//
// Vision pour le CHAT d'Elyne : décrit / répond sur une image en texte LIBRE (pas le
// JSON normalisé de Verify). Même flux mtmd que analyzeImage, mais température
// conversationnelle (0.7) et budget de sortie plus large. Invalide la session chat
// (le KV vision n'est pas le KV texte) → le tour texte suivant refait un prefill.
JNIEXPORT jstring JNICALL
Java_com_elyne_trustid_inference_NativeInference_describeImage(
        JNIEnv *env, jobject /*thiz*/,
        jlong handle, jstring imagePath, jstring prompt, jint maxTokens) {

    auto *s = reinterpret_cast<Session *>(handle);
    if (!s) return env->NewStringUTF("");
    g_abort = false;

    const std::string image_path  = jstr(env, imagePath);
    const std::string user_prompt = jstr(env, prompt);

    llama_memory_clear(llama_get_memory(s->lctx), true);
    s->evaluated.clear();

    auto wrap = mtmd_helper_bitmap_init_from_file(s->mctx, image_path.c_str(), false);
    if (!wrap.bitmap) return env->NewStringUTF("");

    std::string text = std::string("<|turn>user\n")
                     + mtmd_default_marker() + "\n"
                     + user_prompt
                     + "<turn|>\n<|turn>model\n";
    mtmd_input_text it;
    it.text          = text.c_str();
    it.add_special   = true;
    it.parse_special = true;

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    const mtmd_bitmap *bitmaps[1] = { wrap.bitmap };
    int32_t tok = mtmd_tokenize(s->mctx, chunks, &it, bitmaps, 1);
    mtmd_bitmap_free(wrap.bitmap);
    if (tok != 0) { mtmd_input_chunks_free(chunks); return env->NewStringUTF(""); }

    // MESURE DÉCISIVE (Fable) : n_tokens du CHUNK IMAGE = équivalent device du prompt_tokens desktop.
    // ~256-280 alors que debug.elyne.imgtokens=1120 est posé -> le budget N'ATTEINT PAS l'encodeur
    // (param accepté mais ignoré hors chemin CLI -> rebase du fork). ~1120 = budget réellement appliqué.
    for (size_t ci = 0, nc = mtmd_input_chunks_size(chunks); ci < nc; ci++) {
        const mtmd_input_chunk *ch = mtmd_input_chunks_get(chunks, ci);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE)
            LOGI("describeImage: CHUNK IMAGE = %zu tokens (budget effectif encodeur)",
                 mtmd_input_chunk_get_n_tokens(ch));
    }

    llama_pos n_past = 0, new_n_past = 0;
    int32_t ev = mtmd_helper_eval_chunks(s->mctx, s->lctx, chunks, n_past,
                                         /*seq_id*/ 0, s->n_batch,
                                         /*logits_last*/ true, &new_n_past);
    mtmd_input_chunks_free(chunks);
    if (ev != 0) return env->NewStringUTF("");
    n_past = new_n_past;
    LOGI("describeImage: %d tokens (image+prompt) evalues", (int) new_n_past);

    common_params_sampling sp;
    sp.temp = 0.3f; // description LITTÉRALE : temp basse = moins de broderie/hallucination
    common_sampler *smpl = common_sampler_init(s->model, sp);
    if (!smpl) return env->NewStringUTF("");

    const int budget = maxTokens > 0 ? maxTokens : 256;
    std::string out;
    llama_batch batch = llama_batch_init(1, 0, 1);
    for (int i = 0; i < budget; i++) {
        if (g_abort.load()) break; // « Stop »
        llama_token id = common_sampler_sample(smpl, s->lctx, -1);
        common_sampler_accept(smpl, id, true);
        if (llama_vocab_is_eog(s->vocab, id)) break;
        out += common_token_to_piece(s->lctx, id);
        common_batch_clear(batch);
        common_batch_add(batch, id, n_past++, {0}, true);
        if (llama_decode(s->lctx, batch)) break;
    }
    llama_batch_free(batch);
    common_sampler_free(smpl);

    // IMPORTANT : on a vidé evaluated en début d'appel, mais le KV contient désormais
    // l'image + la génération. On le REVIDE pour que le prochain tour (texte OU vision)
    // reparte d'un état cohérent — sinon le texte décode par-dessus des restes vision et
    // produit une sortie vide/corrompue (« elle ne répond plus après une photo »).
    llama_memory_clear(llama_get_memory(s->lctx), true);
    s->evaluated.clear();

    LOGI("describeImage RAW OUTPUT (%zu chars) >>>\n%s\n<<< END", out.size(), out.c_str());
    return env->NewStringUTF(out.c_str());
}

JNIEXPORT void JNICALL
Java_com_elyne_trustid_inference_NativeInference_unloadModel(
        JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (!s) return;
    if (s->mctx) mtmd_free(s->mctx);
    if (s->lctx_os) llama_free(s->lctx_os); // 2e contexte créé à la main -> libéré à la main
    delete s; // common_init_result libère model + contexte primaire
    LOGI("unloadModel OK");
}

// void initRuntime(String nativeLibDir) — à appeler AVANT loadModel.
JNIEXPORT void JNICALL
Java_com_elyne_trustid_inference_NativeInference_initRuntime(
        JNIEnv *env, jobject /*thiz*/, jstring nativeLibDir) {
    const std::string dir = jstr(env, nativeLibDir);
    ensure_runtime(dir.c_str());
    LOGI("initRuntime OK (lib_dir=%s)", dir.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_elyne_trustid_inference_NativeInference_engineInfo(
        JNIEnv *env, jobject /*thiz*/) {
    // S'assure que le probe OpenCL a tourné même si loadModel n'a pas encore
    // été appelé (engineInfo peut être interrogé tôt par le Kotlin).
    ensure_runtime(nullptr);

    // Capacités réelles compilées. turboquant=true (TQ3_0 présent dans le fork),
    // mais KvSelector garde turbo3 non-mobile tant que NEON/NPU non validés.
    // vision_backend / opencl_* : pilotent le bench A/B du jalon OpenCL.
    auto esc = [](const std::string &s) {
        std::string o; o.reserve(s.size());
        for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
        return o;
    };
    std::string json =
            std::string("{\"backend\":\"llama.cpp-tq3\",\"turboquant\":true,\"neon\":true,")
          + "\"kv_types\":[\"f16\",\"q8_0\",\"q4_0\",\"turbo3\"],"
          + "\"vision_backend\":\"" + (g_opencl_available ? "opencl" : "cpu") + "\","
          + "\"opencl_available\":" + (g_opencl_available ? "true" : "false") + ","
          + "\"opencl_device\":\"" + esc(g_opencl_device) + "\","
          + "\"opencl_reason\":\"" + esc(g_opencl_reason) + "\","
          + "\"active_device\":\"" + esc(g_active_device) + "\"}";
    return env->NewStringUTF(json.c_str());
}

// String generate(long handle, String prompt, int maxTokens)
//
// Chat texte d'Elyne Mobile. Même backend que analyzeImage, mais SANS image :
// le prompt fourni est déjà le contexte de continuité complet assemblé par
// l'orchestrateur Elyne (identité + RAG + fil). Modèle gardé chaud (handle
// réutilisé) ; le KV est reconstruit à partir du prompt à chaque tour (sans état
// persistant entre tours en tranche 0).
JNIEXPORT jstring JNICALL
Java_com_elyne_trustid_inference_NativeInference_generate(
        JNIEnv *env, jobject /*thiz*/,
        jlong handle, jstring prompt, jint maxTokens, jboolean oneShot) {

    auto *s = reinterpret_cast<Session *>(handle);
    if (!s) return env->NewStringUTF("");
    g_abort = false; // nouvelle inférence : on repart d'un état non annulé

    // ROUTAGE DU CONTEXTE : un one-shot (consolidation/rêve/porte/assist) décode dans le
    // SECOND contexte -> il ne touche jamais le KV du chat. Chacun garde son propre suivi
    // `evaluated`. Si le second contexte n'a pas pu être créé, repli sur le primaire.
    const bool one_shot = (oneShot == JNI_TRUE) && (s->lctx_os != nullptr);
    llama_context *ctx = one_shot ? s->lctx_os : s->lctx;
    std::vector<llama_token> &ev = one_shot ? s->evaluated_os : s->evaluated;

    const std::string user_prompt = jstr(env, prompt);

    // KV REUSE / génération incrémentale : on NE vide PLUS le KV à chaque tour. On
    // compare le nouveau prompt aux tokens déjà évalués, on garde le préfixe commun
    // dans le cache, et on ne décode QUE le suffixe nouveau. Coût O(nouveau message)
    // au lieu de O(fil entier). Le prompt est formaté en tours par l'orchestrateur
    // (préfixe stable = système + fil ; volatile en fin) — condition pour que ça morde.
    const std::string text = user_prompt;
    std::vector<llama_token> toks =
        common_tokenize(ctx, text, /*add_special*/ true, /*parse_special*/ true);
    if (toks.empty()) return env->NewStringUTF("");

    llama_memory_t mem = llama_get_memory(ctx);

    // 1) Plus long préfixe commun avec le KV courant.
    int n_keep = 0;
    const int max_common = (int) std::min(toks.size(), ev.size());
    while (n_keep < max_common && toks[n_keep] == ev[n_keep]) n_keep++;
    if (n_keep == (int) toks.size()) n_keep = (int) toks.size() - 1; // re-décoder >=1 pour logits
    if (n_keep < 0) n_keep = 0;

    // 2) Retirer du KV tout ce qui dépasse le préfixe commun (stale).
    if (n_keep < (int) ev.size()) {
        if (!llama_memory_seq_rm(mem, 0, n_keep, -1)) {
            llama_memory_clear(mem, true); // suppression partielle refusée -> reset complet
            n_keep = 0;
        }
    } else if (n_keep == 0) {
        // Préfill complet : on garantit un KV vide. Défense si un appel précédent (ex.
        // vision) a laissé des restes dans le KV sans les tracer dans evaluated.
        llama_memory_clear(mem, true);
    }

    const auto t0 = std::chrono::steady_clock::now();

    // 3) Décoder UNIQUEMENT le suffixe nouveau toks[n_keep:].
    llama_batch batch = llama_batch_init(s->n_batch, 0, 1);
    llama_pos n_past = n_keep;
    for (size_t i = (size_t) n_keep; i < toks.size();) {
        common_batch_clear(batch);
        const size_t n = std::min((size_t) s->n_batch, toks.size() - i);
        for (size_t j = 0; j < n; j++) {
            const bool is_last = (i + j == toks.size() - 1);
            common_batch_add(batch, toks[i + j], n_past++, {0}, is_last);
        }
        if (llama_decode(ctx, batch)) {
            llama_batch_free(batch);
            ev.clear(); // état KV incertain -> full prefill au prochain tour
            return env->NewStringUTF("");
        }
        i += n;
    }
    const int prefill_tokens = (int) toks.size() - n_keep;
    const auto t1 = std::chrono::steady_clock::now();

    // 4) Sampling conversationnel. Réglable EN LIVE (recommandations Google pour les
    //    checkpoints QAT : temp 1.0, top-p 0.95, top-k 64) :
    //    adb shell setprop debug.elyne.temp 1.0 ; debug.elyne.topp 0.95 ; debug.elyne.topk 64
    common_params_sampling sp;
    sp.temp = 0.7f;
    {
        char sprop[PROP_VALUE_MAX];
        sprop[0] = 0;
        if (__system_property_get("debug.elyne.temp", sprop) > 0 && sprop[0]) sp.temp = (float) atof(sprop);
        sprop[0] = 0;
        if (__system_property_get("debug.elyne.topp", sprop) > 0 && sprop[0]) sp.top_p = (float) atof(sprop);
        sprop[0] = 0;
        if (__system_property_get("debug.elyne.topk", sprop) > 0 && sprop[0]) sp.top_k = atoi(sprop);
    }
    common_sampler *smpl = common_sampler_init(s->model, sp);
    if (!smpl) { llama_batch_free(batch); return env->NewStringUTF(""); }

    const int budget = maxTokens > 0 ? maxTokens : 256;
    std::string out;
    std::vector<llama_token> gen;
    for (int k = 0; k < budget; k++) {
        if (g_abort.load()) break; // « Stop » demandé -> on rend la sortie partielle
        const llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        if (llama_vocab_is_eog(s->vocab, id)) break;
        out += common_token_to_piece(ctx, id);
        gen.push_back(id);
        common_batch_clear(batch);
        common_batch_add(batch, id, n_past++, {0}, true);
        if (llama_decode(ctx, batch)) break;
    }
    common_sampler_free(smpl);
    llama_batch_free(batch);
    const auto t2 = std::chrono::steady_clock::now();

    // 5) Mémoriser l'état KV pour le prochain tour (prompt + génération).
    ev = toks;
    ev.insert(ev.end(), gen.begin(), gen.end());

    const double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decode_ms  = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double dtoks = gen.empty() ? 1.0 : (double) gen.size();
    LOGI("generate[%s]: reuse=%d prefill=%d tok (%.0f ms) decode=%zu tok (%.0f ms, %.1f tok/s) total=%.0f ms",
         one_shot ? "os" : "chat",
         n_keep, prefill_tokens, prefill_ms, gen.size(), decode_ms,
         1000.0 * dtoks / (decode_ms + 1e-6), prefill_ms + decode_ms);
    return env->NewStringUTF(out.c_str());
}

// ---------------------------------------------------------------------------
// Forwarders package Elyne Mobile (com.elyne.mobile.inference.NativeInference).
//
// Un seul .so sert deux produits : Elyne Verify (symboles com_elyne_trustid,
// ci-dessus) ET Elyne Mobile (com_elyne_mobile, ci-dessous). Les noms de
// symboles JNI dépendant du package Java de la classe appelante, on expose des
// points d'entrée parallèles qui réutilisent l'implémentation trustid. Zéro
// duplication de logique.
// ---------------------------------------------------------------------------

JNIEXPORT void JNICALL
Java_com_elyne_mobile_inference_NativeInference_initRuntime(
        JNIEnv *env, jobject thiz, jstring nativeLibDir) {
    Java_com_elyne_trustid_inference_NativeInference_initRuntime(env, thiz, nativeLibDir);
}

JNIEXPORT jlong JNICALL
Java_com_elyne_mobile_inference_NativeInference_loadModel(
        JNIEnv *env, jobject thiz, jstring modelPath, jstring mmprojPath,
        jint contextSize, jstring kvType, jint threads, jboolean flashAttn) {
    return Java_com_elyne_trustid_inference_NativeInference_loadModel(
        env, thiz, modelPath, mmprojPath, contextSize, kvType, threads, flashAttn);
}

JNIEXPORT jstring JNICALL
Java_com_elyne_mobile_inference_NativeInference_analyzeImage(
        JNIEnv *env, jobject thiz, jlong handle, jstring imagePath, jstring prompt) {
    return Java_com_elyne_trustid_inference_NativeInference_analyzeImage(
        env, thiz, handle, imagePath, prompt);
}

JNIEXPORT jstring JNICALL
Java_com_elyne_mobile_inference_NativeInference_generate(
        JNIEnv *env, jobject thiz, jlong handle, jstring prompt, jint maxTokens, jboolean oneShot) {
    return Java_com_elyne_trustid_inference_NativeInference_generate(
        env, thiz, handle, prompt, maxTokens, oneShot);
}

JNIEXPORT jstring JNICALL
Java_com_elyne_mobile_inference_NativeInference_describeImage(
        JNIEnv *env, jobject thiz, jlong handle, jstring imagePath, jstring prompt, jint maxTokens) {
    return Java_com_elyne_trustid_inference_NativeInference_describeImage(
        env, thiz, handle, imagePath, prompt, maxTokens);
}

JNIEXPORT void JNICALL
Java_com_elyne_mobile_inference_NativeInference_unloadModel(
        JNIEnv *env, jobject thiz, jlong handle) {
    Java_com_elyne_trustid_inference_NativeInference_unloadModel(env, thiz, handle);
}

JNIEXPORT jstring JNICALL
Java_com_elyne_mobile_inference_NativeInference_engineInfo(
        JNIEnv *env, jobject thiz) {
    return Java_com_elyne_trustid_inference_NativeInference_engineInfo(env, thiz);
}

// « Stop » : pose le drapeau d'annulation (thread UI). La génération en cours s'arrête
// au prochain token / decode et rend sa sortie partielle. Pas de verrou : atomic.
JNIEXPORT void JNICALL
Java_com_elyne_trustid_inference_NativeInference_cancel(JNIEnv *, jobject) {
    g_abort = true;
}

JNIEXPORT void JNICALL
Java_com_elyne_mobile_inference_NativeInference_cancel(JNIEnv *, jobject) {
    g_abort = true;
}

// ---- Embeddings (RAG mémoire longue) ----------------------------------------
// Modèle d'embedding GGUF séparé (ex. bge-m3), mode pooling. Contexte distinct du chat,
// résident léger. embed() renvoie un vecteur L2-normalisé (prêt pour similarité cosinus).

JNIEXPORT jlong JNICALL
Java_com_elyne_trustid_inference_NativeInference_loadEmbedder(
        JNIEnv *env, jobject /*thiz*/, jstring modelPath, jint threads) {
    const std::string model_path = jstr(env, modelPath);
    ensure_runtime(nullptr);

    common_params params;
    params.model.path          = model_path;
    params.n_ctx               = 512;  // entrées de mémoire courtes
    params.n_batch             = 512;
    params.n_ubatch            = 512;  // encodeur non causal : ubatch >= longueur séquence
    params.n_gpu_layers        = 0;
    params.cpuparams.n_threads = threads;
    params.embedding           = true;
    params.pooling_type        = LLAMA_POOLING_TYPE_UNSPECIFIED; // défaut du gguf (CLS pour bge)
    params.warmup              = false;
    params.fit_params          = false;

    auto init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) {
        LOGE("loadEmbedder a échoué (model=%s)", model_path.c_str());
        return 0;
    }
    auto *s = new Session();
    s->init      = std::move(init);
    s->model     = s->init->model();
    s->lctx      = s->init->context();
    s->vocab     = llama_model_get_vocab(s->model);
    s->n_batch   = params.n_batch;
    s->n_threads = threads;
    LOGI("loadEmbedder OK : n_embd=%d", llama_model_n_embd(s->model));
    return reinterpret_cast<jlong>(s);
}

JNIEXPORT jfloatArray JNICALL
Java_com_elyne_trustid_inference_NativeInference_embed(
        JNIEnv *env, jobject /*thiz*/, jlong handle, jstring text) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (!s) return nullptr;
    const std::string t = jstr(env, text);
    if (t.empty()) return nullptr;

    std::vector<llama_token> toks = common_tokenize(s->lctx, t, true, true);
    if (toks.empty()) return nullptr;
    const int n_ctx = (int) llama_n_ctx(s->lctx);
    if ((int) toks.size() > n_ctx) toks.resize(n_ctx);

    llama_memory_clear(llama_get_memory(s->lctx), true);
    llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        common_batch_add(batch, toks[i], (llama_pos) i, {0}, true);
    }
    if (llama_decode(s->lctx, batch) != 0) {
        llama_batch_free(batch);
        return nullptr;
    }
    const int n_embd = llama_model_n_embd(s->model);
    const float *emb = llama_get_embeddings_seq(s->lctx, 0);
    if (!emb) { llama_batch_free(batch); return nullptr; }

    std::vector<float> out(n_embd);
    double norm = 0.0;
    for (int i = 0; i < n_embd; i++) norm += (double) emb[i] * emb[i];
    norm = std::sqrt(norm);
    if (norm < 1e-12) norm = 1.0;
    for (int i = 0; i < n_embd; i++) out[i] = (float) (emb[i] / norm);

    llama_batch_free(batch);
    jfloatArray arr = env->NewFloatArray(n_embd);
    env->SetFloatArrayRegion(arr, 0, n_embd, out.data());
    return arr;
}

JNIEXPORT jlong JNICALL
Java_com_elyne_mobile_inference_NativeInference_loadEmbedder(
        JNIEnv *env, jobject thiz, jstring modelPath, jint threads) {
    return Java_com_elyne_trustid_inference_NativeInference_loadEmbedder(env, thiz, modelPath, threads);
}

JNIEXPORT jfloatArray JNICALL
Java_com_elyne_mobile_inference_NativeInference_embed(
        JNIEnv *env, jobject thiz, jlong handle, jstring text) {
    return Java_com_elyne_trustid_inference_NativeInference_embed(env, thiz, handle, text);
}

} // extern "C"
