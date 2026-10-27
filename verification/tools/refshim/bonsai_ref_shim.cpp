// Calibration shim: implements the benchmark ABI on top of llama.cpp.
// Not shipped to agents. Select the backend with BONSAI_SHIM_BACKEND=cpu|vulkan.
#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {

typedef struct bonsai_model bonsai_model;
typedef struct bonsai_context bonsai_context;

struct bonsai_model {
    llama_model * llama = nullptr;
    gguf_context * gguf = nullptr;
    ggml_context * gctx = nullptr;
    std::string path;
};

struct bonsai_context {
    bonsai_model * model = nullptr;
    llama_context * ctx = nullptr;
    int32_t n_ctx = 0;
    int32_t n_batch = 0;
};

static thread_local std::string g_error;

static void set_error(const std::string & s) { g_error = s; }

uint32_t bonsai_abi_version(void) { return 1; }

const char * bonsai_last_error(void) { return g_error.empty() ? nullptr : g_error.c_str(); }

int32_t bonsai_model_open(const char * gguf_path, bonsai_model ** out) {
    if (!gguf_path || !out) { set_error("invalid argument"); return -1; }
    *out = nullptr;
    const char * be = getenv("BONSAI_SHIM_BACKEND");
    const bool vulkan = be && strcmp(be, "vulkan") == 0;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = vulkan ? 99 : 0;
    llama_model * lm = llama_model_load_from_file(gguf_path, mp);
    if (!lm) { set_error("llama_model_load_from_file failed"); return -3; }
    ggml_context * gctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &gctx };
    gguf_context * gg = gguf_init_from_file(gguf_path, gp);
    if (!gg) { llama_model_free(lm); set_error("gguf_init_from_file failed"); return -3; }
    auto * m = new bonsai_model();
    m->llama = lm;
    m->gguf = gg;
    m->gctx = gctx;
    m->path = gguf_path;
    *out = m;
    return 0;
}

void bonsai_model_close(bonsai_model * m) {
    if (!m) return;
    if (m->gguf) gguf_free(m->gguf);
    if (m->gctx) ggml_free(m->gctx);
    if (m->llama) llama_model_free(m->llama);
    delete m;
}

int32_t bonsai_n_vocab(const bonsai_model * m) {
    if (!m) return -1;
    return llama_vocab_n_tokens(llama_model_get_vocab(m->llama));
}

int32_t bonsai_n_ctx_train(const bonsai_model * m) {
    return m ? llama_model_n_ctx_train(m->llama) : -1;
}

int32_t bonsai_tokenize(const bonsai_model * m, const char * utf8, int32_t add_bos,
                        int32_t * ids, int32_t capacity) {
    if (!m || !utf8) { set_error("invalid argument"); return -1; }
    const llama_vocab * v = llama_model_get_vocab(m->llama);
    int32_t n = llama_tokenize(v, utf8, int32_t(strlen(utf8)), nullptr, 0, add_bos != 0, true);
    if (n < 0) n = -n;
    if (!ids) return n;
    if (capacity < n) { set_error("capacity"); return -2; }
    return llama_tokenize(v, utf8, int32_t(strlen(utf8)), ids, capacity, add_bos != 0, true);
}

int32_t bonsai_tensor_count(const bonsai_model * m) {
    return m ? int32_t(gguf_get_n_tensors(m->gguf)) : -1;
}

int32_t bonsai_tensor_info(const bonsai_model * m, int32_t index, const char ** name,
                           int32_t * type, int32_t * rank, int64_t ne[4]) {
    if (!m || index < 0 || index >= gguf_get_n_tensors(m->gguf)) { set_error("bad index"); return -1; }
    if (name) *name = gguf_get_tensor_name(m->gguf, index);
    if (type) *type = int32_t(gguf_get_tensor_type(m->gguf, index));
    const int64_t * n = gguf_get_tensor_ne(m->gguf, index);
    int r = 1;
    for (int i = 1; i < 4; ++i) if (n[i] > 1) r = i + 1;
    if (rank) *rank = r;
    if (ne) for (int i = 0; i < 4; ++i) ne[i] = n[i];
    return 0;
}

int64_t bonsai_tensor_dequant(const bonsai_model * m, const char * name, float * out, int64_t cap) {
    if (!m || !name || !out) { set_error("invalid argument"); return -1; }
    ggml_tensor * t = ggml_get_tensor(m->gctx, name);
    if (!t) { set_error("tensor not found"); return -4; }
    const int64_t n = ggml_nelements(t);
    if (cap < n) { set_error("capacity"); return -2; }
    if (t->type == GGML_TYPE_F32) {
        memcpy(out, t->data, size_t(n) * sizeof(float));
        return n;
    }
    const ggml_type_traits * tr = ggml_get_type_traits(t->type);
    if (!tr || !tr->to_float) { set_error("no dequantizer"); return -4; }
    tr->to_float(t->data, out, n);
    return n;
}

static llama_context * make_ctx(bonsai_model * m, int32_t n_ctx, int32_t n_batch) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = uint32_t(n_ctx);
    cp.n_batch = uint32_t(n_batch);
    cp.n_ubatch = uint32_t(std::min(n_batch, 512));
    cp.n_threads = 6;
    cp.n_threads_batch = 6;
    return llama_init_from_model(m->llama, cp);
}

int32_t bonsai_context_create(bonsai_model * m, int32_t n_ctx, int32_t n_batch, bonsai_context ** out) {
    if (!m || !out || n_ctx <= 0) { set_error("invalid argument"); return -1; }
    *out = nullptr;
    auto * c = new bonsai_context();
    c->model = m;
    c->n_ctx = n_ctx;
    c->n_batch = n_batch > 0 ? std::min(n_batch, n_ctx) : std::min(2048, n_ctx);
    c->ctx = make_ctx(m, n_ctx, c->n_batch);
    if (!c->ctx) { delete c; set_error("llama_init_from_model failed"); return -6; }
    *out = c;
    return 0;
}

void bonsai_context_close(bonsai_context * c) {
    if (!c) return;
    if (c->ctx) llama_free(c->ctx);
    delete c;
}

int32_t bonsai_context_reset(bonsai_context * c) {
    if (!c) { set_error("invalid argument"); return -1; }
    llama_free(c->ctx);
    c->ctx = make_ctx(c->model, c->n_ctx, c->n_batch);
    return c->ctx ? 0 : -6;
}

static int eval_ids(llama_context * ctx, const int32_t * ids, int32_t n, int32_t n_batch) {
    int32_t off = 0;
    while (off < n) {
        const int32_t take = std::min(n - off, n_batch);
        std::vector<llama_token> chunk;
        chunk.resize(size_t(take));
        for (int32_t i = 0; i < take; ++i) chunk[size_t(i)] = llama_token(ids[off + i]);
        llama_batch b = llama_batch_get_one(chunk.data(), take);
        if (llama_decode(ctx, b) != 0) return -1;
        off += take;
    }
    return 0;
}

int32_t bonsai_eval(bonsai_context * c, const int32_t * tokens, int32_t n_tokens,
                    float * logits, int32_t logits_capacity) {
    if (!c || n_tokens <= 0) { set_error("invalid argument"); return -1; }
    if (eval_ids(c->ctx, tokens, n_tokens, c->n_batch) != 0) { set_error("decode failed"); return -6; }
    if (logits) {
        const int32_t n_vocab = bonsai_n_vocab(c->model);
        if (logits_capacity < n_vocab) { set_error("capacity"); return -2; }
        const float * src = llama_get_logits_ith(c->ctx, -1);
        if (!src) { set_error("no logits"); return -6; }
        memcpy(logits, src, size_t(n_vocab) * sizeof(float));
    }
    return 0;
}

int32_t bonsai_generate_greedy(bonsai_context * c, const int32_t * prompt, int32_t n_prompt,
                               int32_t n_predict, uint32_t flags, int32_t * out_ids) {
    if (!c || n_predict < 0 || !out_ids) { set_error("invalid argument"); return -1; }
    if (n_prompt > 0 && eval_ids(c->ctx, prompt, n_prompt, c->n_batch) != 0) {
        set_error("prompt decode failed"); return -6;
    }
    const int32_t n_vocab = bonsai_n_vocab(c->model);
    const llama_token eos = llama_vocab_eos(llama_model_get_vocab(c->model->llama));
    int32_t generated = 0;
    for (int32_t i = 0; i < n_predict; ++i) {
        const float * logits = llama_get_logits_ith(c->ctx, -1);
        if (!logits) { set_error("no logits"); return -6; }
        int32_t best = 0;
        float bv = logits[0];
        for (int32_t j = 1; j < n_vocab; ++j) if (logits[j] > bv) { bv = logits[j]; best = j; }
        out_ids[generated++] = best;
        if (!(flags & 1u) && best == eos) break;
        if (eval_ids(c->ctx, &best, 1, 1) != 0) { set_error("decode failed"); return -6; }
    }
    return generated;
}

} // extern "C"
