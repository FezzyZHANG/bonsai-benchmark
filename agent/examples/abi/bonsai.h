#ifndef BONSAI_VK_ABI_H
#define BONSAI_VK_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BONSAI_ABI_VERSION 1

typedef struct bonsai_model   bonsai_model;
typedef struct bonsai_context bonsai_context;

/* Return codes. Functions return 0 on success unless documented otherwise. */
enum {
    BONSAI_OK                   =  0,
    BONSAI_ERR_INVALID_ARGUMENT = -1,
    BONSAI_ERR_CAPACITY         = -2,
    BONSAI_ERR_IO               = -3,
    BONSAI_ERR_UNSUPPORTED      = -4,
    BONSAI_ERR_STATE            = -5,
    BONSAI_ERR_INTERNAL         = -6
};

/* Generation flags. */
enum {
    BONSAI_GEN_IGNORE_EOS = 1u << 0
};

uint32_t    bonsai_abi_version(void);
const char *bonsai_last_error(void); /* thread-local; NULL if none */

/* Model lifecycle. gguf_path must point to the benchmark GGUF. */
int32_t  bonsai_model_open(const char *gguf_path, bonsai_model **out_model);
void     bonsai_model_close(bonsai_model *model);

int32_t  bonsai_n_vocab(const bonsai_model *model);
int32_t  bonsai_n_ctx_train(const bonsai_model *model);

/* Tokenizer. Returns the number of ids.
 * If ids == NULL, returns the required count without writing.
 * If ids != NULL and capacity is too small, returns BONSAI_ERR_CAPACITY.
 * add_bos != 0 prepends the model BOS token. No chat template is applied. */
int32_t  bonsai_tokenize(const bonsai_model *model, const char *utf8,
                         int32_t add_bos, int32_t *ids, int32_t capacity);

/* Introspection for staged correctness tests.
 * tensor_info returns 0; name is owned by the model and valid while it is open.
 * ne is in GGUF order (ne[0] is the contiguous/inner dimension). */
int32_t  bonsai_tensor_count(const bonsai_model *model);
int32_t  bonsai_tensor_info(const bonsai_model *model, int32_t index,
                            const char **name, int32_t *ggml_type,
                            int32_t *rank, int64_t ne[4]);

/* Dequantize one tensor to f32, row-major with the PyTorch/safetensors shape
 * (GGUF dimensions reversed; ne[0] varies fastest). Returns the element count,
 * or BONSAI_ERR_CAPACITY if capacity is too small. F32 tensors are copied. */
int64_t  bonsai_tensor_dequant(const bonsai_model *model, const char *name,
                               float *out, int64_t capacity);

/* Context lifecycle. n_batch is a processing hint. n_ctx >= 1. */
int32_t  bonsai_context_create(bonsai_model *model, int32_t n_ctx,
                               int32_t n_batch, bonsai_context **out_ctx);
void     bonsai_context_close(bonsai_context *ctx);
int32_t  bonsai_context_reset(bonsai_context *ctx);

/* Append n_tokens ids to the context and evaluate them. If logits != NULL,
 * writes n_vocab f32 logits for the last token. Returns 0 on success. */
int32_t  bonsai_eval(bonsai_context *ctx, const int32_t *tokens, int32_t n_tokens,
                     float *logits, int32_t logits_capacity);

/* Evaluate prompt, then greedily generate up to n_predict tokens (argmax).
 * Writes generated ids to out_ids (capacity >= n_predict) and returns the
 * generated count. Stops at EOS unless BONSAI_GEN_IGNORE_EOS is set.
 * The context state after the call is unspecified; use reset. */
int32_t  bonsai_generate_greedy(bonsai_context *ctx,
                                const int32_t *prompt, int32_t n_prompt,
                                int32_t n_predict, uint32_t flags,
                                int32_t *out_ids);

#ifdef __cplusplus
}
#endif

#endif
