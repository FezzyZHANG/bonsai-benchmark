// Minimal ABI stub. This is only an integration example: every entry point
// returns an error. Replace it with the real engine.
const Model = opaque {};
const Context = opaque {};

export fn bonsai_abi_version() callconv(.c) u32 {
    return 1;
}

export fn bonsai_last_error() callconv(.c) ?[*:0]const u8 {
    return "stub example: engine not implemented";
}

export fn bonsai_model_open(gguf_path: ?[*:0]const u8, out_model: ?*?*Model) callconv(.c) i32 {
    _ = gguf_path;
    if (out_model) |p| p.* = null;
    return -4;
}

export fn bonsai_model_close(model: ?*Model) callconv(.c) void {
    _ = model;
}

export fn bonsai_n_vocab(model: ?*const Model) callconv(.c) i32 {
    _ = model;
    return -5;
}

export fn bonsai_n_ctx_train(model: ?*const Model) callconv(.c) i32 {
    _ = model;
    return -5;
}

export fn bonsai_tokenize(
    model: ?*const Model,
    utf8: ?[*:0]const u8,
    add_bos: i32,
    ids: ?[*]i32,
    capacity: i32,
) callconv(.c) i32 {
    _ = model;
    _ = utf8;
    _ = add_bos;
    _ = ids;
    _ = capacity;
    return -4;
}

export fn bonsai_tensor_count(model: ?*const Model) callconv(.c) i32 {
    _ = model;
    return -5;
}

export fn bonsai_tensor_info(
    model: ?*const Model,
    index: i32,
    name: ?*?[*:0]const u8,
    ggml_type: ?*i32,
    rank: ?*i32,
    ne: ?[*]i64,
) callconv(.c) i32 {
    _ = model;
    _ = index;
    _ = name;
    _ = ggml_type;
    _ = rank;
    _ = ne;
    return -4;
}

export fn bonsai_tensor_dequant(
    model: ?*const Model,
    name: ?[*:0]const u8,
    out: ?[*]f32,
    capacity: i64,
) callconv(.c) i64 {
    _ = model;
    _ = name;
    _ = out;
    _ = capacity;
    return -4;
}

export fn bonsai_context_create(
    model: ?*Model,
    n_ctx: i32,
    n_batch: i32,
    out_ctx: ?*?*Context,
) callconv(.c) i32 {
    _ = model;
    _ = n_ctx;
    _ = n_batch;
    if (out_ctx) |p| p.* = null;
    return -4;
}

export fn bonsai_context_close(ctx: ?*Context) callconv(.c) void {
    _ = ctx;
}

export fn bonsai_context_reset(ctx: ?*Context) callconv(.c) i32 {
    _ = ctx;
    return -5;
}

export fn bonsai_eval(
    ctx: ?*Context,
    tokens: ?[*]const i32,
    n_tokens: i32,
    logits: ?[*]f32,
    logits_capacity: i32,
) callconv(.c) i32 {
    _ = ctx;
    _ = tokens;
    _ = n_tokens;
    _ = logits;
    _ = logits_capacity;
    return -4;
}

export fn bonsai_generate_greedy(
    ctx: ?*Context,
    prompt: ?[*]const i32,
    n_prompt: i32,
    n_predict: i32,
    flags: u32,
    out_ids: ?[*]i32,
) callconv(.c) i32 {
    _ = ctx;
    _ = prompt;
    _ = n_prompt;
    _ = n_predict;
    _ = flags;
    _ = out_ids;
    return -4;
}
