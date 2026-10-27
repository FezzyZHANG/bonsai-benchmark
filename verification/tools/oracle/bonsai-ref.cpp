// Hidden line-protocol reference runner for the benchmark.
// Reads one command per line on stdin, writes one reply line per command.
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct model_deleter {
    void operator()(llama_model * m) const { if (m) llama_model_free(m); }
};
struct context_deleter {
    void operator()(llama_context * c) const { if (c) llama_free(c); }
};

std::string g_model_path;
std::unique_ptr<llama_model, model_deleter> g_model_cpu;
std::unique_ptr<llama_model, model_deleter> g_model_gpu;
std::string g_last_error;

struct ctx_entry {
    std::unique_ptr<llama_context, context_deleter> ctx;
    int32_t n_ctx = 0;
    bool vulkan = false;
};
std::map<int, ctx_entry> g_contexts;

void set_error(const std::string & msg) { g_last_error = msg; }

std::string hex_decode(const std::string & hex) {
    if (hex.size() % 2 != 0) throw std::runtime_error("odd hex length");
    std::string out;
    out.reserve(hex.size() / 2);
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw std::runtime_error("bad hex");
    };
    for (size_t i = 0; i < hex.size(); i += 2) {
        out.push_back(char((nib(hex[i]) << 4) | nib(hex[i + 1])));
    }
    return out;
}

llama_model * model_for(bool vulkan) {
    auto & slot = vulkan ? g_model_gpu : g_model_cpu;
    if (slot) return slot.get();
    if (g_model_path.empty()) throw std::runtime_error("load a model first");
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = vulkan ? 99 : 0;
    llama_model * m = llama_model_load_from_file(g_model_path.c_str(), mp);
    if (!m) throw std::runtime_error("failed to load model");
    slot.reset(m);
    return m;
}

llama_context * make_context(int n_ctx, bool vulkan) {
    llama_model * model = model_for(vulkan);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = uint32_t(n_ctx);
    cp.n_batch = uint32_t(std::min<int64_t>(n_ctx, 2048));
    cp.n_ubatch = uint32_t(std::min<int64_t>(n_ctx, 512));
    cp.n_threads = 6;
    cp.n_threads_batch = 6;
    llama_context * c = llama_init_from_model(model, cp);
    if (!c) throw std::runtime_error("failed to create context");
    return c;
}

void eval_tokens(llama_context * ctx, const std::vector<llama_token> & tokens) {
    const int32_t n_batch = std::max<int32_t>(1, int32_t(llama_n_batch(ctx)));
    size_t off = 0;
    while (off < tokens.size()) {
        const size_t n = std::min<size_t>(tokens.size() - off, size_t(n_batch));
        std::vector<llama_token> chunk(tokens.begin() + off, tokens.begin() + off + n);
        llama_batch batch = llama_batch_get_one(chunk.data(), int32_t(n));
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
        off += n;
    }
}

int32_t argmax(const float * logits, int32_t n) {
    int32_t best = 0;
    float best_v = logits[0];
    for (int32_t i = 1; i < n; ++i) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return best;
}

} // namespace

int main() {
    llama_backend_init();
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        try {
            if (cmd == "quit" || cmd == "exit") {
                break;
            } else if (cmd == "load") {
                std::string path;
                in >> path;
                if (path.empty()) throw std::runtime_error("load needs a path");
                g_model_path = path;
                g_model_cpu.reset();
                g_model_gpu.reset();
                g_contexts.clear();
                // Validate by loading the CPU model now.
                model_for(false);
                std::cout << "ok\n";
            } else if (cmd == "info") {
                llama_model * m = model_for(false);
                const llama_vocab * v = llama_model_get_vocab(m);
                std::cout << "ok arch=qwen35 n_layer=" << llama_model_n_layer(m)
                          << " n_embd=" << llama_model_n_embd(m)
                          << " n_vocab=" << llama_vocab_n_tokens(v)
                          << " n_ctx_train=" << llama_model_n_ctx_train(m) << "\n";
            } else if (cmd == "tokenize") {
                std::string hex;
                in >> hex;
                int add_bos = 0;
                if (!(in >> add_bos)) add_bos = 0;
                std::string text = hex_decode(hex);
                llama_model * m = model_for(false);
                const llama_vocab * v = llama_model_get_vocab(m);
                int32_t n = llama_tokenize(v, text.data(), int32_t(text.size()), nullptr, 0,
                                           add_bos != 0, true);
                if (n < 0) n = -n;
                std::vector<llama_token> ids;
                ids.resize(size_t(n));
                n = llama_tokenize(v, text.data(), int32_t(text.size()), ids.data(), n,
                                   add_bos != 0, true);
                if (n < 0) throw std::runtime_error("tokenize failed");
                std::cout << "ok " << n;
                for (int32_t i = 0; i < n; ++i) std::cout << ' ' << ids[size_t(i)];
                std::cout << "\n";
            } else if (cmd == "ctx") {
                int id = 0, n_ctx = 0;
                std::string backend;
                in >> id >> n_ctx >> backend;
                if (n_ctx <= 0) throw std::runtime_error("n_ctx must be positive");
                bool vk = backend == "vulkan";
                if (backend != "vulkan" && backend != "cpu") throw std::runtime_error("backend must be cpu or vulkan");
                ctx_entry e;
                e.n_ctx = n_ctx;
                e.vulkan = vk;
                e.ctx.reset(make_context(n_ctx, vk));
                g_contexts[id] = std::move(e);
                std::cout << "ok\n";
            } else if (cmd == "reset") {
                int id = 0;
                in >> id;
                auto it = g_contexts.find(id);
                if (it == g_contexts.end()) throw std::runtime_error("unknown context");
                it->second.ctx.reset(make_context(it->second.n_ctx, it->second.vulkan));
                std::cout << "ok\n";
            } else if (cmd == "eval") {
                int id = 0, n = 0;
                in >> id >> n;
                auto it = g_contexts.find(id);
                if (it == g_contexts.end()) throw std::runtime_error("unknown context");
                std::vector<llama_token> ids;
                ids.resize(size_t(n));
                for (int i = 0; i < n; ++i) {
                    int t = 0;
                    if (!(in >> t)) throw std::runtime_error("missing token id");
                    ids[size_t(i)] = llama_token(t);
                }
                eval_tokens(it->second.ctx.get(), ids);
                std::cout << "ok\n";
            } else if (cmd == "logits") {
                int id = 0;
                std::string path;
                in >> id >> path;
                auto it = g_contexts.find(id);
                if (it == g_contexts.end()) throw std::runtime_error("unknown context");
                const llama_model * m = llama_get_model(it->second.ctx.get());
                const llama_vocab * v = llama_model_get_vocab(m);
                const int32_t n_vocab = llama_vocab_n_tokens(v);
                const float * logits = llama_get_logits_ith(it->second.ctx.get(), -1);
                if (!logits) throw std::runtime_error("no logits");
                std::ofstream f(path, std::ios::binary);
                if (!f) throw std::runtime_error("cannot open logits output");
                f.write(reinterpret_cast<const char *>(logits), size_t(n_vocab) * sizeof(float));
                if (!f) throw std::runtime_error("cannot write logits output");
                std::cout << "ok " << n_vocab << "\n";
            } else if (cmd == "gen") {
                int id = 0, n_predict = 0, ignore_eos = 0;
                in >> id >> n_predict;
                if (!(in >> ignore_eos)) ignore_eos = 0;
                auto it = g_contexts.find(id);
                if (it == g_contexts.end()) throw std::runtime_error("unknown context");
                llama_context * ctx = it->second.ctx.get();
                const llama_model * m = llama_get_model(ctx);
                const llama_vocab * v = llama_model_get_vocab(m);
                const int32_t n_vocab = llama_vocab_n_tokens(v);
                const llama_token eos = llama_vocab_eos(v);
                std::vector<int32_t> out;
                for (int i = 0; i < n_predict; ++i) {
                    const float * logits = llama_get_logits_ith(ctx, -1);
                    if (!logits) throw std::runtime_error("no logits");
                    int32_t tok = argmax(logits, n_vocab);
                    out.push_back(tok);
                    if (!ignore_eos && tok == eos) break;
                    std::vector<llama_token> one{llama_token(tok)};
                    eval_tokens(ctx, one);
                }
                std::cout << "ok " << out.size();
                for (int32_t t : out) std::cout << ' ' << t;
                std::cout << "\n";
            } else if (cmd.empty()) {
                continue;
            } else {
                throw std::runtime_error("unknown command");
            }
        } catch (const std::exception & e) {
            std::cout << "error " << e.what() << "\n";
        }
        std::cout.flush();
    }
    g_contexts.clear();
    g_model_gpu.reset();
    g_model_cpu.reset();
    llama_backend_free();
    return 0;
}
