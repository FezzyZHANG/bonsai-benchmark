# Sample prompts

`data/samples/` contains a visible corpus and a short prompt for development.
The grader uses different, hidden prompts and corpora.

Suggested workflow:

```bash
# token ids
ref/bin/bonsai-ref ...   # see ../reference/PROTOCOL.md
# or use the Python gguf reader and the reference tokenizer
```

The model's tokenizer details are in `TOKENIZER.md`. The ABI takes token ids, so
you can debug the forward pass independently of tokenization.
