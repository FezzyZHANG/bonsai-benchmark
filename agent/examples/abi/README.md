# ABI example

This is a minimal Zig shared library that exports the ABI in `../../ABI.md`.
It is an integration example only: every function returns an error. Start from
it, or write the same exports yourself.

```bash
zig build -Doptimize=ReleaseFast
# -> zig-out/lib/libbonsai.so
python3 harness_example.py
```

The example deliberately does **not** include a generated Vulkan binding. Zig
0.17 has no `@cImport`; use `zig translate-c` on
`$VULKAN_SDK/include/vulkan/vulkan.h`, write the small subset of bindings you
need, or use any other approach. The header `bonsai.h` is the contract the
harness uses; it is plain C and can be translated or copied.

`harness_example.py` loads the shared library with `ctypes` and checks the ABI
symbols. The real grader uses the same `dlopen`/`ctypes` style.
