#!/usr/bin/env python3
"""Check a vulkaninfo -j dump against the Khronos Vulkan Roadmap 2026 profile.

Usage:
  check_roadmap2026.py <VP_KHR_roadmap.json> <VP_VULKANINFO_*.json>

The profiles JSON is shipped in the Vulkan SDK; vulkaninfo -j emits the same
Vulkan Profiles schema for the actual device.
"""
import json
import sys


def walk_caps(profile, cap, path, out):
    if isinstance(cap, list):
        out.append(("any", [walk_caps(profile, c, path + [f"alt{i}"], []) for i, c in enumerate(cap)]))
        return out
    c = profile["capabilities"].get(cap)
    if c is None:
        out.append(("missing_cap", cap))
        return out
    out.append(("cap", cap, c))
    return out


def flatten(cap, out):
    """Collect flat requirements; nested lists are OR-groups."""
    if isinstance(cap, list):
        out.append(("or", cap))
        return
    c = cap
    for e, v in c.get("extensions", {}).items():
        out.append(("ext", e, v))
    for s, flags in c.get("features", {}).items():
        for f, v in flags.items():
            out.append(("feat", s, f, v))
    for s, flags in c.get("properties", {}).items():
        for f, v in flags.items():
            out.append(("prop", s, f, v))


def collect(profile, cap, out):
    if isinstance(cap, list):
        out.append(("or", [collect(profile, c, [])[0] if False else c for c in cap]))
        return out
    c = profile["capabilities"].get(cap)
    if c is None:
        out.append(("missing_cap", cap))
        return out
    flatten(c, out)
    return out


def device_ext_ver(dev, name):
    e = dev["extensions"].get(name)
    if e is None:
        return None
    if isinstance(e, dict):
        return e.get("specVersion", e.get("version", 1))
    return e


def dev_feat(dev, struct, field):
    s = dev["features"].get(struct) or {}
    return s.get(field)


def dev_prop(dev, struct, field):
    s = dev["properties"].get(struct) or {}
    return s.get(field)


INSTANCE_EXTS = {
    "VK_KHR_surface",
    "VK_KHR_get_surface_capabilities2",
    "VK_KHR_surface_maintenance1",
}


def check_prop_value(path, v, got):
    if isinstance(v, dict):
        if not isinstance(got, dict):
            return f"property {path} has wrong type"
        for k, vv in v.items():
            g = got.get(k)
            if g is None:
                return f"property {path}.{k} missing"
            m = check_prop_value(f"{path}.{k}", vv, g)
            if m:
                return m
        return None
    if isinstance(v, bool):
        ok = bool(got) == v
    elif isinstance(v, (int, float)) and isinstance(got, (int, float)):
        # Granularity / alignment / atom-size limits are upper bounds; all
        # other numeric property requirements in the roadmap are minimums.
        upper = path.endswith("bufferImageGranularity") or "Alignment" in path or path.endswith("nonCoherentAtomSize")
        ok = got <= v if upper else got >= v
    elif isinstance(v, list) and isinstance(got, list):
        if all(isinstance(x, str) for x in v):
            ok = set(v) <= set(got)
        elif path.endswith("viewportBoundsRange") and len(v) == len(got) == 2:
            ok = got[0] <= v[0] and got[1] >= v[1]
        else:
            ok = len(v) == len(got) and all(g >= r for g, r in zip(got, v))
    else:
        ok = got == v
    return None if ok else f"property {path} = {got}, required {v}"


def check_req(req, dev):
    kind = req[0]
    if kind == "ext":
        _, name, ver = req
        dv = device_ext_ver(dev, name)
        if dv is None:
            if name in INSTANCE_EXTS:
                return None
            return f"extension {name} missing"
        if isinstance(ver, int) and isinstance(dv, int) and dv < ver:
            return f"extension {name} rev {dv} < required {ver}"
        return None
    if kind == "feat":
        _, s, f, v = req
        got = dev_feat(dev, s, f)
        if got is None:
            return f"feature {s}.{f} missing"
        if bool(got) != bool(v):
            return f"feature {s}.{f} = {got}, required {v}"
        return None
    if kind == "prop":
        _, s, f, v = req
        got = dev_prop(dev, s, f)
        if got is None:
            return f"property {s}.{f} missing"
        return check_prop_value(f"{s}.{f}", v, got)
    if kind == "missing_cap":
        return f"profile capability {req[1]} missing from profile file"
    return None


def eval_group(profile, cap, dev):
    """Return list of missing strings for this capability (OR groups pass if any member passes)."""
    if isinstance(cap, list):
        best = None
        for c in cap:
            m = eval_group(profile, c, dev)
            if not m:
                return []
            if best is None or len(m) < len(best):
                best = m
        return ["OR-group: " + "; ".join(best or [])]
    c = profile["capabilities"].get(cap)
    if c is None:
        return [f"profile capability {cap} missing"]
    reqs = []
    flatten(c, reqs)
    return [m for m in (check_req(r, dev) for r in reqs) if m]


def main():
    prof = json.load(open(sys.argv[1]))
    devs = json.load(open(sys.argv[2]))
    dev = devs["capabilities"]["device"]
    profile = prof["profiles"]["VP_KHR_roadmap_2026"]

    missing = []
    for cap in profile["capabilities"]:
        missing += eval_group(prof, cap, dev)
    # dedupe, preserve order
    missing = list(dict.fromkeys(missing))

    print(f"profile api-version: {profile.get('api-version')}")
    print(f"device apiVersion  : {dev['properties']['VkPhysicalDeviceProperties']['apiVersion']}")
    print(f"extensions on device: {len(dev['extensions'])}")
    print(f"missing requirement groups: {len(missing)}")
    for m in missing:
        print("  -", m)


if __name__ == "__main__":
    main()
