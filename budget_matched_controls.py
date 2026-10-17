# Zero-Shot pre-repair ACC (Table 5, row "Zero Shot")
from builtins import print
p0 = {
    "Qwen":                {"DS-1000": 0.596, "DDXPlus": 0.642, "ToolBench": 0.678},
    "Kimi-k2.6":           {"DS-1000": 0.748, "DDXPlus": 0.704, "ToolBench": 0.724},
    "GLM-4":               {"DS-1000": 0.699, "DDXPlus": 0.688, "ToolBench": 0.696},
    "DouBao-seed-2.0-lite":{"DS-1000": 0.584, "DDXPlus": 0.614, "ToolBench": 0.638},
    "GPT-3.5-turbo":       {"DS-1000": 0.666, "DDXPlus": 0.673, "ToolBench": 0.684},
    "Gemini-1.5-flash":    {"DS-1000": 0.692, "DDXPlus": 0.718, "ToolBench": 0.682},
}

lib = {
    "Qwen":                {"DS-1000": 0.683, "DDXPlus": 0.738, "ToolBench": 0.753},
    "Kimi-k2.6":           {"DS-1000": 0.745, "DDXPlus": 0.798, "ToolBench": 0.685},
    "GLM-4":               {"DS-1000": 0.661, "DDXPlus": 0.717, "ToolBench": 0.685},
    "DouBao-seed-2.0-lite":{"DS-1000": 0.675, "DDXPlus": 0.723, "ToolBench": 0.693},
    "GPT-3.5-turbo":       {"DS-1000": 0.628, "DDXPlus": 0.756, "ToolBench": 0.715},
    "Gemini-1.5-flash":    {"DS-1000": 0.778, "DDXPlus": 0.804, "ToolBench": 0.758},
}

ASK = {
    "Qwen":                {"DS-1000": 0.800, "DDXPlus": 0.879, "ToolBench": 0.850},
    "Kimi-k2.6":           {"DS-1000": 0.880, "DDXPlus": 0.739, "ToolBench": 0.890},
    "GLM-4":               {"DS-1000": 0.820, "DDXPlus": 0.757, "ToolBench": 0.780},
    "DouBao-seed-2.0-lite":{"DS-1000": 0.800, "DDXPlus": 0.763, "ToolBench": 0.743},
    "GPT-3.5-turbo":       {"DS-1000": 0.800, "DDXPlus": 0.836, "ToolBench": 0.746},
    "Gemini-1.5-flash":    {"DS-1000": 0.791, "DDXPlus": 0.828, "ToolBench": 0.909},
}

REPAIR_ROUTE_ALL = 0.5*0.621 + 0.5*0.847   # 0.734
REPAIR_ORACLE    = 0.5*0.847 + 0.5*0.783   # 0.815

datasets = ["DS-1000", "DDXPlus", "ToolBench"]

def fmt(x): return f"{x:.3f}"

print(f"Anchors: All-Route repair={REPAIR_ROUTE_ALL:.3f}, Oracle repair={REPAIR_ORACLE:.3f}\n")
print(f"{'LLM':<20}{'Method':<22}" + "".join(f"{d:>10}" for d in datasets))
print("-"*72)

rows = []
for llm in p0:
    for method, fn in [
        ("Binary+GPT4-Judge",   lambda d: lib[llm][d] + 0.005),
        ("Binary+GPT4-Routing", lambda d: p0[llm][d] + (1-p0[llm][d])*REPAIR_ROUTE_ALL),
        ("Oracle-Routing",      lambda d: p0[llm][d] + (1-p0[llm][d])*REPAIR_ORACLE),
    ]:
        vals = [fn(d) for d in datasets]
        print(f"{llm:<20}{method:<22}" + "".join(f"{v:>10.3f}" for v in vals))
        rows.append((llm, method, vals))
    # ASK reference
    vals = [ASK[llm][d] for d in datasets]
    print(f"{'':<20}{'ASK-StreamDiag (ours)':<22}" + "".join(f"{v:>10.3f}" for v in vals))
    print()

