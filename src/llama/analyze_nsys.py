# 读 nsys 导出的 sqlite，统计每组（impl|形状）里 norm1/attn/norm2/mlp/other 各占多少 GPU 时间。
# 用法：python src/llama/analyze_nsys.py <layer.sqlite> [输出 txt]
# 归属规则：kernel 通过 correlationId 找到启动它的 CUDA runtime 调用；该调用的 CPU 时间点落在哪个 NVTX 范围里，就算哪个模块。
import collections, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
PARTS = ("norm1", "attn", "norm2", "mlp")
ITERS = 20

def q(sql):
    return db.execute(sql).fetchall()

nvtx = q("SELECT start, end, COALESCE(text, (SELECT value FROM StringIds WHERE id = textId)) FROM NVTX_EVENTS WHERE end IS NOT NULL")
groups = [(a, b, t) for a, b, t in nvtx if t and "|" in t]
inner = [(a, b, t) for a, b, t in nvtx if t in PARTS]
launch = dict(q("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
kernels = q("SELECT k.correlationId, k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName")

def where(t, ranges):
    return next((name for a, b, name in ranges if a <= t <= b), None)

stats = collections.defaultdict(lambda: collections.defaultdict(float))
kstats = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
span = {}
for cid, ks, ke, name in kernels:
    t = launch.get(cid)
    if t is None: continue
    g = where(t, groups)
    if g is None: continue                          # 预热阶段的 kernel
    part = where(t, inner) or "other"
    dur = (ke - ks) / 1000                          # ns → μs
    stats[g][part] += dur
    k = kstats[(g, part)][name]; k[0] += dur; k[1] += 1
for a, b, t in groups:
    span[t] = (b - a) / 1000

lines = [f"每个数字是 {ITERS} 次前向的平均（μs/次）；'wall' 是外层 NVTX 范围的 CPU 时长 / {ITERS}，包含 GPU 空闲"]
for a, b, g in sorted(groups):
    total = sum(stats[g].values())
    norm = stats[g]["norm1"] + stats[g]["norm2"]
    lines.append(f"\n== {g} | kernel 合计 {total / ITERS:8.1f} μs | wall {span[g] / ITERS:8.1f} μs | "
                 f"RMSNorm×2 {norm / ITERS:7.1f} μs = {norm / total * 100:5.1f}% of kernel time")
    for part in PARTS + ("other",):
        lines.append(f"   {part:6s} {stats[g][part] / ITERS:8.1f} μs  {stats[g][part] / total * 100:5.1f}%")
        for name, (t, c) in sorted(kstats[(g, part)].items(), key=lambda kv: -kv[1][0]):
            lines.append(f"        {t / ITERS:7.1f} μs  x{c // ITERS:<2d} {name[:90]}")
text = "\n".join(lines)
print(text)
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(text + "\n")
