import json
import os
import sys
from pathlib import Path

from jupyter_client.manager import start_new_kernel


def run(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    env = dict(os.environ, PLOTLY_RENDERER="plotly_mimetype")
    km, kc = start_new_kernel(kernel_name="python3", cwd=str(path.parent), env=env)
    ok = True
    count = 0
    try:
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            cell["outputs"] = []
            if not src.strip():
                cell["execution_count"] = None
                continue
            count += 1
            msg_id = kc.execute(src)
            while True:
                msg = kc.get_iopub_msg(timeout=int(os.environ.get("CELL_TIMEOUT", 900)))
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                t, c = msg["msg_type"], msg["content"]
                if t == "status" and c["execution_state"] == "idle":
                    break
                if t == "stream":
                    outs = cell["outputs"]
                    if outs and outs[-1].get("output_type") == "stream" and outs[-1]["name"] == c["name"]:
                        outs[-1]["text"] += c["text"]
                    else:
                        outs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
                elif t in ("execute_result", "display_data"):
                    out = {"output_type": t, "data": c["data"], "metadata": c.get("metadata", {})}
                    if t == "execute_result":
                        out["execution_count"] = count
                    cell["outputs"].append(out)
                elif t == "error":
                    cell["outputs"].append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"], "traceback": c["traceback"]})
                    print(f"  ERRO na célula {idx}: {c['ename']}: {c['evalue']}")
                    ok = False
            cell["execution_count"] = count
            for o in cell["outputs"]:
                if o["output_type"] == "stream":
                    o["text"] = o["text"].splitlines(keepends=True)
            if not ok:
                break
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return ok


for name in sys.argv[1:]:
    p = Path(name)
    print(p.name, "...")
    print("  OK" if run(p) else "  FALHOU")
