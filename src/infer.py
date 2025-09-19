import os, yaml, torch, typer
from rich.table import Table
from src.utils import auto_device, console
from src.model import TinyCNN

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str = typer.Option(..., "--config")):
    cfg = yaml.safe_load(open(config))
    device = auto_device(cfg.get("device", "auto"))
    ckpt = os.path.join(cfg["artifacts_dir"], "model.pt")
    m = TinyCNN()
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval().to(device)

    bs, batches = cfg["batch_size"], cfg["batches"]
    x = torch.rand(bs, 1, 28, 28, device=device)

    # Warmup
    for _ in range(3):
        m(x)

    import time

    latencies = []
    with torch.inference_mode():
        for _ in range(batches):
            t0 = time.perf_counter()
            _ = m(x)
            latencies.append(time.perf_counter() - t0)
    import numpy as np

    table = Table(title="Inference Stats")
    table.add_column("Device")
    table.add_column("Batch")
    table.add_column("P50 (ms)")
    table.add_column("P95 (ms)")
    p50, p95 = np.percentile(latencies, 50) * 1000, np.percentile(latencies, 95) * 1000
    table.add_row(str(device), str(bs), f"{p50:.2f}", f"{p95:.2f}")
    console.print(table)


if __name__ == "__main__":
    app()

