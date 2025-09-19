import yaml, typer, torch, numpy as np
from src.utils import auto_device, console, Timer
from src.model import TinyCNN

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str = typer.Option(..., "--config")):
    cfg = yaml.safe_load(open(config))
    device = auto_device(cfg.get("device", "auto"))
    bs = cfg["batch_size"]
    reps = cfg["batches"]
    m = TinyCNN().to(device).eval()
    x = torch.rand(bs, 1, 28, 28, device=device)
    # warmup
    for _ in range(10):
        m(x)
    times = []
    with torch.inference_mode():
        for _ in range(reps):
            with Timer() as t:
                _ = m(x)
            times.append(t.dt)
    p50, p95 = np.percentile(times, 50) * 1000, np.percentile(times, 95) * 1000
    console.rule("[bold]Throughput/Latency")
    console.log(
        f"Device={device}  Batch={bs}  P50={p50:.2f}ms  P95={p95:.2f}ms  QPS≈{(bs/np.mean(times)):.1f}"
    )


if __name__ == "__main__":
    app()

