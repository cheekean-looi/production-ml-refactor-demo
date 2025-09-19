import os, yaml, typer, torch, torch.optim as optim, torch.nn.functional as F
from rich.progress import Progress
from src.utils import set_seed, auto_device, ensure_dir, console, count_params
from src.data import make_loaders
from src.model import TinyCNN
from src.settings import settings

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str = typer.Option(..., "--config", help="Path to YAML config")):
    cfg = yaml.safe_load(open(config))
    set_seed(cfg["seed"])
    device = auto_device(cfg["training"]["device"])
    ensure_dir(cfg["artifacts_dir"])

    train_loader, val_loader = make_loaders(cfg)
    m = TinyCNN(
        in_channels=cfg["model"]["in_channels"],
        channels=tuple(cfg["model"]["channels"]),
        fc_dim=cfg["model"]["fc_dim"],
        num_classes=10,
    ).to(device)

    console.log(f"Device: {device}, Params: {count_params(m):,}")

    opt = optim.AdamW(m.parameters(), lr=cfg["training"]["lr"])

    best = 0.0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        m.train()
        with Progress() as prog:
            task = prog.add_task(f"[cyan]Epoch {epoch}", total=len(train_loader))
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad()
                logits = m(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
                prog.advance(task)

        # eval
        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = m(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        console.log(f"[green]val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save(m.state_dict(), os.path.join(cfg["artifacts_dir"], "model.pt"))
            console.log(f"[yellow]Saved checkpoint (acc={best:.4f})")

    # Optional MLflow auto-log
    if settings.mlflow_tracking_uri:
        import mlflow, mlflow.pytorch

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        with mlflow.start_run(run_name="train"):
            mlflow.log_metric("best_val_acc", best)
            mlflow.pytorch.log_model(m, "model")


if __name__ == "__main__":
    app()

