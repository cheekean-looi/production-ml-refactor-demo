import os, yaml, torch, typer
from src.model import TinyCNN
from src.utils import auto_device, ensure_dir, console

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str = typer.Option(..., "--config")):
    cfg = yaml.safe_load(open(config))
    device = auto_device("cpu")  # export on CPU for portability
    ckpt = os.path.join(cfg["artifacts_dir"], "model.pt")
    onnx_path = os.path.join(cfg["artifacts_dir"], "model.onnx")
    ensure_dir(cfg["artifacts_dir"])

    m = TinyCNN()
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    x = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        m,
        x,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    console.log(f"[green]Exported ONNX -> {onnx_path}")


if __name__ == "__main__":
    app()

