"""
windows/_fix_ultra_settings.py
------------------------------
把随包的 Ultralytics settings.json 里写死的 Linux 绝对路径改成本机 Windows 路径，
并确保离线安全开关（关闭 sync/hub/各类遥测）。由 install.ps1 用打包的 python 调用：

    python _fix_ultra_settings.py <settings.json 路径> <backend 目录>
"""
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("用法: _fix_ultra_settings.py <settings.json> <backend_dir>")
        return 2
    settings_path = Path(sys.argv[1])
    backend = Path(sys.argv[2])

    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    # 用正斜杠写路径：ultralytics/torch 都能接受，且避免 JSON 里反斜杠转义问题。
    datasets_dir = str((backend / "datasets")).replace("\\", "/")
    data.update({
        "sync": False, "hub": False, "clearml": False, "comet": False,
        "dvc": False, "mlflow": False, "neptune": False, "raytune": False,
        "tensorboard": False, "wandb": False,
        "datasets_dir": datasets_dir,
        "weights_dir": "weights",
        "runs_dir": "runs",
    })
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"[OK] Ultralytics settings.json 已更新: {settings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
