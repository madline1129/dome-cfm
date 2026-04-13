import argparse
import importlib.util
from pathlib import Path

from mmengine import Config


def mark(ok, text):
    prefix = "[OK]" if ok else "[缺失]"
    print(f"{prefix} {text}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="检查 DOME flow+resample 训练前置文件")
    parser.add_argument("--py-config", default="config/train_dome_flow_resample.py")
    parser.add_argument("--vae-ckpt", default="ckpts/occvae_latest.pth")
    parser.add_argument("--resample-root", default="data/resampled_occ")
    parser.add_argument("--data-root", default="data/nuscenes")
    parser.add_argument("--train-imageset", default="data/nuscenes_infos_train_temporal_v3_scene.pkl")
    parser.add_argument("--val-imageset", default="data/nuscenes_infos_val_temporal_v3_scene.pkl")
    args = parser.parse_args()

    root = Path.cwd()
    checks = []

    cfg_path = root / args.py_config
    checks.append(mark(cfg_path.is_file(), f"配置文件: {cfg_path}"))
    if cfg_path.is_file():
        cfg = Config.fromfile(cfg_path)
        checks.append(mark(cfg.sample.get("sample_method") == "flow", "sample.sample_method == 'flow'"))
        checks.append(mark(cfg.train_dataset_config.type == "nuScenesSceneDatasetLidarResample", "训练集使用 nuScenesSceneDatasetLidarResample"))
        checks.append(mark(cfg.model.vae is not None, "配置中包含 OccVAE 结构，用于加载 checkpoint 后冻结"))

    checks.append(mark((root / args.vae_ckpt).is_file(), f"OccVAE checkpoint: {root / args.vae_ckpt}"))
    checks.append(mark((root / args.data_root).is_dir(), f"nuScenes 数据目录: {root / args.data_root}"))
    checks.append(mark((root / args.train_imageset).is_file(), f"train imageset: {root / args.train_imageset}"))
    checks.append(mark((root / args.val_imageset).is_file(), f"val imageset: {root / args.val_imageset}"))

    resample_root = root / args.resample_root
    has_cache = (resample_root / "scene_cache.npz").is_file()
    has_src_scene = any(resample_root.glob("src_scene*")) if resample_root.is_dir() else False
    checks.append(mark(resample_root.is_dir() and (has_cache or has_src_scene), f"resample 数据目录: {resample_root}"))

    torchcfm_spec = importlib.util.find_spec("torchcfm")
    checks.append(mark(torchcfm_spec is not None, "Python 包 torchcfm 可导入"))

    if all(checks):
        print("\n检查通过，可以运行: bash tools/train_dome_flow_resample.sh")
    else:
        print("\n检查未完全通过。缺 resample 数据时先运行: bash tools/prepare_resample_data.sh")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
