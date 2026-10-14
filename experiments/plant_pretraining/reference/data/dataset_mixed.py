import sys
import math
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from data.dataset_road_network import build_road_network_data
from data.dataset_plants import build_plants_network_data
from data.dataset_vessel3d import build_vessel_data


# -------------------------------------------------------------------------
# Builders
# -------------------------------------------------------------------------
SOURCE_BUILDERS = {
    "roads": build_road_network_data,
    "plants": build_plants_network_data,
    # HALVS leaf-vein patches from leaf-veins-extractor/generate_halvs_patches.py
    # are written in the exact raw/seg/vtp layout (128x128, 5px pad,
    # sample_NNNNNN_{data,seg,graph}.{png,png,vtp}) that build_plants_network_data
    # already reads -- no dedicated builder needed, just point SOURCE_DATA_PATHS
    # at a different root.
    "halvs": build_plants_network_data,
}

# We no longer hardcode TARGET_BUILDERS. 
# We assume any target dataset uses the standard 3D vessel builder.

def _as_plain_dict(maybe_cfg):
    if maybe_cfg is None:
        return None
    if isinstance(maybe_cfg, dict):
        return dict(maybe_cfg)
    if hasattr(maybe_cfg, "items"):
        return {k: v for k, v in maybe_cfg.items()}
    if hasattr(maybe_cfg, "__dict__"):
        return vars(maybe_cfg)
    return None

def _as_name_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]

def _source_normalization_message(source_name):
    if source_name in ("roads", "plants", "halvs", "octa"):
        return (
            "Input images are normalized by dividing by 255.0, then scaled to "
            "[-0.5, 0.5] via ScaleIntensity(minv=-0.5, maxv=0.5)."
        )
    return "Normalization strategy depends on the source builder implementation."

def _legacy_source_name_from_dataset_flag(config):
    """Backwards compatibility with older configs."""
    ds = getattr(config.DATA, "DATASET", None)
    if ds in ("mixed_real_vessels", "mixed_synth_3d"):
        return ["roads"]
    if ds in ("mixed_real_vessels_octa", "mixed_synth_3d_octa"):
        return ["octa"]
    return None

def build_mixed_data(
    config,
    mode="split",
    split=0.95,
    use_grayscale=False,
    debug=False,
    rotate=False,
    continuous=False,
    upsample_target_domain=True,
):
    # -----------------------
    # Decide source datasets
    # -----------------------
    include_source = bool(getattr(config.DATA, "MIXED", False))
    source_names = getattr(config.DATA, "SOURCE_DATASET_NAME", None)

    if include_source:
        if source_names is None:
            source_names = _legacy_source_name_from_dataset_flag(config)
        if source_names is None:
            raise ValueError(
                "MIXED=True requires source datasets. "
                "Provide SOURCE_DATASET_NAME and source sample counts, or set MIXED=False."
            )
        if isinstance(source_names, str):
            source_names = [source_names]
        else:
            source_names = list(source_names)
    else:
        source_names = []
        print("MIXED=False: composing target datasets only (source datasets disabled).")

    # -----------------------
    # Per-dataset source paths
    # -----------------------
    source_paths_by_ds = _as_plain_dict(getattr(config.DATA, "SOURCE_DATA_PATHS", None))
    legacy_source_path = getattr(config.DATA, "SOURCE_DATA_PATH", None)
    
    if include_source and source_paths_by_ds is None and legacy_source_path is not None:
        source_paths_by_ds = {name: legacy_source_path for name in source_names}

    if include_source and source_paths_by_ds is None:
        raise ValueError("Missing config.DATA.SOURCE_DATA_PATHS")

    # -----------------------
    # Per-dataset sample counts
    # -----------------------
    counts_by_ds = _as_plain_dict(getattr(config.DATA, "NUM_SOURCE_SAMPLES_BY_DATASET", None))
    legacy_num = getattr(config.DATA, "NUM_SOURCE_SAMPLES", None)
    
    if counts_by_ds is None:
        if legacy_num is None:
            counts_by_ds = {name: 0 for name in source_names}
        else:
            legacy_num = int(legacy_num)
            if len(source_names) == 1:
                counts_by_ds = {source_names[0]: legacy_num}
            else:
                base = legacy_num // len(source_names) if len(source_names) > 0 else 0
                rem = legacy_num - base * len(source_names)
                counts_by_ds = {n: base for n in source_names}
                if source_names:
                    counts_by_ds[source_names[-1]] += rem

    active_source_names = [name for name in source_names if int(counts_by_ds.get(name, 0)) > 0]
    if include_source and not active_source_names:
        raise ValueError(
            "MIXED=True but no source samples are active. "
            "Set MIXED=False for target-only composition or set source sample counts > 0."
        )

    # -----------------------
    # Build SOURCE datasets
    # -----------------------
    source_train_parts, source_val_parts = [], []
    for name in active_source_names:
        if name not in SOURCE_BUILDERS:
            raise ValueError(f"Unknown source dataset '{name}'. Available: {list(SOURCE_BUILDERS.keys())}")

        n_samples = int(counts_by_ds.get(name, 0))
        if n_samples <= 0:
            continue

        if name not in source_paths_by_ds:
            raise ValueError(
                f"Missing SOURCE_DATA_PATHS entry for '{name}'. "
                f"Available keys: {list(source_paths_by_ds.keys())}"
            )

        source_path = source_paths_by_ds[name]
        # Backward compatibility: source builders still read SOURCE_DATA_PATH.
        config.DATA.SOURCE_DATA_PATH = source_path
        # Keep DATA_PATH in sync for builders that rely on generic field.
        config.DATA.DATA_PATH = source_path
        builder = SOURCE_BUILDERS[name]
        print(f"[{name}] Normalization Strategy: {_source_normalization_message(name)}")

        train_i, val_i, _ = builder(
            config,
            mode,
            split,
            debug=debug,
            max_samples=n_samples,
            domain_classification=0,
            gaussian_augment=True,
            rotate=rotate,
            continuous=continuous,
        )
        source_train_parts.append(train_i)
        source_val_parts.append(val_i)

    source_train_data = ConcatDataset(source_train_parts) if source_train_parts else None
    source_val_data = ConcatDataset(source_val_parts) if source_val_parts else None

    # -----------------------
    # Build TARGET datasets
    # -----------------------
    target_names = _as_name_list(getattr(config.DATA, "TARGET_DATASET_NAME", None))
    if not target_names:
        target_names = ["syntheticMRI"]

    target_paths_by_ds = _as_plain_dict(getattr(config.DATA, "TARGET_DATA_PATHS", None))
    legacy_target_path = getattr(config.DATA, "TARGET_DATA_PATH", None)

    explicit_target_counts = _as_plain_dict(getattr(config.DATA, "NUM_TARGET_SAMPLES_BY_DATASET", None))
    legacy_target_num = int(getattr(config.DATA, "NUM_TARGET_SAMPLES", 0) or 0)
    target_counts_by_ds = explicit_target_counts
    if target_counts_by_ds is None:
        # Legacy fallback: preserve old global NUM_TARGET_SAMPLES behavior.
        if len(target_names) == 1:
            target_counts_by_ds = {target_names[0]: legacy_target_num}
        elif legacy_target_num > 0:
            base = legacy_target_num // len(target_names)
            rem = legacy_target_num - base * len(target_names)
            target_counts_by_ds = {n: base for n in target_names}
            target_counts_by_ds[target_names[-1]] += rem
        else:
            # Keep uncapped behavior (max_samples=0) for each target in legacy mode.
            target_counts_by_ds = {n: 0 for n in target_names}

    original_target_name = getattr(config.DATA, "TARGET_DATASET_NAME", None)
    original_target_path = getattr(config.DATA, "TARGET_DATA_PATH", None)

    target_train_parts, target_val_parts = [], []
    for name in target_names:
        # If per-dataset counts are explicitly configured, only load listed datasets.
        if explicit_target_counts is not None and name not in target_counts_by_ds:
            continue

        if target_paths_by_ds is not None and name in target_paths_by_ds:
            target_path = target_paths_by_ds[name]
        elif legacy_target_path is not None:
            target_path = legacy_target_path
        elif target_paths_by_ds is not None:
            raise ValueError(
                f"Missing TARGET_DATA_PATHS entry for '{name}'. "
                f"Available keys: {list(target_paths_by_ds.keys())}"
            )
        else:
            raise ValueError("Missing config.DATA.TARGET_DATA_PATH or config.DATA.TARGET_DATA_PATHS")

        n_target = int(target_counts_by_ds.get(name, legacy_target_num))
        max_target_samples = n_target if n_target > 0 else 0

        config.DATA.TARGET_DATASET_NAME = name
        config.DATA.TARGET_DATA_PATH = target_path
        # Keep DATA_PATH in sync for builders that rely on generic field.
        config.DATA.DATA_PATH = target_path

        train_i, val_i, _ = build_vessel_data(
            config,
            mode,
            split,
            debug=debug,
            max_samples=max_target_samples,
            domain_classification=1,
        )
        target_train_parts.append(train_i)
        target_val_parts.append(val_i)

    config.DATA.TARGET_DATASET_NAME = original_target_name
    config.DATA.TARGET_DATA_PATH = original_target_path

    if not target_train_parts:
        raise ValueError(
            "No target datasets were built. Check TARGET_DATASET_NAME, "
            "NUM_TARGET_SAMPLES_BY_DATASET, and TARGET_DATA_PATHS/TARGET_DATA_PATH."
        )

    target_train_data = ConcatDataset(target_train_parts)
    target_val_data = ConcatDataset(target_val_parts)

    # -----------------------
    # Concatenate + sampler
    # -----------------------
    num_samples_A = len(source_train_data) if source_train_data is not None else 0
    num_samples_B = len(target_train_data)

    if source_train_data is not None:
        target_multiplier = num_samples_A / max(1, num_samples_B)
        weights_A = torch.ones(num_samples_A)
        weights_B = torch.ones(num_samples_B) * target_multiplier
        train_ds = ConcatDataset([source_train_data, target_train_data])
        val_ds = ConcatDataset([source_val_data, target_val_data])
    else:
        weights_A = None
        weights_B = None
        train_ds = target_train_data
        val_ds = target_val_data

    print('-'*50)
    print('train set:', len(train_ds))
    if source_val_data is not None:
        print('validation set source:', len(source_val_data), '+ validation set target:', len(target_val_data), '=', len(val_ds))
    else:
        print('validation set target:', len(target_val_data))
    print('-'*50)

    print(f"samples A: {num_samples_A}")
    print(f"samples B: {num_samples_B}")
    if weights_A is not None and weights_B is not None:
        print(f"weight sum a: {torch.sum(weights_A)}")
        print(f"weight sum B: {torch.sum(weights_B)}")
    print('-'*50)

    if upsample_target_domain and source_train_data is not None:
        print("upsampling")
        sampler = WeightedRandomSampler(
            torch.cat([weights_A, weights_B]),
            num_samples=math.floor(torch.sum(weights_A) + torch.sum(weights_B)),
        )
    elif upsample_target_domain and source_train_data is None:
        print("upsampling disabled (target-only mode)")
        sampler = None
    else:
        print("not upsampling")
        sampler = None

    return train_ds, val_ds, sampler
