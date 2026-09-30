"""Single-field reconstruction, conservation checks, and NPZ export."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from astra.runtime import ROOT, configure_cuda, sha256
from astra.data.inputs import load_input, semantic_input
from astra.model.model import Direct8Model
from astra.model.fp32 import CONSERVATION_TOLERANCE, enable_fp32
from astra.inference.export_diagnostics import fov_grid_diagnostics


def predict(args):
    if args.output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("output or report exists; choose a new output path")
    metadata = json.loads((ROOT / "model/metadata.json").read_text(encoding="utf-8"))
    for name in ("checkpoint.pt", "config.json", "input_gene_ids.json", "output_gene_ids.json", "uni_config.json"):
        key = Path(name).stem + "_sha256"
        if sha256(ROOT / "model" / name) != metadata[key]:
            raise ValueError(f"bundled {name} identity mismatch")
    config = json.loads((ROOT / "model/config.json").read_text(encoding="utf-8"))
    input_genes = json.loads((ROOT / "model/input_gene_ids.json").read_text(encoding="utf-8"))
    output_genes = json.loads((ROOT / "model/output_gene_ids.json").read_text(encoding="utf-8"))
    if (config["kwargs"]["input_gene_ids"] != input_genes
            or config["kwargs"]["output_gene_ids"] != output_genes):
        raise ValueError("configuration and input/output gene panels differ")
    data = load_input(args.input, input_genes)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    memory_settings = configure_cuda(device, args.gpu_memory_gib) if device.type == "cuda" else {}
    started = time.monotonic()
    with torch.no_grad():
        features, valid, semantic_source = semantic_input(data, device, metadata, args.uni_checkpoint)
        # UNI is local to semantic_input; release its unused allocator cache before
        # materializing SR weights and observations on a desktop GPU.
        if device.type == "cuda" and semantic_source == "local_UNI1_RGB":
            torch.cuda.empty_cache()
        fine_tuning_report = {}
        if args.fine_tuned:
            from astra.fine_tuning.checkpoint import load_fine_tuned
            model, payload = load_fine_tuned(args.fine_tuned, config["kwargs"], metadata,
                sample_id=args.sample_id, task=args.task, device=device)
            fine_tuning_report = dict(fine_tuned_sha256=sha256(args.fine_tuned),
                fine_tuning=payload["fine_tuning"])
        elif args.trained_checkpoint:
            from astra.training.checkpoint import load_checkpoint
            model, payload = load_checkpoint(args.trained_checkpoint, device=device,
                expected_panels=dict(input_gene_ids=input_genes, output_gene_ids=output_genes))
            if payload.get("training_state", {}).get("uni_sha") != metadata["uni_checkpoint_sha256"]:
                raise ValueError("training checkpoint UNI identity differs from the published input preprocessing")
            model.eval()
            fine_tuning_report = dict(trained_checkpoint_sha256=sha256(args.trained_checkpoint),
                training_epoch=payload["training_state"]["epoch"],
                engineering_smoke_checkpoint=payload["training_state"].get("smoke", False))
        else:
            model = Direct8Model(**config["kwargs"]).to(device).eval()
            model.load_state_dict(torch.load(ROOT / "model/checkpoint.pt", map_location="cpu", weights_only=True), strict=True)
        if not args.fine_tuned:
            model = enable_fp32(model)
        inputs = {key: torch.from_numpy(data[key])[None].to(device) for key in (
            "parent_counts", "owner_map", "parent_valid", "field_valid", "image_features_2um", "gene_available")}
        inputs["image_features_2um"] = inputs["image_features_2um"].float()
        result = model(**inputs, protocol_id=int(data["protocol_id"]),
                         pathology_features=features, pathology_valid=valid, validate=True)
        output_counts = model.select_output_genes(inputs["parent_counts"])
        scaled = result["parent_conservation_residual"].abs() / (1 + output_counts.abs())
        if not bool(torch.isfinite(scaled).all()) or bool((scaled > CONSERVATION_TOLERANCE).any()):
            raise ValueError("parent conservation failed")
        prediction = result["pred_count_8um"][0].cpu().numpy()
        available = model.select_output_genes(inputs["gene_available"])[0].cpu().numpy()
    if prediction.shape != (32, 32, len(output_genes)) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("invalid prediction shape or values")
    active_sha = fine_tuning_report.get("fine_tuned_sha256", fine_tuning_report.get("trained_checkpoint_sha256", metadata["checkpoint_sha256"]))
    report = dict(model_name="ASTRA", checkpoint_sha256=active_sha,
                  source_resource_id=None if args.trained_checkpoint else metadata["source_resource_id"],
                  device=str(device), seconds=time.monotonic()-started, protocol_id=int(data["protocol_id"]),
                  parent_conservation_max_scaled=float(scaled.max()), semantic_source=semantic_source,
                  prediction_shape=list(prediction.shape), output_cell_um=8, field_extent_um=256,
                  arithmetic_dtype="float32", conservation_tolerance=CONSERVATION_TOLERANCE,
                  batch_size=1, cpu_threads=2, **memory_settings, **fine_tuning_report)
    if device.type == "cuda":
        report.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as reference:
            expected = reference["pred_count_8um"]
            if (expected.shape != prediction.shape or not np.array_equal(reference["gene_ids"], output_genes)
                    or not np.array_equal(reference["gene_available"], available)):
                raise ValueError("reference shape, genes or availability differs")
            passed = bool(np.allclose(prediction, expected, atol=1e-6, rtol=1e-5))
            report["reference"] = dict(passed=passed, max_absolute_error=float(np.max(np.abs(prediction-expected))),
                                       atol=1e-6, rtol=1e-5)
            if not passed:
                raise ValueError(f"reference replay failed: {report['reference']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    valid8 = data["field_valid"].reshape(32, 4, 32, 4).all(axis=(1, 3))
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, pred_count_8um=prediction, gene_ids=np.asarray(output_genes),
                            gene_available=available, field_valid_8um=valid8)
    report['parent_conservation_scope'] = 'pre_export_observation_cell_intersection_mass'
    report['serialized_export'] = fov_grid_diagnostics(args.output, data)
    report['seconds'] = time.monotonic()-started
    report['timing_scope'] = 'UNI if needed, model loading, inference, prediction archive write and serialized-grid diagnostics'
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
