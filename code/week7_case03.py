"""Run the Week 7 Case 3 benchmark from the archived Gwyddion height fields."""

import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from week7_figures import write_figures


# The channel IDs apply to the identified October archive. All heights are
# converted from metres to nanometres before fitting or measuring.
INPUT = dict(objects=0, corrugation=7, clean=8, tilt=4, noise=5, observed=9)
REFERENCES = dict(M0=10, M1=11, M2=13, M4=16, M5=19)
BACKGROUNDS = dict(M1=12, M2=14, M4=17, M5=20)
RESIDUALS = dict(M0=21, M1=22, M2=23, M4=24, M5=25)
WEIGHTS = {
    "mask": ("Heath-AFM-Lab/afMLevel-mask-unet", "mask_unet.pth", "3921e645c990de34cbc41da80b751f08673b6174"),
    "background": ("Heath-AFM-Lab/afMLevel-background-unet", "background_unet.pth", "600f394987ce85aac729f21eee33a6ee09bfbda2"),
}


def centred(values):
    """Align a constant height offset while retaining every spatial component."""
    return values - values.mean()


def fit_surface(height, keep, degree):
    """Fit a total-degree polynomial to the selected samples, evaluate everywhere."""
    y, x = np.indices(height.shape, dtype=float)
    x, y = centred(x) / np.ptp(x), centred(y) / np.ptp(y)
    # Degree 2 uses [1, y, y², x, xy, x²]. Independent x/y degrees would
    # include terms such as x²y² and is a different Gwyddion setting.
    design = np.stack([x**i * y**j for i in range(degree + 1)
                       for j in range(degree + 1 - i)], axis=-1)
    coefficients, _, rank, _ = np.linalg.lstsq(design[keep], height[keep], rcond=None)
    if rank != design.shape[-1]:
        raise ValueError("Selected fit pixels do not provide full rank.")
    return design @ coefficients


def load_archive(path):
    """Check calibration, archived residuals, mask polarity and the synthetic model."""
    import gwyfile

    source = gwyfile.load(str(path))
    fields = {}
    file_channels = {}
    ids = {**INPUT, **{f"gwy_{k}": v for k, v in REFERENCES.items()},
           **{f"gwy_bg_{k}": v for k, v in BACKGROUNDS.items()},
           **{f"gwy_res_{k}": v for k, v in RESIDUALS.items()}}
    for name, i in ids.items():
        key = f"/{i}/data"
        field = source[key]
        if (field.data.shape != (512, 512) or not np.isfinite(field.data).all()
                or any(field[f"si_unit_{axis}"]["unitstr"] != "m" for axis in ("xy", "z"))
                or not np.allclose([field.xreal, field.yreal], [5e-6, 5e-6], rtol=0, atol=1e-14)):
            raise ValueError(f"Incompatible channel {key} ({name})")
        fields[name] = field.data * 1e9
        file_channels[name] = {"id": i, "title": source[key + "/title"],
                               "log": source[key + "/log"]["strings"]}

    z, clean = fields["observed"], fields["clean"]
    np.testing.assert_allclose(z, clean + fields["tilt"] + fields["noise"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(clean, fields["objects"] + fields["corrugation"], atol=1e-8, rtol=0)
    keep = source["/16/mask"].data == 0  # Gwyddion nonzero means excluded foreground here.
    np.testing.assert_array_equal(keep, fields["objects"] <= .500)
    for method in REFERENCES:
        output = fields[f"gwy_{method}"]
        np.testing.assert_allclose(centred(output - clean), fields[f"gwy_res_{method}"], atol=1e-8, rtol=0)
        if method in BACKGROUNDS:
            np.testing.assert_allclose(z - output, fields[f"gwy_bg_{method}"], atol=1e-8, rtol=0)
    return fields, keep, file_channels


def measure(output, data, substrate):
    """Use the November 19 notebook's measurement definitions for all runs."""
    residual = centred(output - data["clean"])
    removed = data["observed"] - output
    foreground = ~substrate
    # Old notebook: object coefficient from full joint regression. Corrugation
    # and remaining tilt from a separate joint fit on the substrate only.
    truth_basis = np.stack([np.ones_like(output), data["objects"],
                            data["corrugation"], data["tilt"]], axis=-1)
    full_fit = np.linalg.lstsq(truth_basis.reshape(-1, 4), output.ravel(), rcond=None)[0]
    local_basis = truth_basis[..., [0, 2, 3]]
    substrate_fit = np.linalg.lstsq(local_basis[substrate], output[substrate], rcond=None)[0]
    return dict(
        rmse_nm=float(np.sqrt(np.mean(residual**2))),
        removal_p2p_nm=float(np.ptp(removed)),
        removal_vs_tilt_rmse_nm=float(np.std(removed - data["tilt"])),
        foreground_rmse_nm=float(np.sqrt(np.mean(residual[foreground]**2))),
        substrate_rmse_nm=float(np.sqrt(np.mean(residual[substrate]**2))),
        object_full_pct=float(100 * full_fit[1]),
        corrugation_substrate_pct=float(100 * substrate_fit[1]),
        remaining_tilt_substrate_pct=float(100 * substrate_fit[2]),
    )


def ml_methods(z, traces):
    """Run pinned afMLevel inference and observe the actual stage arrays."""
    import torch
    from afmlevel.background_model import level_ml_bg
    from afmlevel.mask_model import DEFAULT_ML_ROUTINES, level_ml_mask, ml_mask
    from afmlevel.utils import xyplanefit
    from huggingface_hub import hf_hub_download

    torch.set_num_threads(2)
    models = {k: hf_hub_download(repo, filename, revision=revision)
              for k, (repo, filename, revision) in WEIGHTS.items()}
    mask_args = dict(model_path=models["mask"], device="cpu", threshold=.5, min_size=30)
    # Standalone ML1 deliberately uses a mask inferred from the original input.
    # M6a's internal mask is inferred after its own first levelling stage.
    keep = ml_mask(z, **mask_args) == 1

    def capture(name):
        def observe(image, mask=None):
            traces[name] = image.copy()
            if mask is not None:
                traces["M6a_keep"] = ~mask
            return image
        return {"func": observe}

    original = DEFAULT_ML_ROUTINES["ml-mask"]
    routine = [original[0], capture("M6a_after_first"), original[1],
               original[2], capture("M6a_after_rows"), original[3]]
    m6a = level_ml_mask(z, method="traced", ml_routines={"traced": routine}, **mask_args)
    official = level_ml_mask(z, method="ml-mask", **mask_args)
    np.testing.assert_allclose(m6a, official, atol=1e-8, rtol=0)
    traces["M6a_first_removed"] = z - traces["M6a_after_first"]
    traces["M6a_masked_rows_removed"] = traces["M6a_after_first"] - traces["M6a_after_rows"]
    traces["M6a_masked_plane_removed"] = traces["M6a_after_rows"] - m6a
    np.testing.assert_allclose(
        traces["M6a_first_removed"] + traces["M6a_masked_rows_removed"] +
        traces["M6a_masked_plane_removed"], z - m6a, atol=1e-8, rtol=0)

    bg_args = dict(model_path=models["background"], device="cpu", line_order=3, zero_median=False)
    m6b = level_ml_bg(z, **bg_args)
    # background=True returns only the later prediction after the initial fit.
    traces["M6b_after_first"] = xyplanefit(z, 1, 1)
    traces["M6b_first_removed"] = z - traces["M6b_after_first"]
    traces["M6b_predicted_removed"] = level_ml_bg(z, background=True, **bg_args)
    np.testing.assert_allclose(traces["M6b_after_first"] - traces["M6b_predicted_removed"],
                               m6b, atol=1e-8, rtol=0)
    return keep, m6a, m6b


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(source, out, previous=None):
    data, oracle, channels = load_archive(source)
    z, clean = data["observed"], data["clean"]
    every_pixel = np.ones(z.shape, dtype=bool)
    first = fit_surface(z, every_pixel, 1)
    traces = {"keep_oracle": oracle, "controlled_first_removed": first,
              "controlled_after_first": z - first,
              "M2_removed": fit_surface(z, every_pixel, 2),
              "M4_removed": fit_surface(z, oracle, 2)}
    learned, m6a, m6b = ml_methods(z, traces)
    traces["keep_learned"] = learned
    for label, keep in (("O1", oracle), ("ML1", learned)):
        # The fitted model space stays fixed, so adding the two planes must
        # match fitting the original image directly on the chosen pixels.
        second = fit_surface(z - first, keep, 1)
        traces[f"{label}_second_removed"] = second
        np.testing.assert_allclose(first + second, fit_surface(z, keep, 1), atol=1e-8, rtol=0)
    results = {
        "M0": centred(z), "M1": z - first,
        "M2": z - traces["M2_removed"], "M4": z - traces["M4_removed"],
        "O1": z - first - traces["O1_second_removed"],
        "ML1": z - first - traces["ML1_second_removed"],
        "M6a": m6a, "M6b": m6b,
        "Gwy M5": data["gwy_M5"],
    }
    for name, output in results.items():
        if output.shape != z.shape or not np.isfinite(output).all():
            raise ValueError(f"Invalid output map: {name}")
    metrics = [dict(method=k, origin="Gwyddion archive" if k == "Gwy M5" else "Python",
                    **measure(v, data, oracle)) for k, v in results.items()]
    parity = []
    for k in ("M0", "M1", "M2", "M4"):
        diff = results[k] - data[f"gwy_{k}"]
        np.testing.assert_allclose(centred(diff), 0, atol=1e-8, rtol=0)
        parity.append(dict(method=k, mean_offset_nm=diff.mean(),
                           max_abs_aligned_nm=np.max(abs(centred(diff))),
                           aligned_rmse_nm=np.std(diff)))

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "metrics_vs_clean.csv", metrics)
    write_csv(out / "python_vs_gwyddion.csv", parity)
    if previous:
        old = json.loads(previous.read_text())["methods"]
        checks = []
        for k, v in results.items():
            old_key = "M5" if k == "Gwy M5" else k
            if old_key in old:
                new = measure(v, data, oracle)
                for key, prior_key in (
                    ("rmse_nm", "global_offset_aligned_rmse_nm"),
                    ("object_full_pct", "object_retention_pct_full_joint_fit"),
                    ("corrugation_substrate_pct", "corrugation_retention_pct_substrate_fit"),
                    ("remaining_tilt_substrate_pct", "remaining_plane_pct_substrate_fit"),
                ):
                    prior = old[old_key][prior_key]
                    checks.append(dict(method=old_key, metric=key, november_19=prior,
                                       week7=new[key], delta=new[key] - prior))
                    np.testing.assert_allclose(new[key], prior, atol=1e-8, rtol=0)
        write_csv(out / "nov19_consistency.csv", checks)

    row = 256
    profiles = {"Clean target": clean}
    profiles.update({k: clean + centred(v - clean) for k, v in results.items()})
    values = np.column_stack([np.arange(512) * 5 / 512, *[v[row] for v in profiles.values()]])
    np.savetxt(out / "profile_y2p500um.csv", values, delimiter=",", comments="",
               header="x_um," + ",".join(profiles))
    # Fix the sample locations from the truth before comparing any output.
    # These are pointwise profile errors, not width or fitted peak estimates.
    boundaries = np.flatnonzero(np.diff(np.r_[False, ~oracle[row], False]))
    points = [start + np.argmax(data["objects"][row, start:end])
              for start, end in zip(boundaries[::2], boundaries[1::2])]
    write_csv(out / "profile_truth_peak_errors.csv", [
        dict(method=k, peak=i, x_um=col*5/512, truth_height_nm=clean[row, col],
             aligned_height_nm=aligned_map[row, col],
             signed_height_error_nm=aligned_map[row, col]-clean[row, col])
        for k, aligned_map in profiles.items() if k != "Clean target"
        for i, col in enumerate(points, 1)
    ])
    np.savez_compressed(out / "height_maps_nm.npz", **data, **traces,
                        **{f"output_{k}": v for k, v in results.items()},
                        **{f"removed_{k}": z-v for k, v in results.items()},
                        **{f"residual_{k}": centred(v-clean) for k, v in results.items()})
    with source.open("rb") as downloaded:
        digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
    metadata = dict(
        source=source.name, sha256=digest, channels=channels,
        lateral_size_um=[5, 5], shape=list(z.shape), z_unit="nm", profile_row_zero_based=row,
        profile_y_um=2.5, oracle_threshold_nm=.5, learned_probability_threshold=.5,
        ml1_foreground_iou_oracle=float(np.sum(~learned & ~oracle) / np.sum(~learned | ~oracle)),
        ml1_foreground_fraction=float(np.mean(~learned)), m6a_foreground_fraction=float(np.mean(~traces["M6a_keep"])),
        ml1_and_m6a_fit_masks_equal=bool(np.array_equal(learned, traces["M6a_keep"])),
        oracle_foreground_fraction=float(np.mean(~oracle)), known_tilt_p2p_nm=float(np.ptp(data["tilt"])),
        gaussian_noise_rms_nm=float(np.std(data["noise"])),
        settings=dict(polynomial_degrees=[1, 2], independent_xy_degree=False,
                      m6a_routine="ml-mask", m6b_line_order=3, m6b_zero_median=False,
                      mask_min_size=30, device="cpu"),
        weights=WEIGHTS, versions={p: version(p) for p in
                                   ("numpy", "gwyfile", "matplotlib", "afmlevel", "pnanolocz", "torch")},
    )
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_figures(out, data, results, traces, metrics, row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gwy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous", type=Path, help="Optional October metrics used in the November 19 notebook.")
    args = parser.parse_args()
    run(args.gwy, args.out, args.previous)
