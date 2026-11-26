"""Publication figures from calibrated numerical height arrays."""

import numpy as np


def write_figures(out, data, results, traces, metrics, row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    z, clean = data["observed"], data["clean"]
    residual = {k: a - clean - np.mean(a - clean) for k, a in results.items()}
    height = {k: clean + e for k, e in residual.items()}
    removed = {k: z - a - np.mean(z - a) for k, a in results.items()}
    height_bounds = (min(clean.min(), *(a.min() for a in height.values())),
                     max(clean.max(), *(a.max() for a in height.values())))
    removal_max = max(abs(a).max() for a in removed.values())
    residual_max = max(abs(a).max() for a in residual.values())
    map_extent = (0, 5, 5, 0)

    def write(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(out / f"{name}.{suffix}", dpi=180)
        plt.close(fig)

    def grid_figure(name, columns, rows):
        """Calibrate geometry once and share colour ranges within each field type."""
        n = len(columns)
        fig = plt.figure(figsize=(2.22*n + .8, 2.42*len(rows)), layout="constrained")
        grid = fig.add_gridspec(len(rows), n + 1, width_ratios=[1]*n + [.065])
        for i, (label, arrays, palette, bounds) in enumerate(rows):
            for j, (title, a) in enumerate(zip(columns, arrays)):
                ax = fig.add_subplot(grid[i, j])
                m = ax.imshow(a, extent=map_extent, cmap=palette,
                              vmin=bounds[0], vmax=bounds[1], interpolation="none")
                ax.set_xticks([0, 2.5, 5])
                ax.set_yticks([0, 2.5, 5])
                ax.tick_params(labelleft=j == 0, labelbottom=i == len(rows) - 1)
                if i == 0:
                    ax.set_title(title, fontsize=10)
                if j == 0:
                    ax.set_ylabel("y (µm)")
                if i == len(rows) - 1:
                    ax.set_xlabel("x (µm)")
            fig.colorbar(m, cax=fig.add_subplot(grid[i, n]), label=label)
        write(fig, name)

    for filename, group in (
        ("01_classic_vs_clean", ["M0", "M1", "M2", "M4", "Gwy M5"]),
        ("02_masks_and_ml_vs_clean", ["O1", "ML1", "M6a", "M6b"]),
    ):
        grid_figure(filename, ["Reference", *group], [
            ("Aligned height (nm)", [clean, *[height[k] for k in group]], "cividis", height_bounds),
            ("Centred removal (nm)", [data["tilt"]-data["tilt"].mean(), *[removed[k] for k in group]],
             "RdBu_r", (-removal_max, removal_max)),
            ("Residual (nm)", [np.zeros_like(clean), *[residual[k] for k in group]],
             "RdBu_r", (-residual_max, residual_max)),
        ])

    group = ["M0", "M1", "M2", "M4"]
    gwy = [clean + (data[f"gwy_{k}"] - clean) - np.mean(data[f"gwy_{k}"] - clean) for k in group]
    delta = [height[k] - value for k, value in zip(group, gwy)]
    grid_figure("03_python_vs_gwyddion", group, [
        ("Python height (nm)", [height[k] for k in group], "cividis", height_bounds),
        ("Gwyddion height (nm)", gwy, "cividis", height_bounds),
        ("Aligned difference (nm)", delta, "RdBu_r", (-1e-8, 1e-8)),
    ])

    fig, axes = plt.subplots(2, 4, figsize=(10.3, 5.5), layout="constrained")
    panels = [
        ("Injected tilt", data["tilt"], False),
        ("M1 first removal", traces["controlled_first_removed"], False),
        ("O1 fit pixels", traces["keep_oracle"], True),
        ("O1 second removal", traces["O1_second_removed"], False),
        ("M6a actual fit pixels", traces["M6a_keep"], True),
        ("M6a masked row removal", traces["M6a_masked_rows_removed"], False),
        ("M6a masked plane removal", traces["M6a_masked_plane_removed"], False),
        ("M6b model-stage removal", traces["M6b_predicted_removed"], False),
    ]
    shared = max(np.max(np.abs(a - a.mean())) for _, a, mask in panels if not mask)
    for i, (ax, (title, a, mask)) in enumerate(zip(axes.flat, panels)):
        # The visualisation removes map means, but the NPZ preserves the full
        # stages, including all constants needed to reconstruct the output.
        if mask:
            ax.imshow(a, extent=map_extent, cmap="gray", vmin=0, vmax=1)
        else:
            im = ax.imshow(a-a.mean(), extent=map_extent, cmap="RdBu_r", vmin=-shared, vmax=shared)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([0, 2.5, 5])
        ax.set_yticks([0, 2.5, 5])
        ax.tick_params(labelleft=i % 4 == 0, labelbottom=i >= 4)
    fig.supylabel("y (µm)")
    fig.supxlabel("x (µm)")
    fig.colorbar(im, ax=axes, label="Centred removal (nm)", shrink=.8)
    write(fig, "04_stage_evidence")

    x = np.arange(512)*5/512
    fig, axs = plt.subplots(2, 2, figsize=(11, 5.6), sharex=True, sharey="row", layout="constrained")
    for j, group in enumerate((("M0", "M1", "M2", "M4", "Gwy M5"), ("O1", "ML1", "M6a", "M6b"))):
        axs[0, j].plot(x, clean[row], "k", lw=2.2, label="Clean")
        for k in group:
            line, = axs[0, j].plot(x, height[k][row], lw=1.1, label=k)
            axs[1, j].plot(x, residual[k][row], lw=1.0, color=line.get_color())
        axs[0, j].set_title(("Classical methods", "Mask source and full ML routines")[j])
        axs[0, j].legend(ncols=3, fontsize=8)
        axs[1, j].axhline(0, color="k", lw=.6)
        axs[1, j].set_xlabel("x (µm)")
    axs[0, 0].set_ylabel("Aligned height (nm)")
    axs[1, 0].set_ylabel("Residual (nm)")
    fig.suptitle("Fixed profile: y = 2.500 µm, row 256")
    write(fig, "05_fixed_profile")

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.8), layout="constrained")
    names = [m["method"] for m in metrics]
    palette = ["#1b7776"]*4 + ["#285b91"]*4 + ["#8a8a8a"]
    for ax, (key, label) in zip(axs, (("rmse_nm", "Aligned RMSE (nm)"),
                                      ("corrugation_substrate_pct", "Corrugation coefficient (%)"),
                                      ("remaining_tilt_substrate_pct", "Remaining tilt coefficient (%)"))):
        ax.bar(names, [m[key] for m in metrics], color=palette)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=55)
        ax.axhline(100 if key.startswith("corrugation") else 0, color="k", linestyle="--", lw=.75)
        ax.spines[["top", "right"]].set_visible(False)
    write(fig, "06_metrics_summary")
