def make_config_spec(config, *, keys=None, title="Configuration"):
    if keys is None:
        keys = ["system", "seed", "n_states", "n_controls", "t1_initial", "t1_scale", "t1_max", "fp_tol"]
        keys = [k for k in keys if hasattr(config, k)]
    vals = [getattr(config, k) for k in keys]
    return table_kv(title=title, keys=keys, values=vals)


def make_traincfg_spec(train_cfg, *, keys=None, title="Training configuration"):
    if keys is None:
        keys = ["sup_epochs", "sup_lr", "unsup_epochs", "unsup_lr", "batch_size", "log_every", "grad_clip", "device"]
        keys = [k for k in keys if hasattr(train_cfg, k)]
    vals = [getattr(train_cfg, k) for k in keys]
    return table_kv(title=title, keys=keys, values=vals)


def make_data_summary_spec(config, data, *, title="Dataset summary"):
    import numpy as np
    t_all = np.asarray(data["t"]).reshape(-1)
    X_all = np.asarray(data["X"])
    X_norms = config.norm(X_all).reshape(-1)

    keys = ["n_trajectories", "n_points", "t_min", "t_max", "|X| mean", "|X| max", "||X|| mean", "||X|| max"]
    vals = [
        int(data.get("n_trajectories", -1)),
        int(X_all.shape[1]),
        f"{float(t_all.min()):.2f}",
        f"{float(t_all.max()):.2f}",
        f"{float(np.mean(np.abs(X_all))):.2f}",
        f"{float(np.max(np.abs(X_all))):.2f}",
        f"{float(np.mean(X_norms)):.2f}",
        f"{float(np.max(X_norms)):.2f}",
    ]
    return table_kv(title=title, keys=keys, values=vals)

def save_keyval_table_tex_2row(
    *,
    keys,
    values,
    savepath,
    title: str | None = None,   # optional bold title line above tabular
):
    import os

    def esc(s: str) -> str:
        return (s.replace("\\", "\\textbackslash{}")
                 .replace("&", "\\&")
                 .replace("%", "\\%")
                 .replace("$", "\\$")
                 .replace("#", "\\#")
                 .replace("_", "\\_")
                 .replace("{", "\\{")
                 .replace("}", "\\}")
                 .replace("~", "\\textasciitilde{}")
                 .replace("^", "\\textasciicircum{}"))

    keys = [esc(str(k)) for k in keys]
    vals = [esc(str(v)) for v in values]

    n = len(keys)
    if len(vals) != n:
        raise ValueError("keys and values must have same length")

    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)

    cols = "@{}" + ("c" * n) + "@{}"
    header = " & ".join([f"\\textbf{{{k}}}" for k in keys]) + " \\\\"
    row    = " & ".join(vals) + " \\\\"

    out = []
    if title:
        out.append(f"\\noindent\\textbf{{{esc(title)}}}\\\\")
    out += [
        "\\begingroup",
        "\\setlength{\\tabcolsep}{3.5pt} % tighter columns",
        "\\renewcommand{\\arraystretch}{1.15}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        header,
        "\\midrule",
        row,
        "\\bottomrule",
        "\\end{tabular}",
        "\\endgroup",
        "",
    ]

    with open(savepath, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

def table_kv(*, title=None, keys, values, precision=6):
    import numpy as np
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.{int(precision)}g}"
        return str(v)
    return {"title": title, "keys": list(keys), "values": [fmt(v) for v in values]}


def show_kv_table(spec, *, mode="1row"):
    import pandas as pd
    from IPython.display import display, Markdown

    if spec.get("title"):
        display(Markdown(f"**{spec['title']}**"))

    keys, values = spec["keys"], spec["values"]
    if mode == "2col":
        df = pd.DataFrame({"Parameter": keys, "Value": values})
    else:  # "1row"
        df = pd.DataFrame([values], columns=keys)

    display(df.style.hide(axis="index"))
    return df

def save_traincfg_table_tex(
    *,
    train_cfg,
    savepath="thesis/tables/traincfg.tex",
    keys=None,
    title: str | None = "Training configuration",
):
    if keys is None:
        keys = ["sup_epochs", "sup_lr", "unsup_epochs", "unsup_lr", "batch_size", "log_every", "grad_clip", "device"]
        keys = [k for k in keys if hasattr(train_cfg, k)]
    vals = []
    for k in keys:
        v = getattr(train_cfg, k)
        vals.append("" if v is None else (f"{v:.6g}" if isinstance(v, float) else v))
    save_keyval_table_tex_2row(keys=keys, values=vals, savepath=savepath, title=title)


def save_config_table_tex(
    *,
    config,
    savepath="thesis/tables/config.tex",
    keys=None,
    title: str | None = "Configuration",
):
    if keys is None:
        keys = ["system", "seed", "n_states", "n_controls", "t1_initial", "t1_scale", "t1_max", "fp_tol"]
        keys = [k for k in keys if hasattr(config, k)]
    vals = []
    for k in keys:
        v = getattr(config, k)
        vals.append(f"{v:.6g}" if isinstance(v, float) else v)
    save_keyval_table_tex_2row(keys=keys, values=vals, savepath=savepath, title=title)

def save_data_summary_table_tex(
    *,
    config,
    data,
    savepath="thesis/tables/data_summary.tex",
    title: str | None = "Dataset summary",
):
    import numpy as np

    t_all = np.asarray(data["t"]).reshape(-1)
    X_all = np.asarray(data["X"])  # (d,N)

    n_traj = int(data.get("n_trajectories", -1))
    n_points = int(X_all.shape[1])

    X_abs_mean = float(np.mean(np.abs(X_all)))
    X_abs_max  = float(np.max(np.abs(X_all)))

    X_norms = config.norm(X_all).reshape(-1)
    Xn_mean = float(np.mean(X_norms))
    Xn_max  = float(np.max(X_norms))

    keys = ["n_trajectories", "n_points", "t_min", "t_max", "|X| mean", "|X| max", "||X|| mean", "||X|| max"]
    vals = [
        n_traj,
        n_points,
        f"{float(t_all.min()):.2f}",
        f"{float(t_all.max()):.2f}",
        f"{X_abs_mean:.2f}",
        f"{X_abs_max:.2f}",
        f"{Xn_mean:.2f}",
        f"{Xn_max:.2f}",
    ]

    save_keyval_table_tex_2row(keys=keys, values=vals, savepath=savepath, title=title)

def show_katex_array(spec, *, bold_keys=True, caption=None, rules=True):
    from IPython.display import display, Latex, Markdown

    def esc(s: str) -> str:
        return (s.replace("\\", "\\\\")
                 .replace("&", "\\&")
                 .replace("%", "\\%")
                 .replace("$", "\\$")
                 .replace("#", "\\#")
                 .replace("_", "\\_")
                 .replace("{", "\\{")
                 .replace("}", "\\}")
                 .replace("~", "\\sim ")
                 .replace("^", "\\hat{}"))

    keys = [esc(str(k)) for k in spec["keys"]]
    vals = [esc(str(v)) for v in spec["values"]]
    n = len(keys)

    key_row = " & ".join([rf"\mathbf{{{k}}}" for k in keys]) if bold_keys else " & ".join(keys)
    val_row = " & ".join([rf"\text{{{v}}}" for v in vals])

    h = r"\hline" if rules else ""
    colspec = "c" * n
    latex = rf"\begin{{array}}{{{colspec}}}{h} {key_row} \\ {h} {val_row} \\ {h}\end{{array}}"
    display(Latex(latex))

    if caption:
        display(Markdown(f"<div style='text-align:center; font-size:0.9em;'><b>Table.</b> {caption}</div>"))

def save_results_table_tex(
    results,
    *,
    savepath="thesis/tables/controllers.tex",
    title: str | None = "Controller results",
    datasets=("Standard", "Hard"),
):
    import os
    from collections import defaultdict

    # group rows by controller then dataset
    by_ctrl = defaultdict(dict)
    for r in results:
        c = str(r.get("Controller", ""))
        d = str(r.get("Dataset", ""))
        by_ctrl[c][d] = r

    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)

    # common header keys (config style)
    keys = ["Dataset", "S", "tconv", "J"]
    blocks = []
    if title:
        blocks.append(f"\\noindent\\textbf{{{title}}}\\\\\n")

    for ctrl, rows in by_ctrl.items():
        # build one 2-row table per controller (same style as config)
        vals = []
        for ds in datasets:
            r = rows.get(ds, {})
            vals.append(str(ds))
            vals.append(str(r.get("Stability (S)", "")))
            vals.append(str(r.get("t_conv (med)", "")))
            vals.append(str(r.get("Avg Cost (J)", "")))

        # but keys need to match values -> repeat per dataset
        kk = []
        for ds in datasets:
            tag = "std" if ds.lower().startswith("std") else "hard"
            kk += [f"Dataset_{tag}", f"S_{tag}", f"tconv_{tag}", f"J_{tag}"]

        # reuse your existing 2-row writer into a string by writing to a temp path
        # simplest: directly call save_keyval_table_tex_2row with a per-controller path
        blocks.append(f"\\noindent\\textbf{{{ctrl}}}\\\\\n")
        tmp_path = savepath.replace(".tex", f"_{ctrl}.tex")
        save_keyval_table_tex_2row(keys=kk, values=vals, savepath=tmp_path, title=None)
        blocks.append(f"\\input{{{tmp_path[:-4]}}}\n\\vspace{{0.6em}}\n")

    with open(savepath, "w", encoding="utf-8") as f:
        f.write("".join(blocks))

def make_results_spec_2row(
    results,
    *,
    title="Results",
    datasets=("Standard", "Hard"),
    controller=None,   # if None and multiple controllers exist -> first one
):
    # group by controller then dataset
    from collections import defaultdict
    by_ctrl = defaultdict(dict)
    for r in results:
        c = str(r.get("Controller", ""))
        d = str(r.get("Dataset", ""))
        by_ctrl[c][d] = r

    if not by_ctrl:
        return table_kv(title=title, keys=["(no results)"], values=[""])

    ctrl = controller or sorted(by_ctrl.keys())[0]
    rows = by_ctrl[ctrl]

    keys = []
    vals = []
    for ds in datasets:
        tag = "std" if ds.lower().startswith("std") else "hard"
        r = rows.get(ds, {})
        keys += [f"S_{tag}", f"tconv_{tag}", f"J_{tag}"]
        vals += [
            r.get("Stability (S)", ""),
            r.get("t_conv (med)", ""),
            r.get("Avg Cost (J)", ""),
        ]

    return table_kv(title=f"{title}: {ctrl}", keys=keys, values=vals)
