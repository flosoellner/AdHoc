import os
import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display, Markdown

def _get_savepath(savepath, config):
    """
    Convert savepath to new results directory structure.
    If savepath contains 'tables', redirect to results/{system}/seed_{seed}/thesis/tables/
    """
    if config and hasattr(config, "system") and hasattr(config, "seed") and "tables" in savepath:
        from config import get_results_dir
        # Extract filename
        p = Path(savepath)
        filename = p.name
        # Use new structure
        new_path = get_results_dir(config, "thesis/tables")
        return str(Path(new_path) / filename)
    return savepath

def save_dataframe_latex(df, savepath, caption=None, label=None, precision=4):
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    disp_df = df.copy()
    
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.2e}" if (abs(x) < 1e-4 or abs(x) > 1e4) else f"{x:.{precision}g}"
        return str(x)

    def escape_tex(s):
        return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
        
    for col in disp_df.columns:
        disp_df[col] = disp_df[col].map(fmt).map(escape_tex)
    disp_df.columns = disp_df.columns.map(escape_tex)

    # Use Styler for Pandas 2.0+
    styler = disp_df.style.hide(axis="index")
    latex_str = styler.to_latex(
        column_format="l" + "r" * (len(df.columns) - 1),
        hrules=True, label=label, caption=caption,
        position="H", position_float="centering"
    )

    with open(savepath, "w", encoding="utf-8") as f:
        f.write(latex_str)

# --- Notebook Display Helper ---

def show_spec(obj, keys, title=None):
    """
    Cleaner replacement for show_katex_array.
    Creates a spec on the fly and displays it as a styled HTML table.
    """
    valid_keys = [k for k in keys if hasattr(obj, k)]
    vals = [getattr(obj, k) for k in valid_keys]
    
    df = pd.DataFrame([vals], columns=valid_keys)
    if title:
        display(Markdown(f"**{title}**"))
    
    # Stylish Jupyter display
    display(df.style.hide(axis="index").set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f4f4f4'), ('color', 'black'), ('font-weight', 'bold')]}
    ]))

# --- Specific Thesis Helpers ---

def save_config_table(config, savepath="thesis/tables/config.tex"):
    """Save config table in horizontal format (one row of headers, one row of values)."""
    keys = ["system", "seed", "n_states", "n_controls", "t1_initial", "fp_tol"]
    valid_keys = [k for k in keys if hasattr(config, k)]
    values = [getattr(config, k) for k in valid_keys]
    
    # Format values
    formatted_values = []
    for k, v in zip(valid_keys, values):
        if k == "system":
            formatted_values.append(str(v))
        elif k == "seed":
            formatted_values.append(str(int(v)))
        elif k in ["n_states", "n_controls"]:
            formatted_values.append(str(int(v)))
        elif k == "t1_initial":
            formatted_values.append(f"{v:.1f}")
        elif k == "fp_tol":
            formatted_values.append(f"{v:.0e}")
        else:
            formatted_values.append(str(v))
    
    full_path = _get_savepath(savepath, config)
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    
    # Format header names
    header_map = {
        "system": "system",
        "seed": "seed",
        "n_states": "$n$",
        "n_controls": "$m$",
        "t1_initial": "$t_1$",
        "fp_tol": "$\\epsilon$"
    }
    headers = [header_map.get(k, k) for k in valid_keys]
    
    with open(full_path, "w", encoding="utf-8") as f:
        n_cols = len(valid_keys)
        f.write(f"\\begin{{tabular}}{{@{{}}{'c' * n_cols}@{{}}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        f.write(" & ".join(formatted_values) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

def save_results_table(results_list, config=None, savepath="thesis/tables/results.tex"):
    """
    Reformats the flat results list into a professional Thesis table.
    Groups by Controller and stacks Standard/Hard datasets as rows.
    """
    import pandas as pd
    df = pd.DataFrame(results_list)
    
    # 1. Clean up the naming for the Thesis
    # Use LaTeX math mode for headers (e.g., $t_{conv}$)
    rename_map = {
        "Controller": "Method",
        "Dataset": "Data",
        "Stability (S)": "$S$",
        "t_conv (med)": "$t_{\text{conv}}$",
        "Avg Cost (J)": "$J_{\text{avg}}$"
    }
    df = df.rename(columns=rename_map)

    # 2. Reorder columns to put Method and Data first
    cols = ["Method", "Data", "$S$", "$t_{\text{conv}}$", "$J_{\text{avg}}$"]
    df = df[[c for c in cols if c in df.columns]]

    # 3. Format 'S' as a proper percentage if it's a float
    if "$S$" in df.columns:
        df["$S$"] = df["$S$"].apply(lambda x: f"{x:.0%}" if isinstance(x, (float, int)) else x)

    # 4. Save using our fixed LaTeX function
    full_path = _get_savepath(savepath, config)
    save_dataframe_latex(
        df, 
        full_path, 
        caption="Performance comparison of learned controllers vs. LQR baseline.", 
        label="tab:results"
    )

def save_data_summary_table(config, data, savepath="thesis/tables/data_summary.tex"):
    """Save data summary table in specific LaTeX format."""
    import numpy as np
    
    X_all = np.asarray(data["X"])
    t_all = np.asarray(data["t"]).reshape(-1)
    X_norms = config.norm(X_all).reshape(-1)
    
    n_traj = int(data.get("n_trajectories", -1))
    n_points = int(X_all.shape[1])
    t_min = t_all.min()
    t_max = t_all.max()
    abs_x_mean = np.mean(np.abs(X_all))
    abs_x_max = np.max(np.abs(X_all))
    norm_x_mean = np.mean(X_norms)
    norm_x_max = np.max(X_norms)
    
    full_path = _get_savepath(savepath, config)
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    
    with open(full_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{@{}cccccccc@{}}\n")
        f.write("\\toprule\n")
        f.write("trajectories & data points & $t$ (min) & $t$ (max) & $|x|$ (mean) & $|x|$ (max)& $||x||$ (mean)& $||x||$ (max)\\\\\n")
        f.write("\\midrule\n")
        f.write(f"{n_traj} & {n_points} & {t_min:.2f} & {t_max:.2f} & {abs_x_mean:.2f} & {abs_x_max:.2f} & {norm_x_mean:.2f} & {norm_x_max:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    
    # Return a DataFrame for compatibility
    return pd.DataFrame({
        "trajectories": [n_traj],
        "data points": [n_points],
        "t (min)": [t_min],
        "t (max)": [t_max],
        "|x| (mean)": [abs_x_mean],
        "|x| (max)": [abs_x_max],
        "||x|| (mean)": [norm_x_mean],
        "||x|| (max)": [norm_x_max]
    })

def save_params_table(obj, savepath, title="Configuration", keys=None, config=None):
    """Generic function to save any object's attributes to a LaTeX table."""
    if keys is None:
        keys = [k for k in obj.__dict__.keys() if not k.startswith('_')]
    
    data = {
        "Parameter": [str(k) for k in keys if hasattr(obj, k)],
        "Value": [getattr(obj, k) for k in keys if hasattr(obj, k)]
    }
    df = pd.DataFrame(data)
    
    full_path = _get_savepath(savepath, config)
    
    # Special handling for training config
    if "traincfg" in savepath.lower() or "train" in savepath.lower():
        _save_train_config_latex(obj, full_path)
    else:
        save_dataframe_latex(df, full_path, caption=title, label=f"tab:{Path(savepath).stem}")

def _save_train_config_latex(cfg, savepath):
    """Save training config in specific LaTeX format."""
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    
    def fmt_lr(x):
        """Format learning rate as LaTeX scientific notation."""
        if isinstance(x, (int, float)):
            if x >= 1e-3:
                return f"{x:.0f}"
            # Format as $5\times 10^{-4}$ style
            exp = int(np.floor(np.log10(x)))
            coeff = x / (10 ** exp)
            if abs(coeff - int(coeff)) < 1e-6:
                return f"${int(coeff)}\\times 10^{{{exp}}}$"
            return f"${coeff:.1f}\\times 10^{{{exp}}}$"
        return str(x)
    
    epochs_sup = getattr(cfg, 'sup_epochs', 1)
    epochs_unsup = getattr(cfg, 'unsup_epochs', 5)
    steps = getattr(cfg, 'unsup_n_steps', 70)
    horizon = getattr(cfg, 'horizon_initial', 30)
    lr = getattr(cfg, 'unsup_lr', 5e-4)
    batch_size = getattr(cfg, 'batch_size', None)
    batch_size_str = str(int(batch_size)) if batch_size is not None else "None"
    
    with open(savepath, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{@{}cccccc@{}}\n")
        f.write("\\toprule\n")
        f.write("epochs (sup) & epochs (unsup) & steps & rollout horizon & lr & batch\\_size  \\\\\n")
        f.write("\\midrule\n")
        f.write(f"{epochs_sup} & {epochs_unsup} & {steps} & {horizon} & {fmt_lr(lr)} & {batch_size_str}  \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def show_monte_carlo_results(results_dict, controller_name="Controller", title=None):
    """
    Display Monte Carlo evaluation results in a nice formatted table.
    
    Parameters
    ----------
    results_dict : dict or dict of dicts
        Output from simulation.monte_carlo() - can be single controller or multiple
    controller_name : str, optional
        Name if single controller (ignored if results_dict contains multiple)
    title : str, optional
        Title for the table
    """
    import numpy as np
    
    # Check if this is a dict of results (multiple controllers)
    if isinstance(results_dict, dict) and 'X0_pool' not in results_dict:
        # Multiple controllers - create comparison table
        rows = []
        for name, res in results_dict.items():
            rows.append(_compute_monte_carlo_stats(res, name))
        
        df = pd.DataFrame(rows)
    else:
        # Single controller
        df = _compute_monte_carlo_stats(results_dict, controller_name)
        df = pd.DataFrame([df])
    
    if title:
        display(Markdown(f"**{title}**"))
    
    display(df.style.hide(axis="index").set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f4f4f4'), ('color', 'black'), ('font-weight', 'bold')]}
    ]))
    
    return df

def _compute_monte_carlo_stats(results_dict, controller_name):
    """Helper to compute stats for a single controller."""
    import numpy as np
    
    init_dists = np.asarray(results_dict['init_dists'])
    final_dists = np.asarray(results_dict['final_dists'])
    NN_final_times = np.asarray(results_dict['NN_final_times'])
    NN_costs = np.asarray(results_dict['NN_costs'])
    
    converged_mask = np.isfinite(NN_final_times)
    n_converged = np.sum(converged_mask)
    n_total = len(NN_final_times)
    stability = n_converged / n_total if n_total > 0 else 0.0
    
    return {
        "Controller": controller_name,
        "Stability (S)": f"{stability:.1%}",
      #  "Converged / Total": f"{n_converged} / {n_total}",
      #  "Initial ||X|| (mean)": f"{np.mean(init_dists):.4f}",
        "Final ||X|| (mean)": f"{np.mean(final_dists[converged_mask]):.4f}" if n_converged > 0 else "N/A",
        "t_conv (median)": f"{np.median(NN_final_times[converged_mask]):.2f}" if n_converged > 0 else "N/A",
        "t_conv (mean)": f"{np.mean(NN_final_times[converged_mask]):.2f}" if n_converged > 0 else "N/A",
        "Cost J (median)": f"{np.median(NN_costs[converged_mask]):.4f}" if n_converged > 0 else "N/A",
        "Cost J (mean)": f"{np.mean(NN_costs[converged_mask]):.4f}" if n_converged > 0 else "N/A"
    }


def save_monte_carlo_results(results_dict, controller_name="Controller", config=None, 
                            savepath="thesis/tables/monte_carlo.tex"):
    """Save Monte Carlo results to LaTeX table with format: Model | Stability S | t_conv (mean) | Cost J (mean)."""
    import numpy as np
    
    if isinstance(results_dict, dict) and 'X0_pool' not in results_dict:
        rows = []
        for name, res in results_dict.items():
            rows.append(_compute_monte_carlo_stats_latex(res, name))
        df = pd.DataFrame(rows)
    else:
        stats = _compute_monte_carlo_stats_latex(results_dict, controller_name)
        df = pd.DataFrame([stats])
    
    full_path = _get_savepath(savepath, config)
    _save_monte_carlo_latex(df, full_path)
    return df

def _save_monte_carlo_latex(df, savepath):
    """Save Monte Carlo table in specific LaTeX format with toprule/midrule/bottomrule."""
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    
    def fmt_stability(x):
        if isinstance(x, (float, np.floating)) and not np.isnan(x):
            return f"{x*100:.1f}\\%"
        return "N/A"
    
    def fmt_float(x, decimals=2):
        if isinstance(x, (float, np.floating)) and not np.isnan(x):
            return f"{x:.{decimals}f}"
        return "N/A"
    
    def escape_tex(s):
        return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
    
    with open(savepath, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Model & Stability $S$ & $t_{\\text{conv}}$ (mean)  & Cost $J$ (mean) \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in df.iterrows():
            model = escape_tex(row["Model"])
            stability = fmt_stability(row["Stability $S$"])
            t_conv = fmt_float(row["$t_{\\text{conv}}$ (mean)"], decimals=2)
            cost = fmt_float(row["Cost $J$ (mean)"], decimals=4)
            f.write(f"{model} & {stability}  & {t_conv}  & {cost} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

def _compute_monte_carlo_stats_latex(results_dict, controller_name):
    """Helper for LaTeX formatting - returns only Model, Stability S, t_conv (mean), Cost J (mean)."""
    import numpy as np
    
    NN_final_times = np.asarray(results_dict['NN_final_times'])
    NN_costs = np.asarray(results_dict['NN_costs'])
    
    converged_mask = np.isfinite(NN_final_times)
    n_converged = np.sum(converged_mask)
    n_total = len(NN_final_times)
    stability = n_converged / n_total if n_total > 0 else 0.0
    
    t_conv_mean = np.mean(NN_final_times[converged_mask]) if n_converged > 0 else np.nan
    cost_mean = np.mean(NN_costs[converged_mask]) if n_converged > 0 else np.nan
    
    return {
        "Model": controller_name,
        "Stability $S$": stability,
        "$t_{\\text{conv}}$ (mean)": t_conv_mean,
        "Cost $J$ (mean)": cost_mean
    }