import os
import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display, Markdown

def _get_savepath(savepath, config):
    if config and hasattr(config, "system") and "tables" in savepath:
        p = Path(savepath)
        if config.system not in p.parts:
            parts = list(p.parts)
            try:
                idx = parts.index("tables")
                parts.insert(idx + 1, str(config.system))
                return str(Path(*parts))
            except ValueError: pass
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
    keys = ["system", "seed", "n_states", "n_controls", "t1_initial", "fp_tol"]
    data = {"Parameter": [k for k in keys if hasattr(config, k)],
            "Value": [getattr(config, k) for k in keys if hasattr(config, k)]}
    save_dataframe_latex(pd.DataFrame(data), _get_savepath(savepath, config), 
                         caption="Problem configuration.", label="tab:config")

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
    """
    Extracts statistics from the data dictionary and saves a summary table.
    """
    import numpy as np
    
    X_all = np.asarray(data["X"])
    t_all = np.asarray(data["t"]).reshape(-1)
    X_norms = config.norm(X_all).reshape(-1)

    stats = {
        "Metric": ["Trajectories", "Total Points", "t min/max", "|X| mean/max", "||X|| mean/max"],
        "Value": [
            int(data.get("n_trajectories", -1)),
            int(X_all.shape[1]),
            f"{t_all.min():.2f} / {t_all.max():.2f}",
            f"{np.mean(np.abs(X_all)):.2f} / {np.max(np.abs(X_all)):.2f}",
            f"{np.mean(X_norms):.2f} / {np.max(X_norms):.2f}"
        ]
    }
    df_stats = pd.DataFrame(stats)
    save_dataframe_latex(df_stats, _get_savepath(savepath, config), 
                         caption="Dataset numerical summary.", label="tab:data_summary")

    return df_stats

def save_params_table(obj, savepath, title="Configuration", keys=None, config=None):
    """
    Generic function to save any object's attributes to a LaTeX table.
    """
    # If no keys provided, try to get all non-private attributes
    if keys is None:
        keys = [k for k in obj.__dict__.keys() if not k.startswith('_')]
    
    data = {
        "Parameter": [str(k) for k in keys if hasattr(obj, k)],
        "Value": [getattr(obj, k) for k in keys if hasattr(obj, k)]
    }
    df = pd.DataFrame(data)
    
    # Use the system-specific path if config is provided
    full_path = _get_savepath(savepath, config)
    
    save_dataframe_latex(df, full_path, caption=title, label=f"tab:{Path(savepath).stem}")


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
    """
    Save Monte Carlo evaluation results to a LaTeX table file.
    
    Parameters
    ----------
    results_dict : dict
        Output from simulation.monte_carlo()
    controller_name : str
        Name of the controller being evaluated
    config : config object, optional
        Used for system-specific path handling
    savepath : str
        Path to save the LaTeX table
    """
    import numpy as np
    
    if isinstance(results_dict, dict) and 'X0_pool' not in results_dict:
        # Multiple controllers - create comparison table
        rows = []
        for name, res in results_dict.items():
            rows.append(_compute_monte_carlo_stats_latex(res, name))
        df = pd.DataFrame(rows)
        caption = "Monte Carlo evaluation results comparing multiple controllers."
    else:
        # Single controller
        stats = _compute_monte_carlo_stats_latex(results_dict, controller_name)
        df = pd.DataFrame([stats])
        caption = f"Monte Carlo evaluation results for {controller_name}."
    
    full_path = _get_savepath(savepath, config)
    save_dataframe_latex(df, full_path, caption=caption, label="tab:monte_carlo")
    return df

def _compute_monte_carlo_stats_latex(results_dict, controller_name):
    """Helper for LaTeX formatting."""
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
        "Stability ($S$)": f"{stability:.1%}",
     #   "Converged / Total": f"{n_converged} / {n_total}",
     #   "Initial $\\|X\\|$ (mean)": f"{np.mean(init_dists):.4f}",
        "Final $\\|X\\|$ (mean)": f"{np.mean(final_dists[converged_mask]):.4f}" if n_converged > 0 else "N/A",
        "$t_{\\text{conv}}$ (median)": f"{np.median(NN_final_times[converged_mask]):.2f}" if n_converged > 0 else "N/A",
        "$t_{\\text{conv}}$ (mean)": f"{np.mean(NN_final_times[converged_mask]):.2f}" if n_converged > 0 else "N/A",
        "Cost $J$ (median)": f"{np.median(NN_costs[converged_mask]):.4f}" if n_converged > 0 else "N/A",
        "Cost $J$ (mean)": f"{np.mean(NN_costs[converged_mask]):.4f}" if n_converged > 0 else "N/A"
    }