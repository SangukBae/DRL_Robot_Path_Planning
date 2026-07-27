"""Post-hoc analysis / plotting utilities (canonical home of the flat legacy
``scripts/utils/{aggregate_results,analyze_*,aux_ablation_summary,
check_reproducibility,plot_*,sim_validation_summary}.py`` modules).

Pure offline tooling: reads CSV/JSON run artifacts under ``runtime/`` and
produces tables/plots. No ROS/torch dependency beyond what a given script
already needed (e.g. plot_trajectories_on_map.py uses matplotlib).
"""
