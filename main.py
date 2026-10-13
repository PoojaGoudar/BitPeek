import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from experiments.run_benchmarks import run_benchmark_experiment
from experiments.run_xai_analysis import run_explainability_study
from experiments.run_ablation_study import run_ablation_study

def main():
    start_time = time.time()
    print("=" * 80)
    print(" STARTING COMPLETE BITCOIN FORECASTING RESEARCH PIPELINE (END-TO-END)")
    print("=" * 80)

    # Phase 1: Benchmark Study Across 8 Models + SOTA
    print("\n>>> PHASE 1: RUNNING BENCHMARK EXPERIMENTS ON 8 BASELINES + PROPOSED SOTA <<<")
    results_df, trained_models, pipeline, data = run_benchmark_experiment()

    # Phase 2: Explainable AI (XAI) Suite
    print("\n>>> PHASE 2: RUNNING EXPLAINABLE AI (XAI) ANALYSIS (SHAP, ATTENTION, IG) <<<")
    shap_res, attn_matrix = run_explainability_study()

    # Phase 3: Ablation Study
    print("\n>>> PHASE 3: RUNNING ABLATION STUDY <<<")
    ablation_df = run_ablation_study()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"[+] ALL EXPERIMENTS COMPLETED SUCCESSFULLY IN {elapsed/60:.2f} MINUTES!")
    print("=" * 80)
    print("Generated Artifacts for Research Paper:")
    print(" - results/tables/benchmark_results.csv")
    print(" - results/tables/table1_benchmark.tex")
    print(" - results/tables/ablation_study_results.csv")
    print(" - results/tables/table2_ablation.tex")
    print(" - results/tables/shap_feature_importance.csv")
    print(" - results/tables/table_shap_importance.tex")
    print(" - results/figures/fig_benchmark_comparison.png")
    print(" - results/figures/fig_price_forecast.png")
    print(" - results/figures/fig_confusion_matrix.png")
    print(" - results/figures/fig_trading_pnl.png")
    print(" - results/figures/fig_attention_heatmap.png")
    print(" - results/figures/fig_feature_importance.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
