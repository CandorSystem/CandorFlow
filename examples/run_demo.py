"""Thin wrapper script to run the CandorFlow demo."""

from candorflow.demo import run_demo, plot_results

if __name__ == "__main__":
    results = run_demo()
    plot_results(results)

