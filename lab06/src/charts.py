from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"




def build():
    raw = pd.read_csv(RESULTS / "raw_results.csv", sep=";", encoding="utf-8-sig")
    main = raw[(raw["run"] == "main") & (raw["error"].isna())]

    by_type = pd.read_csv(RESULTS / "summary_by_type.csv", sep=";", encoding="utf-8-sig").pivot(index="type", columns="backend", values="score")
    by_type.plot(kind="bar", ylim=(0, 2), rot=15, figsize=(8, 4.5))
    plt.ylabel("средняя оценка, 0-2"); plt.title("Качество по типам заданий"); plt.tight_layout()
    plt.savefig(RESULTS / "quality_by_type.png", dpi=150); plt.close()

    main.boxplot(column="latency_s", by="backend", figsize=(6, 4.5))
    plt.suptitle(""); plt.title("Распределение полной задержки"); plt.ylabel("с"); plt.tight_layout()
    plt.savefig(RESULTS / "latency_distribution.png", dpi=150); plt.close()

    main.boxplot(column="tokens_per_s", by="backend", figsize=(6, 4.5))
    plt.suptitle(""); plt.title("Скорость генерации"); plt.ylabel("токен/с"); plt.tight_layout()
    plt.savefig(RESULTS / "tokens_per_second.png", dpi=150); plt.close()


if __name__ == "__main__":
    build()
