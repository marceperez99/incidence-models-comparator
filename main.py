# main.py

import argparse
import pandas as pd
from evaluation import graphing
from models.lstm.evaluation import run_lstm
from models.random_forest.evaluation import run_random_forest
from models.subexponential.evaluation import run_subexponential
from models.subexponential_amortized.evaluation import run_subexponential_amort
from models.svr.evaluation import run_svr
from models.exponential.evaluation import run_exponential_multiprocess
from models.knn.evaluation import run_knn
from models.ann.evaluation import run_ann
from models.autoarima.evaluation import run_autoarima

from utils import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Comparación de modelos de predicción de dengue")

    parser.add_argument("--models", nargs="+",
                        default=["svr", "rf", "knn", "ann", "lstm", "autoarima", "exp", "subexp", "subexp-amort"],
                        help="Lista de modelos a ejecutar (ej: --models svr ann rf)")

    parser.add_argument("--plot", action="store_true", help="Generar gráficos de comparación")
    parser.add_argument(
        "--case_types",
        nargs="*",  # 0 or more args → always a list
        default=[],  # even if flag is absent → empty list
        help="Niveles a analizar"
    )
    parser.add_argument("--level_names", nargs="*", default=[], help="Niveles a analizar")
    parser.add_argument("--diseases", nargs="*", default=[], help="Enfermedades a analizar")
    return parser.parse_args()


def main():
    args = parse_args()


    # 1. Cargar y preprocesar datos
    full_dataset = get_dataset("data/case_data_full.csv", args.level_names, args.diseases, args.case_types)
    # dengue_dataset = get_dataset("data/dengue_confirmado_central.csv")
    # chiku_dataset = get_dataset("data/chikungunya_confirmado_central.csv")
    print(full_dataset['id_proy'].unique())

    full_dataset = [grupo.copy() for _, grupo in full_dataset.groupby('id_proy')]

    if "ann" in args.models:
        full_dataset = get_dataset("data/case_data_full.csv")
        architectures = [
            [16, 8],
            [32, 16],
            [32, 16, 8],
            [64, 32],
            [64, 32, 16],
            [32, 32],
            [16],  # muy simple, buen baseline
            [128, 64, 32],  # más profunda, solo si no hay sobreajuste
        ]
        best_loss, y_true, y_pred, best_config = run_ann(full_dataset, 4, 4, architectures)
        graphing.plot_observed_vs_predicted(y_true, y_pred, f'plt_eval_ann', 'outputs/plots/ann/dengue', title='ANN',
                                            description='descripcion')
        graphing.plot_scatter(y_true, y_pred, f'plt_scatter_ann', 'Random Forest', title='ANN',
                              description='descripcion', output_dir=f'outputs/plots/ann/dengue')

    if "lstm" in args.models:
        full_dataset = get_dataset("data/case_data_full.csv.csv")

        run_lstm(full_dataset, test_id_proy="CENTRAL-DENGUE-CONFIRMADO", sequence_length=8, prediction_window=4,
                 epochs=200)

    if "rf" in args.models:
        run_random_forest([dengue_dataset, ], 4)

    if "svr" in args.models:
        run_svr([dengue_dataset, chiku_dataset], 4)

    if "autoarima" in args.models:
        run_autoarima([dengue_dataset, chiku_dataset], 4)

    if "exp" in args.models:
        run_exponential_multiprocess(full_dataset, 4)

    if "subexp" in args.models:
        run_subexponential([dengue_dataset, chiku_dataset], 4)

    if "subexp-amort" in args.models:
        run_subexponential_amort([dengue_dataset, chiku_dataset], 4)

    if "knn" in args.models:
        run_knn([dengue_dataset, chiku_dataset], 4)


if __name__ == "__main__":
    main()
