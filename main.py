# main.py

import argparse

from models.lstm.evaluation import run_lstm_multiprocess
from models.random_forest.evaluation import run_random_forest_multiprocess
from models.subexponential.evaluation import  run_subexponential_multiprocess
from models.subexponential_amortized.evaluation import run_subexponential_amortized_multiprocess
from models.svr.evaluation import  run_svr_multiprocess
from models.exponential.evaluation import run_exponential_multiprocess
from models.knn.evaluation import  run_knn_multiprocess
from models.autoarima.evaluation import run_autoarima
from models.ann.evaluation import run_mlp_multiprocess
from utils import get_dataset
import shutil


# shutil.rmtree('temp/', ignore_errors=True)


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

    if "exp" in args.models or "all" in args.models:
        run_exponential_multiprocess(full_dataset, 4)

    if "subexp" in args.models or "all" in args.models:
        run_subexponential_multiprocess(full_dataset, 4)

    if "subexp-amort" in args.models or "all" in args.models:
        run_subexponential_amortized_multiprocess(full_dataset, 4)

    if "svr" in args.models or "all" in args.models:
        run_svr_multiprocess(full_dataset, 4)

    if "autoarima" in args.models or "all" in args.models:
        run_autoarima(full_dataset, 4)

    if "knn" in args.models or "all" in args.models:
        run_knn_multiprocess(full_dataset, 4)

    if "ann" in args.models or "all" in args.models:
        run_mlp_multiprocess(full_dataset, 4)

    if "lstm" in args.models or "all" in args.models:
        run_lstm_multiprocess(full_dataset, 4)

    if "rf" in args.models or "all" in args.models:
        run_random_forest_multiprocess(full_dataset, 4)


if __name__ == "__main__":
    main()
