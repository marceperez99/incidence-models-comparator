# main.py

import argparse
import pandas as pd

from models.subexponential.evaluation import run_subexponential
from models.subexponential_amortized.evaluation import run_subexponential_amort
from models.svr.evaluation import run_svr
from models.exponential.evaluation import run_exponential
from models.knn.evaluation import run_knn

from utils import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Comparación de modelos de predicción de dengue")

    parser.add_argument("--models", nargs="+",
                        default=["svr", "rf", "knn", "ann", "lstm", "sarima", "exp", "subexp", "subexp-amort"],
                        help="Lista de modelos a ejecutar (ej: --models svr ann rf)")

    parser.add_argument("--plot", action="store_true", help="Generar gráficos de comparación")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Cargar y preprocesar datos
    dengue_dataset = get_dataset("data/dengue_confirmado_central.csv")
    chiku_dataset = get_dataset("data/chikungunya_confirmado_central.csv")

    # 2. Ejecutar modelos seleccionados
    results = []

    if "svr" in args.models:
        run_svr([dengue_dataset, ], 4)

    if "rf" in args.models:
        pass


    if "ann" in args.models:
        full_dataset = pd.read_csv("data/case_data_full.csv")
        # run_ann(full_dataset)

    if "lstm" in args.models:
        full_dataset = pd.read_csv("data/case_data_full.csv")
        # run_lstm(full_dataset)

    if "sarima" in args.models:
        pass

    if "exp" in args.models:
        run_exponential([dengue_dataset, chiku_dataset], 4)
    if "subexp" in args.models:
        run_subexponential([dengue_dataset, chiku_dataset], 4)
    if "subexp-amort" in args.models:
        run_subexponential_amort([dengue_dataset, chiku_dataset], 4)
    if "knn" in args.models:
        run_knn([dengue_dataset, chiku_dataset], 4)



if __name__ == "__main__":
    main()
