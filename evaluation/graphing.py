from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from constants import LOSS_METRIC

observed_color = '#F08700'  # azul noche
predicted_color = '#457B9D'  # rojo suave


def plot_mae_bar(model_names: List[str], mae_scores: List[float], output_dir: str = 'plots') -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(model_names, mae_scores, color='skyblue')
    plt.title('Comparación de modelos - MAE')
    plt.ylabel('MAE')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/barplot_mae.png')
    plt.close()


def plot_observed_vs_predicted(
        x,
        y_true: List[float],
        y_pred: List[float],
        filename: str,
        output_dir: str = 'outputs/plots',
        title: str = '',
        description: str = ""
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_true, label='Observado', color=observed_color)
    ax.plot(x, y_pred, label='Predicho', color=predicted_color)

    ax.set_title(title)
    ax.set_xlabel('Semana')
    ax.set_ylabel('Nro de Casos')
    ax.legend()
    ax.grid(True)

    if description:
        # Colocar descripción en el pie del gráfico (fuera del área de ejes)
        fig.text(0.5, 0.01, description, wrap=True,
                 ha='center', va='bottom', fontsize=9)

    # Ajustar para que haya espacio suficiente para el texto
    fig.subplots_adjust(bottom=0.2)

    fig.savefig(f'{output_dir}/{filename}.png')
    plt.close(fig)


def plot_scatter(y_true: List[float], y_pred: List[float], filename: str, model_name: str,
                 output_dir: str = 'outputs/plots', title: str = "", description: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)




    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7, label=model_name, color=predicted_color)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2)

    ax.set_xlabel('Observado')
    ax.set_ylabel('Predicho')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if description:
        fig.text(0.5, 0.01, description, ha='center', fontsize=9, wrap=True)

    fig.subplots_adjust(bottom=0.2)
    fig.savefig(f'{output_dir}/{filename}.png')
    plt.close(fig)


def plot_boxplot_errors(error_dict: Dict[str, List[float]], output_dir: str = 'plots') -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=pd.DataFrame(error_dict))
    plt.title("Distribución del error absoluto por modelo")
    plt.ylabel("Error absoluto")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplot_errores.png')
    plt.close()


def plot_metrics_heatmap(model_names: List[str], mae_scores: List[float], rmse_scores: List[float],
                         output_dir: str = 'plots') -> None:
    os.makedirs(output_dir, exist_ok=True)
    df_metrics = pd.DataFrame({
        'Model': model_names,
        'MAE': mae_scores,
        'RMSE': rmse_scores
    }).set_index('Model')

    plt.figure(figsize=(8, 4))
    sns.heatmap(df_metrics, annot=True, cmap='Blues', fmt=".2f")
    plt.title("Métricas por modelo")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_metricas.png')
    plt.close()


import matplotlib.pyplot as plt
import os


import matplotlib.pyplot as plt
import os
import pandas as pd

def graficar_predicciones(
    dataset,
    predictions,
    pie_dict=None,
    method='exponential'
):
    """
    Genera y guarda un gráfico de comparación entre casos observados y predicciones.

    Parámetros:
    - dataset: DataFrame con las columnas 't', 'date', 'i_cases'
    - predictions: lista de tuplas (n_semanas, x_t_values, y), donde x_t_values contiene valores de 't'
    - pie_dict: dict (opcional), claves/valores para agregar como pie del gráfico
    """
    plt.figure(figsize=(10, 5))

    # Aseguramos que las fechas sean datetime
    dataset['date'] = pd.to_datetime(dataset['date'])

    # Creamos un mapa de t -> date
    t_to_date = dict(zip(dataset['t'], dataset['date']))

    # Determinar rango para la serie observada
    t_min = min(x[0] for _, x, _ in predictions)
    t_max = max(x[-1] for _, x, _ in predictions)
    mask = (dataset['t'] >= t_min) & (dataset['t'] <= t_max)
    dataset_recortado = dataset[mask]

    # Serie observada
    plt.plot(dataset_recortado['date'], dataset_recortado['i_cases'], label='Observado')

    # Predicciones
    for semana, x_t, y in predictions:
        x_dates = [t_to_date.get(ti, pd.NaT) for ti in x_t]
        plt.plot(x_dates, y, label=f'Predicción a {semana} semanas')

    plt.title("Número de casos predichos")
    plt.xlabel("Fecha")
    plt.ylabel("Número de casos")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Pie de imagen (caption)
    if pie_dict:
        pie_texto = ', '.join([f"{k}: {v}" for k, v in pie_dict.items()])
        plt.figtext(0.5, -0.07, pie_texto, wrap=True, horizontalalignment='center', fontsize=10)

    disease = dataset['disease'].iloc[0].lower()
    level = dataset['name'].iloc[0].lower()
    classification = dataset['classification'].iloc[0].lower()
    directory = f'outputs/plots/{LOSS_METRIC}/{method}/{disease}/{level}/{classification}'
    os.makedirs(directory, exist_ok=True)
    archivo_salida = f'{directory}/nro_de_casos.png'

    plt.savefig(archivo_salida, bbox_inches='tight')
    plt.close()
