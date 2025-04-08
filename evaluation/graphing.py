from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


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
        y_true: List[float],
        y_pred: List[float],
        filename: str,
        output_dir: str = 'outputs/plots',
        title: str = '',
        description: str = ""
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_true, label='Observado')
    ax.plot(y_pred, label='Predicho')

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
    ax.scatter(y_true, y_pred, alpha=0.7, label=model_name)
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
