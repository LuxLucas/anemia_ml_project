import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def donut_chart(
    df,
    coluna,
    mapa_nomes=None,
    titulo_geral=None,
    xlabel="Categoria",
    ylabel="Quantidade de Pacientes",
    paleta="viridis",
    rotacao_x=15,
):
    serie = df[coluna].map(mapa_nomes) if mapa_nomes else df[coluna]

    contagens = serie.value_counts()
    ordem = contagens.index
    porcentagens = (contagens / contagens.sum()) * 100
    cores = sns.color_palette(paleta, n_colors=len(ordem))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))

    if titulo_geral:
        fig.suptitle(titulo_geral, fontsize=16, fontweight="bold", y=1.02)

    sns.countplot(
        x=serie,
        hue=serie,
        palette=cores,
        order=ordem,
        legend=False,
        ax=ax1,
    )

    for container in ax1.containers:
        ax1.bar_label(container, padding=3)

    ax1.set_title("Distribuição (Quantidade)", fontsize=14)
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.tick_params(axis="x", rotation=rotacao_x)

    wedges, _ = ax2.pie(
        contagens.values,
        colors=cores,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
    )

    posicoes_y_usadas = []

    for i, wedge in enumerate(wedges):
        angulo = np.deg2rad((wedge.theta2 + wedge.theta1) / 2.0)
        x = np.cos(angulo)
        y = np.sin(angulo)

        lado = 1 if x >= 0 else -1
        alinhamento = "left" if lado == 1 else "right"

        x_texto = 1.35 * lado
        y_texto = 1.15 * y

        for y_ant, lado_ant in posicoes_y_usadas:
            if lado == lado_ant and abs(y_texto - y_ant) < 0.16:
                y_texto = y_ant + (0.16 if y_texto >= y_ant else -0.16)

        posicoes_y_usadas.append((y_texto, lado))

        texto = f"{ordem[i]} ({porcentagens.iloc[i]:.1f}%)"

        ax2.annotate(
            texto,
            xy=(x, y),
            xytext=(x_texto, y_texto),
            ha=alinhamento,
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#333333",
            arrowprops=dict(
                arrowstyle="-",
                color=cores[i],
                lw=1.5,
                connectionstyle="angle,angleA=0,angleB=90,rad=5",
            ),
        )

    ax2.set_xlim(-2.0, 2.2)
    ax2.set_ylim(-1.4, 1.4)
    ax2.set_title("Proporção (%)", fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()
