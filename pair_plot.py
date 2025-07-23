import seaborn as sns
import matplotlib.pyplot as plt
from utils import clean_csv

def create_pairplot(data):
    """
    Crée et affiche un pairplot Seaborn des caractéristiques numériques du DataFrame, 
    coloré par maison de Hogwarts, avec une mise en forme personnalisée.

    Le pairplot affiche les relations bivariées entre toutes les colonnes numériques,
    en utilisant la colonne 'Hogwarts House' pour colorer les points par maison.
    L'histogramme est affiché sur la diagonale. La légende est déplacée en bas 
    du graphique, et les labels d'axe sont formatés pour une meilleure lisibilité.
    Le graphique est également sauvegardé dans un fichier PNG.

    Args:
        data (pandas.DataFrame): DataFrame contenant les colonnes numériques des caractéristiques 
                                 et une colonne 'Hogwarts House' indiquant la maison de chaque élève.

    Returns:
        None: La fonction affiche le graphique et sauvegarde l'image, mais ne retourne rien.

    Side effects:
        - Affiche un pairplot avec matplotlib.
        - Sauvegarde le pairplot dans le fichier "pairplot_houses.png".
    """
    colors = {'Gryffindor': 'red', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue', 'Slytherin': 'green'}

    sns.set(style="white")

    pair_plot = sns.pairplot(data, hue='Hogwarts House', palette=colors,
                             diag_kind="hist", height=2, plot_kws={'alpha': 0.6, 'edgecolor': 'k', 's': 35})

    pair_plot.fig.suptitle('Pair Plot des Caractéristiques par Maison', y=1.02, fontsize=16, weight='bold')

    for i, ax_row in enumerate(pair_plot.axes):
        for j, ax in enumerate(ax_row):
            if ax is not None:
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.grid(False)

                if i < len(pair_plot.axes) - 1:
                    ax.set_xticks([])
                    ax.set_xticklabels([])

                if j > 0:
                    ax.set_yticks([])
                    ax.set_yticklabels([])

                if j == 0:
                    ylabel = ax.get_ylabel()
                    ylabel_multi = ylabel.replace(' ', '\n')
                    ax.set_ylabel(ylabel_multi, rotation=0, labelpad=50, fontsize=9, va='center')

    handles, labels = pair_plot.axes[0, 0].get_legend_handles_labels()
    pair_plot._legend.remove()
    pair_plot.fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=11)

    pair_plot.fig.subplots_adjust(bottom=0.15, top=0.95, wspace=0.8, hspace=0.8)

    pair_plot.savefig("pairplot_houses.png", dpi=300, bbox_inches='tight')

    plt.show()

def main(filename):
    data = clean_csv(filename)
    create_pairplot(data)

if __name__ == "__main__":
    filename = 'dataset_train.csv'
    main(filename)
