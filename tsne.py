"""
Clustering sobre os Embeddigns (Resumo com Stemmer/Lemma)
Os embeddings serão avaliados com a mesmo modelo de representação numérica para fins de comparação.
"""

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns



# # Contagem de Pareceres
# grupos_embeddings.groupby('Parecer').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
# plt.gca().spines[['top', 'right',]].set_visible(False)

# grupos_embeddings['Parecer'].value_counts()

def get_tsne_embeddings(vetor_stemmer, grupos_embeddings: pd.DataFrame) -> pd.DataFrame:
    """Generates t-SNE embeddings for the given data and returns a DataFrame with the results.
    This function performs the following steps:
    1. Initializes a t-SNE object with 2 components and a fixed random state for reproducibility.
    2. Fits the t-SNE model to the input data `X_resumo_stemmer` and transforms it.
    3. Creates a DataFrame with the t-SNE results, naming the columns 'TSNE1' and 'TSNE2'.
    4. Adds a 'Parecer' column to the DataFrame from the `grupos_embeddings` DataFrame.
    Returns:
        pd.DataFrame: A DataFrame containing the t-SNE embeddings with columns 'TSNE1', 'TSNE2', and 'Parecer'.
    """

    tsne = TSNE(n_components=2, random_state=42)
    tsne_resumo_stemmer = tsne.fit_transform(vetor_stemmer)
    df_tsne_resumo_stemmer = pd.DataFrame(tsne_resumo_stemmer, columns=['TSNE1', 'TSNE2'])
    df_tsne_resumo_stemmer['Parecer'] = grupos_embeddings['Parecer']
    df_tsne_resumo_stemmer
    return df_tsne_resumo_stemmer


def plot_tsne_scatter(df_tsne_resumo_stemmer):
    fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
    sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
    sns.scatterplot(data=df_tsne_resumo_stemmer, x='TSNE1', y='TSNE2', hue='Parecer', palette='hls')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title('Scatter plot of news using t-SNE');
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.axis('equal')
    fig.show()


df_tsne_resumo_stemmer = get_tsne_embeddings()
plot_tsne_scatter(df_tsne_resumo_stemmer)
