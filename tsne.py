"""
Clustering sobre os Embeddigns (Resumo com Stemmer/Lemma)
Os embeddings serão avaliados com a mesmo modelo de representação numérica para fins de comparação.
"""

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

# GEMINI_EMBEDD = "models/text-embedding-004"
# ADA_EMBEDD = "text-embedding-ada-002"

# def get_embeddings(text: str, chunk_size: int = 2048):  # 2K characters to avoid exceeding limit
#     """Gera embeddings por texto preprocessado"""

#     encoding = tiktoken.encoding_for_model(ADA_EMBEDD)
#     num_tokens = len(encoding.encode(text))

#     # Quebra em chunks, se maior que chunk_size
#     if num_tokens > chunk_size:
#         chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
#         embeddings = []
#         for chunk in chunks:
#             result = genai.embed_content(
#                 model=GEMINI_EMBEDD,
#                 content=chunk
#             )
#             embeddings.extend(result['embedding'])  # Estende a lista de embeddings com resultado por chunk
#         return embeddings
#     else:
#         # Process text as usual if it's within the size limit
#         result = genai.embed_content(
#             model=GEMINI_EMBEDD,
#             content=text
#         )
#         return result['embedding']


# grupos_embeddings = pd.DataFrame()
# grupos_embeddings['Processo'] = processos
# grupos_embeddings['Doc_Tokens'] = tokens_docs
# grupos_embeddings['Resumo_Tokens'] = tokens_resumo
# grupos_embeddings['Resumo_Tokens_Stemmer'] = tokens_resumo_stemmer
# grupos_embeddings['Resumo_Tokens_Lemma'] = tokens_resumo_lemma
# grupos_embeddings['Parecer'] = pareceres_instrutivo


# resumo_embed, resumo_embed_stemmer, resumo_embedd_lemma = [],[],[]

# for index, row in votos.iterrows():
#     resumo_embed.append(get_embeddings(row['Resumo']))
#     resumo_embed_stemmer.append(get_embeddings(row['Resumo_Stemmer']))
#     resumo_embedd_lemma.append(get_embeddings(row['Resumo_Lemma']))

# grupos_embeddings['Resumo_Embed'] = resumo_embed
# grupos_embeddings['Resumo_Embed_Stemmer'] = resumo_embed_stemmer
# grupos_embeddings['Resumo_Embed_Lemma'] = resumo_embedd_lemma

# grupos_embeddings.to_csv('data/grupos_embeddings.csv', sep=';', encoding='utf-8', index=False)


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
