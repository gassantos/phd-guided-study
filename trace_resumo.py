# Importando as bibliotecas necessárias
import sys
import google.generativeai as genai
import nltk
import os
import pandas as pd
import re
import spacy
import tiktoken
import zipfile
import time

from datetime import datetime, timedelta
from docx import Document
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from unidecode import unidecode
import logging

# Configurar o logger
log_filename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
log_directory = 'log'
os.makedirs(log_directory, exist_ok=True)
log_path = os.path.join(log_directory, log_filename)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Função para logar exceções
def log_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_exception
logging.info("Iniciando a aplicação")

time_exec = time.time()

# Carregar variáveis de ambiente
load_dotenv()

# Baixar recursos do nltk se necessário
nltk.download('stopwords')
nltk.download('rslp')

# Carregar stopwords para português
STOPWORDS = set(stopwords.words('portuguese'))

def get_text_no_stopwords(text: str) -> list[str]:

    # Tokenização (separar palavras)
    tokens = text.split()

    # Remoção de stopwords
    tokens = [word for word in tokens if word not in STOPWORDS]
    return tokens

# Exemplo para obter datas com timedelta
def get_mondaydate_onweek():
    now = datetime.now()
    monday = now - timedelta(days = now.weekday())
    print(monday.date())
    logging.info(f"Data da Segunda-feira da Semana: {monday.date()}")

# Coloca os arquivos no diretório especificado
def extract_zip(zip_file_path, extract_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# Função de pré-processamento com lematização e stemming
def preprocess_text(text, use_lemma=False, use_stemm=False):
    # Carregar o modelo de lematização em português do spaCy
    nlp = spacy.load("pt_core_news_sm")

    # Carregar o stemmer RSLP para português
    stemmer = RSLPStemmer()

    # 1. Normalização (lowercase)
    text = text.lower()

    # 2. Remoção de acentuação (diacríticos)
    text = unidecode(text)  # Remove acentos como á, é, í

    # 3. Remoção de pontuação e caracteres especiais
    text = re.sub(r'[^a-z\s]', '', text)  # Remove tudo que não for letras

    # 4. Remoção de stopwords
    tokens = get_text_no_stopwords(text)

    # 5. Lematização (opcional)
    if use_lemma:
        # Usando spaCy para lematização
        doc = nlp(" ".join(tokens))  # Processa o texto tokenizado
        tokens = [token.lemma_ for token in doc]  # Substitui por lemmas

    # 6. Stemming (opcional)
    if use_stemm:
        # Usando RSLPStemmer para stemming
        tokens = [stemmer.stem(word) for word in tokens]  # Aplica o stemmer

    # 7. Remoção de múltiplos espaços
    processed_text = ' '.join(tokens)
    return processed_text



print(f"Iniciando a aplicação às {datetime.now():%Y-%m-%d %H:%M:%S}")
logging.info(f"Iniciando a aplicação às {datetime.now():%Y-%m-%d %H:%M:%S}")
inicio = time.time()

PATTERN_VOTO_RELATOR    = r"\barquivamento\b|\bcomunicação\b|encaminhamento\b"
PATTERN_STATUS_PARECER  = r"de acordo|parcialmente|desacordo"
PATTERN_PARECER_MPC_SGE = r"posiciono-me|de acordo|parcialmente|desacordo|Ministério Público de Contas|\
                            Parquet de Contas|Corpo Instrutivo|Corpo Técnico|Unidade Técnica|Relatório de Auditoria"

MODEL_GPT_35_TURBO = "gpt-3.5-turbo-0301"
MODEL_GOOGLE_GEMINI = "gemini-1.5-flash"

# os.environ['AZURE_OPENAI_KEY']= OPENAI_KEY
# openai.api_type = "azure"
# openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
# openai.api_key = os.getenv("AZURE_OPENAI_KEY")

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

def get_text_pdf(filename: str):
    """Function to extract text from PDF"""
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    text=""
    for page in pages:
        text += page.page_content
    return text

def get_text_doc(filename: str):
    """Function to extract text from Word Document"""
    f = open(filename, 'rb')
    doc = Document(f)
    f.close()

    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text+'\n'
    return text


def get_len_tokens(text:str) -> int:
    """Counting Tokens for Summarization Models"""
    encoding = tiktoken.encoding_for_model(MODEL_GPT_35_TURBO)
    # print(f"Were found {tokens} tokens in Document file of the {filename}.")
    logging.info(f"Quantidade de tokens no texto: {len(encoding.encode(text))}")
    return len(encoding.encode(text))

def get_length_resumo(text:str) -> int:
    """Retorna a quantidade proporcional de palavras em 20% do texto"""
    length_resumo = get_len_tokens(text)
    if length_resumo < 500:
        return 100
    else:
        return int(length_resumo * 0.2)


def get_last_parecer(resumos: list) -> list:
    """Gera os último parágrafo de cada resumo."""
    return [str(list(nlp(resumo).sents)[-1]) for resumo in resumos]


def get_parecer_mpinstrutivo(texto: str) -> str:
    """Gera os pareceres do Ministério Público de Contas."""
    doc = nlp(texto)
    parecer = set()
    for sent in doc.sents:
        matches = re.findall(PATTERN_PARECER_MPC_SGE, sent.text, flags=re.IGNORECASE)
        matching_sentences = [re.search(r".*?{}.*?".format(match), sent.text).group(0) for match in matches]
        for sentence in matching_sentences:
            if 'Ministério Público de Contas' in sentence or 'Ministério Público Especial' in sentence \
                or 'Corpo Instrutivo' in sentence or 'Corpo Técnico' in sentence:
                    parecer.add(sentence)

    return parecer

def get_decisao_voto(texto: str) -> list:
    """Extrai o conteúdo da decisão de voto do Relator."""
    decisao = set()
    matches = re.findall(PATTERN_VOTO_RELATOR, texto, flags=re.IGNORECASE)  # Case-insensitive matching
    matching_sentences = [re.search(r".*?{}.*?".format(match),
                                    texto).group(0)
                                    for match in matches] # get all matches in thoses sents with
    for sentence in matching_sentences:
        if 'arquivamento' in sentence.lower():
            decisao.add('arquivamento')
        if 'comunicação' in sentence.lower():
            decisao.add('comunicação')
        if 'encaminhamento' in sentence.lower():
            decisao.add('encaminhamento.')
        if 'determinação' in sentence.lower():
            decisao.add('determinação')

    return decisao


def get_status_parecer(pareceres: list) -> str:
    """Status de parecer do Ministério Público de Contas e Corpo Técnico"""
    status = set()
    for parecer in pareceres:
        parecer = parecer.lower()
        if 'desacordo' in parecer:
            if 'parcialmente'  in parecer:
                status.add('parcialmente desacordo')
            else:
                status.add('desacordo')
        else:
            if 'de acordo' in parecer:
                if 'parcialmente'  in parecer:
                    status.add('parcialmente de acordo')
                else:
                    status.add('de acordo')

    return '; '.join(status)


# Use spacy.cli.download("pt_core_news_sm") for automatic download
# If make a download locally, you need to run:
# ```pip install pt_core_news_sm-3.7.0.tar.gz``` and
# the next step, execute spacy.load().

nlp = spacy.load("pt_core_news_sm") # Load the model for Portuguese language

def get_response(text:str, prompt:str = '') -> str:
    """
    Retorna o texto gerado com base no prompt de LLM executado na tarefa atribuída.

    This method supports multi-turn chats but is **stateless**: the entire conversation history needs to be sent with each
    request. This takes some manual management but gives you complete control:

    >>> messages = [{'role':'user', 'parts': ['hello']}]
    >>> response = model.generate_content(messages) # "Hello, how can I help"
    >>> messages.append(response.candidates[0].content)
    >>> messages.append({'role':'user', 'parts': ['How does quantum physics work?']})
    """

    length_text = get_length_resumo(text)
    if 'resumo' in prompt:
        conversation=[{"role": "user", "parts": f"Você é um especialista da área de controle externo. \
                                                Resuma o voto do processo com decisão e parecer em {length_text} palavras"}]
    else:
        conversation=[{"role": "user", "parts": f"Identifique os dispositivos de lei, quando existir."}]
    conversation.append({"role": "user", "parts": text})


    response = model.generate_content(conversation, request_options={"timeout": 600})
    # print(response.usage_metadata)
    logging.info(f"{response.usage_metadata}")

    doc = nlp(response.text)
    return ' '.join([sent.text for sent in doc.sents])


# Obtaining all docs from that session date
docs = pd.read_csv('data/doc_votos.csv', sep=';', encoding= 'utf-8')
# docs = pd.read_excel('data/doc_representacao.xlsx', engine='openpyxl')

# ordenando os votos pelo maiores texto
docs_sort = docs.sort_values(by='LENGTH', ascending=False)
docs_sort.reset_index(drop=True, inplace=True)
print(docs_sort)
logging.info(f"Amostra dos dados de documentos: {docs_sort}")

# Selecionando o TEXTO conforme ASSUNTO: APOSENTADORIA
df_aposentadoria = docs_sort[docs_sort['TEXTO'].str.contains('ASSUNTO: APOSENTADORIA', case=False, na=False)]
docs_aposentadoria = df_aposentadoria.set_index('PROCESSO')['TEXTO'].to_dict()
aposentadoria_10_itens = dict(list(docs_aposentadoria.items()))

# Selecionando o TEXTO conforme ASSUNTO: REPRESENTAÇÃO
df_representacao = docs_sort[docs_sort['TEXTO'].str.contains('ASSUNTO: REPRESENTAÇÃO', case=False, na=False)]
docs_representacao = df_representacao.set_index('PROCESSO')['TEXTO'].to_dict()
representacao_10_itens = dict(list(docs_representacao.items()))

# Carregando todos os documentos
df_docs = docs_sort.set_index('PROCESSO')['TEXTO'].to_dict()
df_docs = dict(list(df_docs.items()))
print(len(list(df_docs.items())))
input("Pressione Enter para continuar...")

# Resumo de Votos Estruturado
processos, resumos, entidades, pareceres, pareceres_mpc, pareceres_instrutivo, decisoes = [],[],[],[],[],[],[]
docs, tokens_docs, tokens_resumo, tokens_lemma, docs_lemma, tokens_stemmer, tokens_resumo_lemma, tokens_resumo_stemmer,docs_stemmer, resumos_lemma, resumos_stemmer  = [],[],[],[],[],[],[],[],[],[],[]

# Processar os primeiros 10 itens
start_inicio = time.time()
for processo in df_docs.keys():
    print("\nProcessando o documento do Processo Nº ",processo)
    documento = df_docs.get(processo)
    num_processo = processo
    tokens = get_len_tokens(documento)
    if tokens <= 131072:
        print("\nO documento do Processo Nº "+processo+" contém ", tokens)
        logging.info(f"O documento do Processo Nº {processo} contém {tokens} tokens.")
        processos.append(num_processo)
        docs.append(documento)
        tokens_docs.append(tokens)
        resumos.append(get_response(documento, 'resumo'))
        tokens_resumo.append(get_len_tokens(get_response(documento, 'resumo')))

        docs_lemma.append(preprocess_text(documento, use_lemma=True))
        tokens_lemma.append(get_len_tokens(preprocess_text(documento, use_lemma=True)))
        resumos_lemma.append(get_response(preprocess_text(documento, use_lemma=True), 'resumo'))
        tokens_resumo_lemma.append(get_len_tokens(get_response(preprocess_text(documento, use_lemma=True), 'resumo')))

        docs_stemmer.append(preprocess_text(documento, use_stemm=True))
        tokens_stemmer.append(get_len_tokens(preprocess_text(documento, use_stemm=True)))
        resumos_stemmer.append(get_response(preprocess_text(documento, use_stemm=True), 'resumo'))
        tokens_resumo_stemmer.append(get_len_tokens(get_response(preprocess_text(documento, use_stemm=True), 'resumo')))

        entidades.append(get_response(documento,''))
        parecer = get_parecer_mpinstrutivo(documento)
        pareceres.append(parecer)
        status_parecer = get_status_parecer(parecer)
        pareceres_mpc.append(status_parecer)
        pareceres_instrutivo.append(status_parecer)
        decisoes.append(get_decisao_voto(documento))
        print(f"O documento do Processo Nº {processo}, foi processado em {time.time() - start_inicio:.2f} segundos.")
        logging.info(f"O documento do Processo Nº {processo}, foi processado em {time.time() - start_inicio:.2f} segundos.")
    else:
        msg = f"O documento do Processo Nº {processo} está superior ao limite de 128K tokens {tokens} permitido pelo modelo {MODEL_GOOGLE_GEMINI}"
        print(msg)
        logging.info(msg)

print(f"\nProcedimento de sumarização finalizado às {datetime.now()} com tempo de execução de {time.time() - inicio:.2f} segundos.\n")
logging.info(f"Procedimento de sumarização finalizado às {datetime.now()} com tempo de execução de {time.time() - inicio:.2f} segundos.\n")

votos = pd.DataFrame()
votos['Processo'] = processos
votos['Texto'] = docs
votos['Texto_Tokens'] = tokens_docs
votos['Resumo'] = resumos
votos['Texto_Tokens_Resumo'] = tokens_resumo

votos['Texto_Stemmer'] = docs_stemmer
votos['Texto_Tokens_Stemmer'] = tokens_stemmer
votos['Resumo_Stemmer'] = resumos_stemmer
votos['Texto_Tokens_Resumo_Stemmer'] = tokens_resumo_stemmer

votos['Texto_Lemma'] = docs_lemma
votos['Texto_Tokens_Lemma'] = tokens_lemma
votos['Resumo_Lemma'] = resumos_lemma
votos['Texto_Tokens_Resumo_Lemma'] = tokens_resumo_lemma

votos['Legislação'] = entidades
votos['Ministério Público de Contas'] = pareceres_mpc
votos['Corpo Instrutivo'] = pareceres_instrutivo
votos['Parecer'] = pareceres
votos['Decisão'] = decisoes


# Salvar o resumo dos votos
votos.to_csv('data/resumo_sumaria.csv', sep=';', encoding='utf-8', index=False)
print(f"Finalizando a aplicação às {datetime.now():%Y-%m-%d %H:%M:%S} com tempo de execução de {time.time() - time_exec:.2f} segundos.")
logging.info(f"Finalizando a aplicação às {datetime.now():%Y-%m-%d %H:%M:%S} com tempo de execução de {time.time() - time_exec:.2f} segundos.")
sys.exit(0)
