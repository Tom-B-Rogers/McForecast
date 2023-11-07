from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from language_tool_python import LanguageTool
import pandas as pd
import numpy as np
import nltk
import PyPDF2
import re
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import Counter

annual_report_path = '/Users/tomrogers/Desktop/InvestorPresentations/'
rba_path = "/Users/tomrogers/Desktop/Python/Refinitiv/Example.DataLibrary.Python-main/Configuration/rba_minutes.txt"

nlp = spacy.load('en_core_web_sm')

rbaminutes = pd.read_csv(rba_path, sep="\t", header=None)
rbaminutes.rename({0: "Content"}, axis=1, inplace=True)


def readfiles(path):

    os.chdir(path)
    pdfs = []
    for file in glob.glob("*.pdf"):
        pdfs.append(path+file)
    return pdfs


def cleanText(dictionary):

    for key, value in dictionary.items():

        text = value.replace('\n', ' ').replace('\xa0', ' ')
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s.,0-9]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()

        dictionary[key] = text

    return dictionary


def findTerms(text, terms, fraction_required=1.0, all_terms_required=False):
    """
    This function checks if a certain fraction of terms are present in a text. 

    :param text: The text to search.
    :param terms: A list of terms to search for. If 'all_terms_required' is set to True, this should be a nested list,
                  where all terms in each sublist must be present.
    :param fraction_required: The fraction of terms that must be present. Defaults to 1.0, meaning all terms must be present.
    :param all_terms_required: If True, all terms in each sublist of 'terms' must be present.
    :return: The original text if the required terms are present, None otherwise.
    """

    if all_terms_required:
        for term_list in terms:
            all_terms_present = all(
                re.search(term, text, flags=re.IGNORECASE) for term in term_list)
            if all_terms_present:
                return text
    else:
        total_terms = len(terms)
        found_terms = sum(1 for term in terms if re.search(
            term, text, flags=re.IGNORECASE))

        if found_terms / total_terms >= fraction_required:
            return text

    return None


def extract_sentences(text, keyword_list):

    # split the text into sentences
    sentences = nltk.sent_tokenize(text)
    extracted_sentences = []

    for keyword in keyword_list:
        for i in range(len(sentences)):
            # if the keyword is in the sentence, add it and the following sentence to the list
            if keyword in sentences[i]:
                extracted_sentences.append(sentences[i])
                if i+1 < len(sentences):
                    extracted_sentences.append(sentences[i+1])

    return ' '.join(extracted_sentences)


def cosine_similarity_df(dataframe, comparison_df, comparison_col, cols):

    df_scores = pd.DataFrame()
    vectorizer = TfidfVectorizer()
    comparison_vector = vectorizer.fit_transform(comparison_df[comparison_col])

    for col in cols:

        dataCol = dataframe[(dataframe[col].astype(str).str.len() > 0) | (
            ~dataframe[col].isnull())][col]

        if isinstance(dataCol.iloc[0], str):

            dataCol_vectorized = vectorizer.transform(dataCol.astype(str))

            similarity_scores = cosine_similarity(
                dataCol_vectorized, comparison_vector)

            max_scores = np.mean(similarity_scores, axis=1)

            df_scores[col + '_cs'] = pd.Series(max_scores, index=dataCol.index)

    final_df = pd.concat([dataframe, df_scores], axis=1)

    return final_df


dataframes = {}
counter = 1

links = readfiles(annual_report_path)
for l in links:
    try:
        with open(l, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(l)

        textDict = {}

        for i in range(0, len(pdf_reader.pages)):
            text = pdf_reader.pages[i]
            textDict[str(i + 1)] = text.extract_text()

        cleanDict = cleanText(textDict)
        ecoFact = pd.DataFrame.from_dict(cleanDict, orient="index")
        dataframes[counter] = ecoFact
    except AttributeError:
        print(f"AttributeError occurred at iteration {counter} for file {l}")
        continue
    finally:
        counter += 1

keywords_list = ['economic risk', 'business risk', 'outlook',
                 'economic condition', 'headwind', 'macroeconomic', 'pressure', 'trends']

annualreports = pd.concat(dataframes, ignore_index=False)
annualreports.columns = ["content"]
annualreports = annualreports.reset_index()
annualreports = annualreports[~annualreports["content"].str.contains(
    "notes to the")]

annualreports["sentences"] = annualreports["content"].str.split("\. ")
annualreports = annualreports.explode("sentences").reset_index()
annualreports.rename({"index": "reportNum"}, axis=1, inplace=True)

annualreports["dup"] = annualreports['content'].duplicated(keep="first")
annualreports.loc[annualreports['dup'] == True, 'content'] = np.nan
annualreports = annualreports.drop("dup", axis=1)

paraProxy = 3
annualreports.reset_index(inplace=True)
annualreports["groupingID"] = annualreports.groupby(
    "reportNum").cumcount() // paraProxy + 1
annualreports.drop("index", axis=1, inplace=True)
annualreports["paragraphs"] = annualreports.groupby(["reportNum", "groupingID"])[
    "sentences"].apply(lambda x: ' '.join(x)).reset_index()["sentences"]
annualreports.drop(["groupingID"], axis=1, inplace=True)
annualreports = annualreports[["reportNum",
                               "content", "paragraphs", "sentences"]]


def filter_noun(text):
    doc = nlp(text)
    noun = [token.text for token in doc if token.pos_ == "NOUN"]
    return ' '.join(noun)


rba_nouns = rbaminutes["Content"].apply(filter_noun)
rba_nouns = pd.DataFrame(rba_nouns, columns=["Content"])

cosine_annualreports = cosine_similarity_df(
    annualreports, rbaminutes, "Content", ['content', 'paragraphs', 'sentences'])
annualreport_numcount = cosine_annualreports[[
    "content", "paragraphs", "sentences"]].apply(lambda x: x.str.count(r'\d'))
annualreport_strlen = cosine_annualreports[[
    "content", "paragraphs", "sentences"]].apply(lambda x: x.str.len())

ar = pd.merge(cosine_annualreports, annualreport_numcount,
              left_index=True, right_index=True, suffixes=("", "_count"))
ar = pd.merge(ar, annualreport_strlen, left_index=True,
              right_index=True, suffixes=("", "_len"))
sentences_df = ar[["sentences", "sentences_cs", "sentences_count", "sentences_len"]
                  ].drop_duplicates("sentences").sort_values("sentences_cs", ascending=False)

# filter dataframe to ensure the count of numbers in the text are less than 3 and the length of the string is more than 100 letters.
# I have found that typically numbers in the text mean that it is not referring to macroeconomic features but talking about accounting numbers and percentages
filt_sentences_df = sentences_df[(sentences_df["sentences_count"] < 3) & (
    sentences_df["sentences_len"] > 100)].sort_values("sentences_cs", ascending=False)
# take the percentage of the filtered dataframe
pct = int(len(filt_sentences_df) * 0.025)
macro_summary = filt_sentences_df[:pct]

# take the frequency count of nouns in the rba minutes to find some similiarities in the text (taking top 200)
concatString = ' '.join(rba_nouns["Content"]).lower().split()
freqMinutes = Counter(concatString)
rba_noun_freq = pd.DataFrame.from_dict(
    freqMinutes, orient='index', columns=['Frequency'])
rba_noun_200 = rba_noun_freq.sort_values("Frequency", ascending=False)[
    :200].index.to_list()
# removing text that is not typically 'economic / financial'
remove_words = ["members", "market", "year", "quarter", "months", "time", "period", "levels", "term", "year", "level", "p", "meeting", "support", "c", "end", "firms", "discussion", "pace", "d-19", "liason",
                "bank", "authorities", "loan", "half", "work", "payments", "u", "part", "information", "system", "cities", "headline", "line", "asset", "point", "approvals", "g", "hours", "survey", "quarters",
                "extend", "depreciation", "parts", "state", "v", "people", "owner", "rents", "intentions"]
rba_noun_200 = [word for word in rba_noun_200 if word not in remove_words]


def count_func(row): return sum(row.str.count(word) for word in rba_noun_200)


macro_summary['count'] = macro_summary[["sentences"]].apply(count_func)
macro_summary = macro_summary[macro_summary["count"] > 5][["sentences"]]

# manually cleaning the text
replacements = {
    "in ation": "inflation",
    "nancial": "financial",
    "signi cant": "significant",
    "con dence": "confidence",
    "speci c": "specific",
    "andor ": "and/or",
    "fifinancial": "financial",
    "scal": "scale",
    "covidrelated": "covid related",
    "and/orinvestment": "and/or investment"
}

contentCol = macro_summary["sentences"]

matchedSentences = [sentence for sentence in contentCol]
for i in range(len(matchedSentences)):
    for key, value in replacements.items():
        matchedSentences[i] = matchedSentences[i].replace(key, value)

# apply implied spell check


def clean_spelling(text):

    sentences = nltk.sent_tokenize(text)
    tool = LanguageTool('en-AU')

    corrected_sentences = []
    for sentence in sentences:
        corrected_sentence = tool.correct(sentence)
        corrected_sentences.append(corrected_sentence)

    coherent_text = ' '.join(corrected_sentences)

    return coherent_text


output = matchedSentences
text = ' - '.join(output)
coherent_text = clean_spelling(text)
df = pd.DataFrame(coherent_text.split(" - "))


model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = model.encode(df[0].values)
kmeans_model = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans_model.fit_predict(sentence_embeddings)

cluster_dfs = [df[df['cluster'] == i] for i in range(kmeans_model.n_clusters)]

for i, cluster_df in enumerate(cluster_dfs):
    cluster_center = kmeans_model.cluster_centers_[i]
    distances = np.linalg.norm(
        sentence_embeddings[cluster_df.index] - cluster_center, axis=1)

    closest_sentence_index = np.argmin(distances)

    print(f"Cluster {i}: {cluster_df.iloc[closest_sentence_index][0]}")

# selected_cluster = input("Which cluster do you want to output?")
# selected_cluster = int(selected_cluster)
selected_cluster = 0

selected_cluster_df = df[df["cluster"] == selected_cluster]
text = selected_cluster_df[0].values
text = '. '.join(text)

with open('macroeconomic_factors_summarised.txt', "w") as f:
    f.write(text)
