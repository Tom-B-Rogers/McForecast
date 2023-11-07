import re
import requests
import ssl
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By


def setup_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('stopwords')


def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    return webdriver.Chrome(options=options)


def clean_text(text):
    text = re.sub(
        r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^r|", "", text)
    text = re.sub(r'\t', " ", text)
    text = re.sub(r'(?<!\s)(?=[A-Z])', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def get_content(text, start_str, end_str):
    start = text.find(start_str)
    end = text.find(end_str)
    return text[start:end]


def extract_content(data_dict, start_str, end_str):
    for date, html in data_dict.items():
        response = requests.get(html)
        soup = BeautifulSoup(response.text, "html.parser")
        text = clean_text(soup.get_text())
        data_dict[date] = get_content(text, start_str, end_str)


def tokenize_content(df):
    stop_words = set(stopwords.words('english'))
    df["tokenized"] = df["Content"].apply(lambda x: " ".join(
        token for token in word_tokenize(x) if token.lower() not in stop_words))


def get_monetary_policy_decisions(driver):
    rbaMedia = "https://www.rba.gov.au/media-releases/"
    MonPolDecs = []

    driver.get(rbaMedia)
    for element in driver.find_elements(By.CSS_SELECTOR, '[itemprop="headline"]'):
        headline = element.get_attribute('innerHTML')
        link_element = element.find_element(By.XPATH, './parent::a[@href]')
        link = link_element.get_attribute('href')
        MonPolDecs.append((headline, link))

    return [t[1] for t in MonPolDecs if "Monetary Policy Decision" in t[0]]


def get_monetary_policy_minutes(driver):
    rbaMinutes = "https://www.rba.gov.au/monetary-policy/rba-board-minutes/{}/"
    MonPolMin = {}

    for year in range(2016, 2024):
        MinuteLinks = rbaMinutes.format(year)
        driver.get(MinuteLinks)
        article_list = driver.find_element(By.CLASS_NAME, 'list-articles')
        elements = article_list.find_elements(By.XPATH, './li/a[@href]')
        for element in elements:
            headline = element.text
            link = element.get_attribute('href')
            MonPolMin[headline] = link

    return MonPolMin


if __name__ == "__main__":
    setup_nltk()
    driver = create_driver()

    # Get Monetary Policy Decisions
    MonPolDecs = get_monetary_policy_decisions(driver)

    # Get Monetary Policy Minutes
    MonPolMin = get_monetary_policy_minutes(driver)

    # Extracting content
    extract_content(
        MonPolMin, "Members commenced their discussion", "Back to top")

    MonPolMin_df = pd.DataFrame.from_dict(MonPolMin, orient='index')
    MonPolMin_df.columns = ["Content"]
    MonPolMin_df.index = pd.DatetimeIndex(MonPolMin_df.index)
    tokenize_content(MonPolMin_df)
    MonPolMin_df = MonPolMin_df.sort_index()

    dfExport = MonPolMin_df[MonPolMin_df["tokenized"].str.contains(
        '[A-Za-z]')][["tokenized"]]
    dfExport.to_csv('rba_minutes.txt', sep='\n', index=False, header=False)
