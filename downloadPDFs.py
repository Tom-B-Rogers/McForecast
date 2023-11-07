import findConsumerDiscretionary as fCD
import globalFunctions as gf
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import re
import urllib.request
from selenium.webdriver.common.by import By
import datetime

# Takes about 8 minutes to run
search_results = 4
url_google = "https://www.google.com.au/search?client=safari&rls=en&q="
search_term = "Investor Centre Annual Report 2023 PDF"

icAnnouncements = []
options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome(options=options)

for comp in list(fCD.companyList):
    url = url_google + " " + comp + " " + search_term
    driver.get(url)
    linkList = []
    for link in driver.find_elements(By.CSS_SELECTOR, 'a'):
        url = link.get_attribute('href')
        if url and url.startswith('http') and 'google.com' not in url:
            linkList.append(url)
    icAnnouncements.append(linkList[:search_results])

driver.quit()


def findPDFs(link: str):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome(options=options)
    driver.get(link)
    driver.implicitly_wait(10)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    pdf_links = soup.find_all('a', href=re.compile(r'.+\.pdf'))

    appLinks = []
    for l in pdf_links:
        pdf_link = l.get('href')
        if not pdf_link.startswith('http'):
            pdf_link = urllib.parse.urljoin(link, pdf_link)
        appLinks.append(pdf_link)

    appLinks = sorted(appLinks, key=lambda x: re.search(
        r'\d{4}', x).group() if re.search(r'\d{4}', x) else '', reverse=True)
    latestReport = next(
        (l for l in appLinks if 'Annual' in l.title() and 'Report' in l.title()), None)

    driver.quit()

    if latestReport:
        return [latestReport], 1
    else:
        return [], 0


flattenedList = [item for sublist in icAnnouncements for item in sublist]
flattenedList = [x for x in flattenedList if str(x) != 'nan']

investorList = []
indexer = 0
investorDict = {}
for company in flattenedList:
    tempData, count = findPDFs(company)
    investorList.append(tempData)
    investorDict[indexer] = count
    indexer += 1

investorList = list(filter(None, investorList))
FinvestorList = []
for sublist in investorList:
    FinvestorList.extend(sublist)

tempList = []
for item in FinvestorList:
    if "Annual" in item.title() and "Report" in item.title():
        tempList.append(item)

counter = 1
for link in tempList:
    destination_path = f'/Users/tomrogers/Desktop/InvestorPresentations/{counter}.pdf'
    counter += 1
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"}
        response = requests.get(link, headers=headers)
        with open(destination_path, 'wb') as f:
            f.write(response.content)
    except:
        print(f"cannot find {link}")
