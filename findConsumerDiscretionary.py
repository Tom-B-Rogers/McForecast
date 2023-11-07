import pandas as pd
from bs4 import BeautifulSoup
import requests

# Obtain information for Consumer Discretionary companies within the ASX200

url = "https://en.wikipedia.org/wiki/S%26P/ASX_200"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
soupTable = soup.find_all("table")[1]
tableASX = pd.DataFrame(pd.read_html(str(soupTable))[0])
consumerDiscretionary = tableASX[tableASX["Sector"] ==
                                 "Consumer Discretionary"][["Code", "Company", "Sector"]]

codeList = list(consumerDiscretionary["Code"])
companyList = list(consumerDiscretionary["Company"])
