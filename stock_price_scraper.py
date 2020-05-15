import requests
from bs4 import BeautifulSoup
import pandas as pd

qimai= "https://www.qimai.cn/rank/rankTrend/brand/free/country/cn/genre/6014/device/iphone"

from time import sleep
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(qimai)

html = driver.page_source
soup = BeautifulSoup(html)

for tag in soup.find_all("table"):
    print(tag.text)


if __name__ == "__main__":
    html = requests.get(qimai).content
    # beautifulsoup object
    soup = BeautifulSoup(html , "html.parser")
    #print(soup.prettify())
    print(soup.find_all('table', class_ = "data-table" ))
    headings = [heading.text for heading in soup.find("thead").find_all("tr")]

    datasets = []
    for row in soup.find("tbody").find_all("tr"):
        dataset = dict(zip(headings, (td.get_text() for td in row.find_all("td"))))
        datasets.append(dataset)

    price_data = pd.DataFrame(datasets)

    dataset2 = []
    for row in soup.find_all("tr", attrs={"class": "BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)"}):
        dataset = dict(zip(headings, (td.get_text() for td in row.find_all("td"))))
        dataset2.append(dataset)

    price_data2 = pd.DataFrame(dataset2)
    print(price_data2)





