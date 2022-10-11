#!/usr/bin/env python3
# Author: Armit
# Create Time: 2020/09/20 

from time import sleep
from re import compile as R
from bs4 import BeautifulSoup
from pathlib import Path
from requests import Session

BASE_PATH = Path(__file__).absolute().parent

ARXIV_URL = 'https://arxiv.org/%s/%s'
ARXIV_REGEX = R(r'[\.v0-9]+.pdf')
ARXIV_FILEDS = ['abs', 'abs/quant-ph', 'quant-ph']
FILENAME_ESCAPE = R(r'\\|/|:|\*|\?|\"|<|>|\|')

HTTP = Session()

for fp in BASE_PATH.iterdir():
  if not ARXIV_REGEX.match(fp.name): continue
  print(f'[{fp.stem}]')
  
  for fld in ARXIV_FILEDS:  
    url = ARXIV_URL % (fld, fp.stem)
    print(f'  [GET] {url} ', end='')
    res = HTTP.get(url, timeout=30)
    if res.status_code != 200:
      print(f'(failed: {res.status_code})')
      continue
    else:
      print('(ok)')

    html = BeautifulSoup(res.content, features="html5lib")
    h1 = html.find('h1', attrs={'class': 'title mathjax'})
    if h1:
      title = h1.text
      if title.startswith('Title:'):
        title = title[len('Title:'):]
      print(f'  rename => {title}')
      title = FILENAME_ESCAPE.sub('-', title)
      try:
        fp.rename(str(title + fp.suffix))
        break
      except OSError as e:
        print(e)
    else:
      sleep(0.25)
