from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
import json

URLS = [
    'https://data.gov.tw/dataset/16799',
    'https://data.gov.tw/dataset/61131',
    'https://data.gov.tw/dataset/61132',
]

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._href = None
        self._text = []
    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return
        attrs = dict(attrs)
        self._href = attrs.get('href')
        self._text = []
    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data.strip())
    def handle_endtag(self, tag):
        if tag == 'a' and self._href is not None:
            self.links.append({'href': self._href, 'text': ' '.join(x for x in self._text if x)})
            self._href = None
            self._text = []

out = {}
for url in URLS:
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(req, timeout=30).read().decode('utf-8', errors='replace')
    parser = LinkParser()
    parser.feed(html)
    links = [x for x in parser.links if any(k in (x['href'] or '').lower() for k in ['csv', 'download', 'resource', 'file'])]
    out[url] = links
print(json.dumps(out, ensure_ascii=False, indent=2))
