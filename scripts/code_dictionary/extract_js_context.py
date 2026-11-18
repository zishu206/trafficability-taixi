from pathlib import Path

BASE = Path(__file__).resolve().parent / 'js_inspect'
TERMS = ['vfS', 'ZfS', 'TYPE_OF_ROCK', '極細砂', '細砂含粉砂', '資料空缺']

for path in sorted(BASE.glob('*.js')):
    text = path.read_text(encoding='utf-8', errors='replace')
    print(f'### {path.name}, length={len(text)}')
    for term in TERMS:
        positions = []
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            positions.append(idx)
            start = idx + len(term)
            if len(positions) >= 10:
                break
        print(f'-- {term}: {len(positions)} shown / first positions {positions}')
        for idx in positions[:3]:
            lo = max(0, idx - 260)
            hi = min(len(text), idx + len(term) + 260)
            print(text[lo:hi].replace('\n', ' '))
            print('---')
