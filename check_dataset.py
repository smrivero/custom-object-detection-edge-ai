from pathlib import Path
from collections import Counter
import math
root=Path(__file__).resolve().parent/'dataset'
seen=set()
for split in ('train','val','test'):
    images=sorted((root/'images'/split).glob('*.jpg'))
    counts=Counter(); negatives=0
    for p in images:
        assert p.name not in seen, f'Image repeated across splits: {p.name}'
        seen.add(p.name)
        lab=root/'labels'/split/f'{p.stem}.txt'
        assert lab.is_file(), f'Missing label: {lab}'
        lines=lab.read_text().splitlines()
        negatives+=int(not lines)
        for line in lines:
            tokens=line.split(); assert len(tokens)==5, (lab,line)
            cls=int(tokens[0]); box=list(map(float,tokens[1:]))
            assert cls in (0,1) and all(math.isfinite(v) and 0<=v<=1 for v in box), (lab,line)
            assert box[2]>0 and box[3]>0, (lab,line)
            counts[cls]+=1
    assert counts[0]>0 and counts[1]>0
    print(f'{split:5s}: {len(images):2d} images | mate={counts[0]} boxes | termo={counts[1]} boxes | negatives={negatives}')
assert len(seen)==99
print('Dataset OK: 99 images, two classes, no repeated filenames across splits.')
