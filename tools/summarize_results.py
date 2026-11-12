"""Recompute published aggregates from released tables, without dataset access."""
import csv
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def summarize():
    per = rows('results/dino/per_category.csv')
    macro = rows('results/dino/performance_macro.csv')
    assert len(per) == 56 and len(macro) == 7
    print('DINO: bad-only category pooling, equal 8-category means (%)')
    for row in macro:
        group = [r for r in per if r['method'] == row['method']]
        assert len(group) == 8 and len({r['category'] for r in group}) == 8
        values = []
        for src,dst in [('pauroc_005','pAUROC_005_percent'),('aupro_005','AU_PRO_005_percent')]:
            valid = [float(r[src]) for r in group if r[src] != '']
            if not valid:
                assert row[dst] == ''
                values.append('N/A')
            else:
                assert len(valid) == 8
                value = 100*mean(valid)
                assert math.isclose(value,float(row[dst]),rel_tol=0,abs_tol=1e-10)
                values.append(f'{value:.4f}')
        print(f"{row['method']}: pAUROC={values[0]}, AU-PRO={values[1]}")
    patch = rows('results/patchcore/per_category.csv')
    pm = json.loads((ROOT/'results/patchcore/macro.json').read_text(encoding='utf-8'))
    for method in ['CP100','coreset']:
        group = [r for r in patch if r['method'] == method]
        assert len(group) == 8
        for key,value in pm[method].items():
            assert math.isclose(mean(float(r[key]) for r in group),value,rel_tol=0,abs_tol=1e-12)
    pairs = rows('results/geometry/cross_pairs_672.csv')
    dirs = rows('results/geometry/cross_directional_1344.csv')
    assert len(pairs) == 672 and len(dirs) == 1344
    assert len({(r['category'],r['layer'],r['a'],r['b']) for r in pairs}) == 672
    assert len({(r['category'],r['layer'],r['source'],r['target']) for r in dirs}) == 1344
    ids = rows('results/geometry/id_overlap_per_bank.csv')
    summary = rows('results/geometry/id_overlap_summary.csv')
    assert len(ids) == 480 and len(summary) == 15
    for s in summary:
        rr = [r for r in ids if (r['method_a'],r['method_b']) == (s['method_a'],s['method_b'])]
        assert len(rr) == 32
        for src,dst in [('fraction_a_in_b','mean_fraction_a_in_b'),('fraction_b_in_a','mean_fraction_b_in_a'),('Jaccard','mean_Jaccard')]:
            assert math.isclose(mean(float(r[src]) for r in rr),float(s[dst]),rel_tol=0,abs_tol=1e-12)
    print('PatchCore macro, 480 ID pairs, 672 cross pairs, 1344 directions: verified')
    print('SuperADD paper-reported reference: 8-category pAUROC=83.93%, fixed F1=62.61%; separate conditions, no local delta.')


if __name__ == '__main__':
    summarize()
