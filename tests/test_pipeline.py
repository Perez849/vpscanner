"""
Prueba de integración sin red: validación final + escaneo diario sobre datos sintéticos con una reversión plantada.
Comprueba que el registro se genera, que el scanner lo lee, emite alertas bien formadas y que el seguimiento es idempotente.
Ejecutar: python tests/test_pipeline.py   (≈ 1 minuto)
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scanner')


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return r.stdout


def inject_fake_pelotazo(path):
    """Los datos sintéticos no tienen colas gruesas: se añade al registro un plan de pelotazos con un modelo aleatorio solo para ejercitar scan/seguimiento."""
    import numpy as np
    sys.path.insert(0, ROOT)
    import feats, model as MD, setups as SU, research as R
    reg = json.load(open(path))
    names = [x.id for x in SU.PEL_SETUPS]
    fidx = [feats.FEATURES.index(f) for f in list(MD.MODEL_FEATURES) + R.PEL_EXTRA]
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, len(feats.FEATURES))); FL = (rng.random((n, len(names))) < 0.2).astype(np.float32)
    gid = rng.integers(0, len(feats.GROUPS), n); y = (rng.random(n) < 0.15).astype(np.float32)
    m = MD.LinearLogitModel(fidx, names, feats.GROUPS).fit(X, FL, gid, y)
    cal = MD.LogitModel(fidx, names, feats.GROUPS); cal.set_calibration(m.predict_raw(X, FL, gid), y); m.calib = cal.calib
    v = R.PEL_VARIANTS[0]
    reg['pelotazo'] = {'id': R.PEL_PLAN['id'], 'label': R.PEL_PLAN['label'], 'blurb': R.PEL_PLAN['blurb'], 'role': 'pelotazo',
                       'exit': {'kind': v.kind, 'T': v.T, 'S': v.S, 'H': v.H}, 'groups': R.PEL_GROUPS, 'nPerDay': 3, 'floorRaw': 0.0, 'bigPct': 12.0,
                       'paused': False, 'patterns': [{'id': k, 'label': SU.BY_ID[k].label} for k in names],
                       'stats': {'oos': {'n': 500, 'wr': 48.0, 'mean': 1.2, 'median': -1.0, 'pf': 1.1, 'p20': 22.0, 'p30': 10.0}}, 'model': m.to_json()}
    json.dump(reg, open(path, 'w'))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        reg_dir, out = os.path.join(tmp, 'model'), os.path.join(tmp, 'data')
        run('research.py', 'final', '--synthetic', '300', '--synthetic-ar', '-0.12', '--out', reg_dir)
        reg = json.load(open(os.path.join(reg_dir, 'validated.json')))
        assert reg['strategies'], 'sin planes validados en datos con reversión plantada'
        for s in reg['strategies']:
            assert s['model']['type'] == 'linear' and 0.4 < s['floorRaw'] < 0.9 and s['stats']['oos']['n'] > 500
            assert s['nPerDay'] in (3, 5) and s['stats']['byRank'] and s['stats']['vsSpy'] and s['stats']['portfolio']['10x10']['cagr'] is not None
        inject_fake_pelotazo(os.path.join(reg_dir, 'validated.json'))
        run('scan.py', '--synthetic', '400', '--registry', os.path.join(reg_dir, 'validated.json'), '--out', out, '--no-notify')
        al = json.load(open(os.path.join(out, 'alerts.json')))
        assert al['alerts'], 'el scanner no emitió alertas'
        assert al['pelotazos'] and all(p['exit']['kind'] == 'trl' and 0 < p['pBig'] < 1 and p['levels']['stop'] < p['sigClose'] for p in al['pelotazos'])
        tr0 = json.load(open(os.path.join(out, 'tracking.json')))
        assert tr0['stats']['pelotazo']['pending'] == len(al['pelotazos']), 'los pelotazos deben entrar en el seguimiento (aparte de las estadísticas globales)'
        assert tr0['stats']['global']['pending'] == sum(len(a['plans']) for a in al['alerts'])
        for a in al['alerts']:
            assert a['plans'] and all(0 < p['p'] < 1 for p in a['plans'])
            assert all(p['rank'] <= next(x['nPerDay'] for x in reg['strategies'] if x['id'] == p['strategy']) for p in a['plans']), 'más alertas por plan que su N por día'
            assert a['levels']['stop'] < a['sigClose']
        t1 = open(os.path.join(out, 'tracking.json')).read()
        run('scan.py', '--synthetic', '400', '--registry', os.path.join(reg_dir, 'validated.json'), '--out', out, '--no-notify')
        tr = json.load(open(os.path.join(out, 'tracking.json')))
        ids = [t['id'] for t in tr['trades']]
        assert len(ids) == len(set(ids)), 'el seguimiento duplicó operaciones al re-ejecutar'
        # escaneo previo al cierre (MOC): mismos datos con la última barra marcada como provisional
        out2 = os.path.join(tmp, 'data2')
        reg_p = os.path.join(reg_dir, 'validated.json')
        r = subprocess.run([sys.executable, 'scan.py', '--synthetic', '400', '--registry', reg_p, '--out', out2, '--no-notify', '--mode', 'preclose'],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0 and 'Fuera de la ventana' in r.stdout, 'preclose debe respetar la ventana horaria sin --force'
        assert not os.path.exists(os.path.join(out2, 'alerts_pre.json'))
        run('scan.py', '--synthetic', '400', '--registry', reg_p, '--out', out2, '--no-notify', '--mode', 'preclose', '--force')
        pre = json.load(open(os.path.join(out2, 'alerts_pre.json')))
        assert pre['mode'] == 'moc' and pre['alerts'], 'el escaneo previo no emitió avisos'
        assert not json.load(open(os.path.join(out2, 'tracking.json')))['stats']['pelotazo']['pending'], 'preclose no debe emitir pelotazos'
        assert all(a['id'].endswith('|moc') and a['mode'] == 'moc' and a['sym'] in pre['candles'] for a in pre['alerts'])
        assert not os.path.exists(os.path.join(out2, 'alerts.json')), 'preclose no debe sobrescribir alerts.json'
        trp = json.load(open(os.path.join(out2, 'tracking.json')))
        assert trp['trades'] and all(t['mode'] == 'moc' and t['status'] == 'pendiente' for t in trp['trades'])
        # escaneo posterior al cierre con los mismos datos: no duplica las operaciones ya avisadas
        n_moc = len(trp['trades'])
        run('scan.py', '--synthetic', '400', '--registry', reg_p, '--out', out2, '--no-notify')
        al2 = json.load(open(os.path.join(out2, 'alerts.json')))
        tr2 = json.load(open(os.path.join(out2, 'tracking.json')))
        assert sum(1 for t in tr2['trades'] if t.get('mode') == 'moc') == n_moc
        assert len(tr2['trades']) == len({t['id'] for t in tr2['trades']})
        seen = {(t['sym'], t['setup'], t['sigTs']) for t in tr2['trades'] if t.get('mode') == 'moc'}
        for t in tr2['trades']:
            if t.get('mode') != 'moc':
                assert (t['sym'], t['setup'], t['sigTs']) not in seen, 'operación de apertura duplicada de una ya avisada al cierre'
        assert any(a['pre'] for a in al2['alerts']), 'las alertas ya avisadas antes del cierre deben marcarse'
        # sin registro → sin alertas y sin errores
        run('scan.py', '--synthetic', '100', '--registry', os.path.join(tmp, 'no_existe.json'), '--out', os.path.join(tmp, 'vacio'), '--no-notify')
        assert json.load(open(os.path.join(tmp, 'vacio', 'alerts.json')))['alerts'] == []
    print('TODO OK · pipeline de validación + escaneo + seguimiento')


if __name__ == '__main__':
    main()
