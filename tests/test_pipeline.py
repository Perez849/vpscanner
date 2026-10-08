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


def main():
    with tempfile.TemporaryDirectory() as tmp:
        reg_dir, out = os.path.join(tmp, 'model'), os.path.join(tmp, 'data')
        run('research.py', 'final', '--synthetic', '300', '--synthetic-ar', '-0.12', '--out', reg_dir)
        reg = json.load(open(os.path.join(reg_dir, 'validated.json')))
        assert reg['strategies'], 'sin planes validados en datos con reversión plantada'
        for s in reg['strategies']:
            assert s['model']['type'] == 'linear' and 0.4 < s['floorRaw'] < 0.9 and s['stats']['oos']['n'] > 500
        run('scan.py', '--synthetic', '400', '--registry', os.path.join(reg_dir, 'validated.json'), '--out', out, '--no-notify')
        al = json.load(open(os.path.join(out, 'alerts.json')))
        assert al['alerts'], 'el scanner no emitió alertas'
        for a in al['alerts']:
            assert a['plans'] and all(0 < p['p'] < 1 for p in a['plans'])
            assert a['levels']['stop'] < a['sigClose']
        t1 = open(os.path.join(out, 'tracking.json')).read()
        run('scan.py', '--synthetic', '400', '--registry', os.path.join(reg_dir, 'validated.json'), '--out', out, '--no-notify')
        tr = json.load(open(os.path.join(out, 'tracking.json')))
        ids = [t['id'] for t in tr['trades']]
        assert len(ids) == len(set(ids)), 'el seguimiento duplicó operaciones al re-ejecutar'
        # sin registro → sin alertas y sin errores
        run('scan.py', '--synthetic', '100', '--registry', os.path.join(tmp, 'no_existe.json'), '--out', os.path.join(tmp, 'vacio'), '--no-notify')
        assert json.load(open(os.path.join(tmp, 'vacio', 'alerts.json')))['alerts'] == []
    print('TODO OK · pipeline de validación + escaneo + seguimiento')


if __name__ == '__main__':
    main()
