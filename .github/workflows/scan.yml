name: Scan diario VP

# Cuándo corre el robot:
on:
  schedule:
    # 22:30 UTC de lunes a viernes (tras el cierre de Wall Street, 16:00 ET).
    # GitHub usa UTC. Ajusta si quieres otra hora. Formato cron: min hora * * díasemana
    - cron: '30 22 * * 1-5'
  # Permite lanzarlo a mano desde la pestaña Actions (botón "Run workflow").
  workflow_dispatch:

# Permiso para que la Action haga commit de los JSON generados.
permissions:
  contents: write

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - name: Descargar el repo
        uses: actions/checkout@v4

      - name: Preparar Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      # El scanner usa solo librería estándar (urllib, json) — no hace falta pip.
      - name: Ejecutar el scanner
        run: |
          cd scanner
          python scanner.py

      - name: Guardar los JSON generados
        run: |
          git config user.name  "vp-scanner-bot"
          git config user.email "bot@users.noreply.github.com"
          git add scanner/data/*.json
          # Solo commitea si hubo cambios (evita commits vacíos).
          if git diff --staged --quiet; then
            echo "Sin cambios en los datos."
          else
            git commit -m "datos: scan $(date -u +%Y-%m-%d)"
            git push
          fi
