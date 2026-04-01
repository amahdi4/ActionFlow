---
paths:
  - "src/demo/**/*.py"
---

- Demo config lives in `configs/schema.py:DemoConfig` — use it for host, port, share, enabled_backends
- Use `gr.File(type="filepath")` for video uploads, not binary mode
- The demo supports upload and webcam modes per the scaffold design
- Test locally with: `PYTHONPATH=src python -m scripts.cli launch-demo`
- Enabled flow backends come from `config.demo.enabled_backends`, not hardcoded
