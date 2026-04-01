---
paths:
  - "src/training/**/*.py"
  - "src/models/**/*.py"
---

- Use `torch.no_grad()` context manager in all evaluation/inference paths
- Handle mixed precision with `torch.amp.autocast()` and `GradScaler`
- Keep hyperparameters (batch size, lr, epochs) in the config schema, never hardcode
- Save checkpoints with model `state_dict()`, optimizer state, epoch, and loss
- Always check `is_available()` on backends before using them
- Device selection goes through `utils.device.detect_device()`, never hardcode device strings
