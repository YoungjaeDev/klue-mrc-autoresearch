<!-- 강의 중 Claude Code에 그대로 붙여 넣는 후보 1회 실험 프롬프트 -->

```text
Read program.md first and follow it exactly.

Run exactly ONE new candidate:
1. Write one hypothesis and change train.py only as program.md allows.
2. Train for 30 optimizer steps (same budget, data, seed as the reference).
3. Evaluate on the full search split (all 1,168 questions).
4. Append one row to results.tsv (EM, ROUGE-W, time, VRAM, keep/discard).
5. Keep the change only if search EM is strictly higher than the current best.

Never run the final evaluation or open the final split.
One GPU job at a time.
If any GPU/CUDA/kernel/OOM error appears: stop immediately, show the error,
and report. Do not change drivers, CUDA, global Python, or the model to work around it.
```
