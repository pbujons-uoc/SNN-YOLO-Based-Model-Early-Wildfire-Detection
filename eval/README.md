# eval/ — unified test-set evaluation (accuracy, time, energy)

Put this folder at `code_paper_2/eval/` (next to `Results_train`, `SpikeYOLO`, `ultralytics`, `VanillaCNN`). venv `tfm_paper` active.
Needs: torch, numpy, scipy, pandas, nvidia-ml-py (`pynvml`, already installed with CodeCarbon).

## Files
| file | what it does |
|---|---|
| `models.py` | finds the finished runs (`Results_train/*/weights/best.pt`), loads each model family, builds the test/val loader |
| `dump_predictions.py` | one model, one split -> every prediction saved to `Results_test/<model>/<split>_seed<k>.npz` |
| `metrics.py` | numpy only: mAP50, mAP50-95, per-class AP, threshold from val, TP/FP/FN/P/R/F1 on test, recall by size, false alarms, bootstrap CIs |
| `benchmark_efficiency.py` | latency + GPU energy (NVML counter): network only (encoding bypassed), encoding alone, end-to-end, idle baseline |
| `aggregate.py` | tables: accuracy (+CI), paired bootstrap differences, efficiency mean/sd/CI over repeats, Welch/Mann-Whitney with Holm |
| `run_eval.py` | runs everything, resumable, interleaves models across benchmark repeats |
| `selftest_metrics.py`, `selftest_pipeline.py` | synthetic-data tests of the metric/statistics code (no GPU) |

## Order
```
python selftest_metrics.py && python selftest_pipeline.py            # 1 minute, no GPU

# 1) check the pipeline on ONE finished model (can run while the queue is training)
python dump_predictions.py --model yolov8m_s0 --split val
python metrics.py --val ../Results_test/yolov8m_s0/val_seed0.npz --test ../Results_test/yolov8m_s0/val_seed0.npz --out /tmp/check.json --n-boot 0
#    mAP50 / mAP50-95 must be close to the best row of Results_train/yolov8m_s0/epoch_metrics.csv
#    (small differences: here a square 640x640 letterbox for every model, the Ultralytics validator uses rect batches)

# 2) accuracy of every finished run (queue may still be running)
python run_eval.py --stage accuracy --repeats 5

# 3) when the queue has finished: time and energy, 5 repeats, models interleaved
nohup setsid python run_eval.py --stage bench --repeats 5 > eval_bench.log 2>&1 &

# 4) tables
python run_eval.py --stage aggregate --carbon-intensity <g CO2/kWh>
```
## Notes
* `--repeats` = number of Poisson generator seeds (accuracy) AND number of benchmark repeats. Direct RGB, latency, YOLOv8m and
  VanillaCNN are deterministic: one accuracy pass is enough, their uncertainty is the bootstrap interval over the test images.
* Training variance is not measured (only training seed 0).
* The test split is read from `data.yaml` (`test:`); if the key is missing it falls back to `<path>/test`. Add `test: test/` to be explicit.
* NMS is multi-label for all models (as the Ultralytics validator): conf 0.001, IoU 0.7, max 300.
* Never benchmark while `run_queue.py` is running; the script refuses (NVML sees other GPU processes).
* Optional for stable energy numbers: `sudo nvidia-smi -pm 1` (persistence), and nothing else running on the machine.
