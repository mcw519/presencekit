# YuNet 候選框解碼與 NMS Rust PoC

這個實驗固定 YuNet ONNX 推論，對照現行 NumPy 解碼與 NMS、NumPy 解碼加 OpenCV `NMSBoxes`、Rust 解碼與 NMS。
Rust 入口以一次 PyO3 呼叫讀取 12 個 float32 輸出張量。輸入透過 `PyBuffer` 複製為連續 float32 buffer，
保留的偵測列以 float32 bytes 回傳，由 NumPy 建立檢視。NumPy 和 Rust 的 exp 實作可能有少量
ULP 差異。若候選框靠近整數截斷的邊界，該幀使用原本的 NumPy 判斷。部署維持原本的 detector。

## 重跑

需要 rust-anything 3.2、Rust 1.97 以上、Python 3.11 以上，以及本機 YuNet、SFace、MoveNet ONNX。
從專案根目錄執行，讓 Rust 和 Python 都可從 `PATH` 找到。

```sh
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD"
out=$(mktemp -d)
(cd experiments/yunet_rust/native && rust-anything build .)
python experiments/yunet_rust/run.py verify "$out/inputs.jsonl" > "$out/parity.json"
(
  cd experiments/yunet_rust/native
  rust-anything record . --count 0 --inputs "$out/inputs.jsonl" --float-tolerance 1e-6
  rust-anything diff . --json > "$out/diff.json"
)
python experiments/yunet_rust/run.py bench --repeats 7 > "$out/benchmark.json"
```

`verify` 檢查 confidence 與 IoU 門檻兩側、相等、平手順序、top-k、空輸出及整數框回退。
影片的原始 YuNet 張量先逐幀對照三種後處理。預期值來自實際的 `OrtYuNetDetector`。分數要求完全相同，
框與 landmark 的絕對誤差需在 0.001 px 以內。SDK 回放另比對追蹤 ID、配對及狀態事件。
`rust-anything diff` 使用相同 Rust 內核的盒裝回放介面，直接針對 PyBuffer 的測試由 `verify` 執行。
如使用 `rust-anything refresh` 更新 `lib.rs`，保留 `buffer_bridge` 的註冊。

`verify` 也在同一批已通過驗證的張量上量後處理 p95，包含 PyO3 輸入與回傳轉換。完整基準預設讀取
專案的多人示範影片。也可以在子命令前加上 `--video 路徑`。以 2 fps 取樣，每幀寬度至多 640 px，
將同一批記憶體內 JPEG 傳給每個新的 Python 子程序。固定 ONNX Runtime CPU、2 intra-op 執行緒、
runtime 預設 spinning、OpenCV 2 執行緒、Python Kalman tracker 與 HMM。每幀跑 YuNet，SFace 與
MoveNet 維持各自的快取。SDK p95 和 CPU 時間含 JPEG 解碼及 PyO3，排除暖機。峰值 RSS 含子程序啟動與暖機，
來源影片的解碼在父程序。每輪輪換版本順序。
