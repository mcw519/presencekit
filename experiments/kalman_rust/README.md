# Kalman 配對 Rust PoC

`RustTracker` 透過 rust-anything 產生的 PyO3 bridge，每幀把 Kalman 預測、配對成本、Hungarian 指派和已匹配濾波器更新交給 Rust。未配對軌跡的延續、新軌跡與參與者的生命週期，以及 session 狀態與事件，由兩個版本共用的 Python tracker 處理。應用程式預設使用 Python。PoC 透過驗證和基準腳本載入 Rust。

濾波器使用四組獨立的位置與速度協方差，對應原來的 8 × 8 矩陣。Rust 端接收每幀的已篩選偵測與現有軌跡。
快速入口將 embedding 與外觀特徵以連續 bytes 傳過 PyO3，保留每個元素的數值精度，並省去逐元素建立 Python float。
當 cosine 接近弱或強門檻、配對分數接近下限，或指派出現近似平手時，Rust 會把整幀交回原本的 Python 配對器，
由它從該幀開始前的濾波狀態判斷。驗證涵蓋門檻兩側和平手，也逐幀比較 ID、配對、狀態和事件。
rust-anything 以 `1e-6` 容差額外重播原有巢狀 list 介面的數值結果。

## 重跑驗證和基準

需要 rust-anything 3.2、Cargo 與 Rust 1.97 以上、專案的 Python 環境，以及本機 YuNet、SFace、MoveNet ONNX 模型。從專案根目錄執行。

```sh
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD"
out=$(mktemp -d)
(cd experiments/kalman_rust/native && rust-anything build .)
python experiments/kalman_rust/run.py verify "$out/inputs.jsonl"
(
  cd experiments/kalman_rust/native
  rust-anything record . --count 0 --inputs "$out/inputs.jsonl" --float-tolerance 1e-6
  rust-anything diff . --json > "$out/diff.json"
)
python experiments/kalman_rust/run.py bench --repeats 7 > "$out/benchmark.json"
python experiments/kalman_rust/run.py bench --workload tracking --repeats 5 > "$out/tracking-benchmark.json"
```

變更 `oracle.py` 或其依賴的 Python tracker 時，重新產生錄製資料。若用 rust-anything 重生 `lib.rs`，需保留
`buffer_bridge` 的註冊。基準比較 Python、巢狀 list 版 Rust 和 buffer 版 Rust，固定 ONNX Runtime CPU、每個模型
2 個 intra-op 執行緒、runtime 預設 spinning，以及 OpenCV 2 個執行緒。每個版本在新的 Python 程序處理同一段
180 張 JPEG，包含 1–3 個畫面內的人，使用 2 fps 的虛擬時間。每輪輪換執行順序。p95 與 CPU 時間取自暖機後的
SDK 回放，包含資料轉換與 PyO3 呼叫。峰值 RSS 取自每個子程序的整個生命週期。基準使用同一張測試臉生成畫面，
另以不同身分的合成偵測驗證配對。

`--workload tracking` 將事先產生的偵測直接送進 tracker，量測含 PyO3 呼叫的整個追蹤更新。CPU 時間排除合成輸入的產生時間。

rust-anything 的 `gate` 以 `oracle.py` 的行數作為大小檢查基準。這個檔案透過既有 Python 追蹤器執行預測、配對與更新，故大小檢查的分母只包含回放介面。本次 PoC 以 `diff` 和完整追蹤驗證確認行為。
