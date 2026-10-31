"""Replay synthetic metadata in a real browser against a Git ref and current UI.

Run with the project environment, then open the printed localhost URL. No camera,
microphone, cloud API or real WebSocket is used. Results are printed as JSON.
This measures rendering work, not model inference or camera encoding.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "presencekit" / "static"
BOOT = """
<script>
window.benchErrors=[];
addEventListener('error',e=>benchErrors.push(e.message));
window.WebSocket=class {
 constructor(){this.readyState=1;this.bufferedAmount=0;setTimeout(()=>this.onopen?.(),0)}
 send(){} close(){} };
window.fetch=async()=>({json:async()=>({set:false})});
</script>
"""
PARENT = """<!doctype html><meta charset="utf-8"><title>Frontend replay benchmark</title>
<style>body{font:16px system-ui;background:#111827;color:#eee}iframe{width:1200px;height:800px}
pre{white-space:pre-wrap}button{padding:12px}</style>
<h1>Frontend metadata replay</h1><p>Each run: 10 Hz audio, 2 Hz state, 1 Hz stats.
Synthetic data only; camera, microphone and cloud services are disabled.</p>
<button id="run">Run before / after</button><pre id="results"></pre><div id="host"></div>
<script>
const results=[];let phase=0;
function next(){const version=phase?'after':'before';
 document.querySelector('#host').innerHTML=`<iframe src="/${version}/index.html?lang=en"></iframe>`}
document.querySelector('#run').onclick=()=>{results.length=0;phase=0;next()};
addEventListener('message',async e=>{
 if(e.origin!==location.origin||e.source!==document.querySelector('iframe')?.contentWindow||e.data?.kind!=='benchmark')return;
 results.push(e.data);document.querySelector('#results').textContent=JSON.stringify(results,null,2);
 if(!phase){phase=1;next()}else{
  document.querySelector('#host').innerHTML='';
  await fetch('/results',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(results)});
  document.title='Frontend replay complete';
 }
});
</script>"""


def fixtures():
    from presencekit.audio import AudioStream
    from presencekit.stats import ConnStats, build_stats
    from presencekit.tracker import Config, PresenceTracker

    class Iso:
        base_mix, delay, vad_heads, resets = 0.9, 480, [], 0

        class config:
            win_length = 512

    state = PresenceTracker(Config()).snapshot()
    state.update(type="state", running=True, pipeline="Synthetic benchmark", recommended_fps=0.5)
    stream = AudioStream(Config(), isolator_factory=Iso)
    audio = stream.meta()
    audio.update(
        type="audio",
        available=True,
        rtf=0.2,
        rtf_recent=0.2,
        proc_ms=4,
        proc_p95_ms=5,
        queue_wait_ms_avg=0.1,
        queue_wait_ms_p95=0.2,
        t_client=0,
        t_vision_client=0,
        queue=0,
        dropped=0,
        backlog_max=0,
        state="ENDED",
    )
    stats = build_stats(ConnStats(), {"pipeline": "Synthetic", "opencv": "benchmark", "cv_threads": 2, "cpu_count": 10})
    stats["conn"]["optim"]["recommended_fps"] = 0.5
    return {"state": state, "audio": audio, "stats": stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before-ref", default="HEAD")
    ap.add_argument("--before-dir", type=Path, help="saved static/ directory; overrides --before-ref")
    ap.add_argument("--mode", choices=("metadata", "capture"), default="metadata")
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.seconds < 1:
        ap.error("--seconds must be positive")
    fixture = fixtures()
    script_name = "benchmark_capture.js" if args.mode == "capture" else "benchmark_frontend.js"
    replay = (ROOT / "scripts" / script_name).read_text()
    with tempfile.TemporaryDirectory(prefix="presence-frontend-") as folder:
        root = Path(folder)
        parent = PARENT
        if args.mode == "capture":
            parent = parent.replace(
                "Each run: 10 Hz audio, 2 Hz state, 1 Hz stats.",
                "Each run: fixed 5 fps synthetic 640×480 canvas through the production capture callback.",
            )
        (root / "index.html").write_text(parent)
        for version in ("before", "after"):
            for file in STATIC.rglob("*"):
                if not file.is_file():
                    continue
                relative = file.relative_to(STATIC)
                if version == "before" and args.before_dir:
                    old_file = args.before_dir / relative
                    if not old_file.is_file():
                        continue
                    data = old_file.read_bytes()
                elif version == "before":
                    for package in ("presencekit", "presence_demo"):
                        source = subprocess.run(
                            ["git", "show", f"{args.before_ref}:{package}/static/{relative}"],
                            cwd=ROOT,
                            capture_output=True,
                            check=False,
                        )
                        if source.returncode == 0:
                            break
                    if source.returncode:
                        continue
                    data = source.stdout
                else:
                    data = file.read_bytes()
                dest = root / version / "static" / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            index = (root / version / "static" / "index.html").read_text()
            index = index.replace("/static/", f"/{version}/static/").replace("<head>", "<head>" + BOOT)
            if args.mode == "capture":
                instrumentation = """<script>
window.benchCaptureTicks=[];window.benchObserverMs=0;
const benchNativeInterval=window.setInterval.bind(window);
window.setInterval=(fn,ms,...args)=>{let id;id=benchNativeInterval(()=>{
 const t=performance.now(),observer=benchObserverMs;try{fn(...args)}finally{
  if(id===window.benchCaptureTimer)benchCaptureTicks.push(performance.now()-t-(benchObserverMs-observer))
 }
},ms);return id};
</script>"""
                index = index.replace("<head>", "<head>" + instrumentation)
            # Replay the actual pre-fix protocol: only stats carried the recommendation.
            version_fixture = json.loads(json.dumps(fixture))
            if version == "before" and not args.before_dir:
                version_fixture["state"].pop("recommended_fps")
            script = "const FIXTURE=" + json.dumps(version_fixture) + ";\n"
            script += f"const BENCH_SECONDS={args.seconds}, BENCH_VERSION='{version}';\n" + replay
            (root / version / "replay.js").write_text(script)
            index = index.replace("</body>", f'<script src="/{version}/replay.js"></script></body>')
            (root / version / "index.html").write_text(index)

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(root), **kw)

            def log_message(self, *_):
                pass

            def do_POST(self):
                if self.path != "/results":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", 0))
                if length > 100_000:
                    self.send_error(413)
                    return
                result = json.loads(self.rfile.read(length))
                formatted = json.dumps(result, indent=2)
                print(formatted, flush=True)
                if args.output:
                    args.output.write_text(formatted + "\n")
                self.send_response(204)
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        print(f"Open http://127.0.0.1:{args.port}/ and click Run before / after", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
