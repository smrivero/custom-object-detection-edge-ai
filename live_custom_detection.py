#!/usr/bin/env python3
"""RealSense RGB + custom YOLO, served as MJPEG on the local network."""
import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit

PAGE = """<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RealSense · Mate y termo</title>
<style>body{margin:24px;background:#101820;color:#eef3f7;font:16px system-ui}
main{max-width:960px;margin:auto}img{width:100%;background:#000;border-radius:8px}
nav{display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin:16px 0}
a{color:#8ed5ff}input{vertical-align:middle}#status{line-height:1.6}</style>
<main><h1>RealSense · Mate y termo</h1>
<nav><label>Confianza mínima: <output id="value">0.40</output>
<input id="conf" type="range" min="0.05" max="0.95" step="0.05" value="0.40"></label>
<a href="/snapshot.jpg" download="mate-termo-live.jpg">Guardar captura</a></nav>
<img src="/stream.mjpg" alt="Cámara con detecciones">
<p id="status">Conectando…</p><p>FPS de cuadros procesados en la Jetson.</p></main>
<script>
const slider=document.getElementById('conf'), value=document.getElementById('value');
slider.oninput=()=>value.textContent=Number(slider.value).toFixed(2);
slider.onchange=async()=>{try{const r=await fetch('/settings?conf='+slider.value);
if(!r.ok)throw Error(await r.text());}catch(e){document.getElementById('status').textContent=e.message;}};
async function update(){try{const r=await fetch('/health',{cache:'no-store'});
const s=await r.json();const objects=Object.entries(s.counts).map(([k,v])=>k+': '+v).join(' · ');
document.getElementById('status').textContent=s.backend+' · '+s.fps.toFixed(1)+' FPS · '+
(objects||'Sin detecciones')+(s.stopped?' · Detenido':'');
if(document.activeElement!==slider){slider.value=s.conf;value.textContent=s.conf.toFixed(2);}
}catch(e){document.getElementById('status').textContent='Sin conexión con la Jetson';}}
update();setInterval(update,1000);
</script></html>"""


class State:
    def __init__(self, conf):
        self.condition = threading.Condition()
        self.conf = conf
        self.jpeg = None
        self.sequence = 0
        self.fps = 0.0
        self.counts = {}
        self.backend = 'Iniciando'
        self.stopped = False

    def publish(self, jpeg, fps, counts):
        with self.condition:
            self.jpeg, self.fps, self.counts = jpeg, fps, counts
            self.sequence += 1
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_bytes(self, status, mime, body):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path == '/':
                    self.send_bytes(200, 'text/html; charset=utf-8', PAGE.encode())
                elif parsed.path == '/health':
                    with state.condition:
                        data = dict(fps=state.fps, counts=state.counts, conf=state.conf,
                                    backend=state.backend, stopped=state.stopped)
                    self.send_bytes(200, 'application/json', json.dumps(data).encode())
                elif parsed.path == '/settings':
                    try:
                        conf = float(parse_qs(parsed.query)['conf'][0])
                        if not math.isfinite(conf) or not 0.01 <= conf <= 0.99:
                            raise ValueError()
                    except (KeyError, ValueError, IndexError):
                        self.send_bytes(400, 'text/plain', b'Confianza fuera de rango')
                        return
                    with state.condition:
                        state.conf = conf
                    self.send_bytes(200, 'application/json', b'{"ok":true}')
                elif parsed.path == '/snapshot.jpg':
                    with state.condition:
                        jpeg = state.jpeg
                    self.send_bytes(200 if jpeg else 503, 'image/jpeg' if jpeg else 'text/plain',
                                    jpeg or b'Todavia no hay imagen')
                elif parsed.path == '/stream.mjpg':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    last = 0
                    while True:
                        with state.condition:
                            state.condition.wait_for(
                                lambda: state.stopped or state.sequence != last, timeout=15)
                            if state.stopped:
                                return
                            jpeg, sequence = state.jpeg, state.sequence
                        if jpeg is None or sequence == last:
                            continue
                        last = sequence
                        header = b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                        self.wfile.write(header + str(len(jpeg)).encode() + b'\r\n\r\n')
                        self.wfile.write(jpeg + b'\r\n')
                        self.wfile.flush()
                else:
                    self.send_bytes(404, 'text/plain', b'No encontrado')
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Modelo .pt o .engine personalizado')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--conf', type=float, default=0.40)
    args = parser.parse_args()
    if not math.isfinite(args.conf) or not 0.01 <= args.conf <= 0.99:
        parser.error('--conf debe estar entre 0.01 y 0.99')

    import cv2
    import numpy as np
    import pyrealsense2 as rs
    from ultralytics import YOLO

    cv2.setNumThreads(1)
    weights = Path(__file__).resolve().parent / 'runs/mate-termo-v2/weights'
    path = args.model or (weights / 'best.engine' if (weights / 'best.engine').is_file()
                          else weights / 'best.pt')
    if not path.is_file():
        raise SystemExit(f'No existe el modelo: {path}')
    state = State(args.conf)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    server.daemon_threads = True
    pipeline = rs.pipeline()
    started = False
    serving = False
    try:
        def load(model_path):
            print(f'Cargando: {model_path}', flush=True)
            model = YOLO(str(model_path), task='detect')
            result = model.predict(np.zeros((480, 640, 3), dtype=np.uint8),
                                   imgsz=640, rect=False, device=0, verbose=False)[0]
            if {str(v).lower() for v in result.names.values()} != {'mate', 'termo'}:
                raise RuntimeError(f'Clases inesperadas: {result.names}')
            return model

        try:
            model = load(path)
        except Exception as error:
            if args.model or path.suffix != '.engine' or not (weights / 'best.pt').is_file():
                raise
            print(f'No se pudo abrir TensorRT: {error}\nUsando best.pt.', flush=True)
            path = weights / 'best.pt'
            model = load(path)
        state.backend = 'TensorRT' if path.suffix == '.engine' else 'PyTorch'

        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError('No se detecto ninguna RealSense.')
        device = devices[0]
        print('Camara:', device.get_info(rs.camera_info.name), flush=True)
        config = rs.config()
        config.enable_device(device.get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        pipeline.start(config)
        started = True
        for _ in range(15):
            pipeline.wait_for_frames(10000)

        threading.Thread(target=server.serve_forever, daemon=True).start()
        serving = True
        print(f'Visor: http://<JETSON_IP>:{args.port}\nCtrl+C para detener.', flush=True)
        previous = time.perf_counter()
        fps = 0.0
        last_log = previous
        while True:
            frames = pipeline.wait_for_frames(10000)
            color = frames.get_color_frame()
            if not color:
                continue
            image = np.asanyarray(color.get_data()).copy()
            with state.condition:
                conf = state.conf
            result = model.predict(image, imgsz=640, rect=False, conf=conf,
                                   device=0, verbose=False)[0]
            counts = dict(Counter(result.names[int(c)] for c in result.boxes.cls.tolist()))
            annotated = result.plot(line_width=2, font_size=14)
            cv2.putText(annotated, f'{fps:.1f} FPS | conf {conf:.2f}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            ok, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if not ok:
                raise RuntimeError('No se pudo codificar el cuadro.')
            now = time.perf_counter()
            current = 1.0 / max(now - previous, 1e-6)
            fps = current if not fps else 0.9 * fps + 0.1 * current
            previous = now
            state.publish(encoded.tobytes(), fps, counts)
            if now - last_log >= 5:
                print(f'{fps:.1f} FPS | {counts or "Sin detecciones"}', flush=True)
                last_log = now
    except KeyboardInterrupt:
        print('\nDetenido.', flush=True)
    finally:
        state.stop()
        if started:
            pipeline.stop()
        if serving:
            server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
