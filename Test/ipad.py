#!/usr/bin/env python3
from flask import Flask, render_template_string
import threading


class ZoomWebInterface:
    def __init__(self, meeting_url, host="0.0.0.0", port=5000):
        self.meeting_url = meeting_url
        self.host = host
        self.port = port
        self.started = False
        self.web_thread = None

        self.face_detected = False
        self.status_text = "Waiting for patient..."
        self.lock = threading.Lock()

        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route("/")
        def home():
            with self.lock:
                face_detected = self.face_detected
                status_text = self.status_text

            html_page = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Robot Zoom Link</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="2">
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #f4f4f4;
            text-align: center;
            padding: 40px;
        }}
        .card {{
            background: white;
            border-radius: 16px;
            padding: 30px;
            max-width: 700px;
            margin: auto;
            box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}
        p {{
            font-size: 1.2rem;
        }}
        .status {{
            font-size: 1.35rem;
            margin-top: 20px;
            margin-bottom: 20px;
        }}
        .waiting {{
            color: #666;
        }}
        .ready {{
            color: #1a7f37;
            font-weight: bold;
        }}
        .join-btn {{
            display: inline-block;
            margin-top: 20px;
            padding: 18px 28px;
            background: #007aff;
            color: white;
            text-decoration: none;
            border-radius: 12px;
            font-size: 1.3rem;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Hospital Robot Interface</h1>
        <p class="status {'ready' if face_detected else 'waiting'}">{status_text}</p>
        {
            f'<a class="join-btn" href="{self.meeting_url}" target="_blank">Tap to Join Zoom</a>'
            if face_detected else
            '<p>Please wait while the robot looks for the patient.</p>'
        }
    </div>
</body>
</html>
"""
            return render_template_string(html_page)

    def set_face_detected(self, detected):
        with self.lock:
            self.face_detected = detected
            if detected:
                self.status_text = "Doctor available — tap to connect"
            else:
                self.status_text = "Waiting for patient..."

    def set_status(self, text):
        with self.lock:
            self.status_text = text

    def run_server(self):
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

    def start(self):
        if self.started:
            print("Web interface already running")
            return

        self.web_thread = threading.Thread(target=self.run_server, daemon=True)
        self.web_thread.start()
        self.started = True
        print(f"Web interface started at http://<PI_IP>:{self.port}")
