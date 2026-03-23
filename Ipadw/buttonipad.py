#!/usr/bin/env python3
from flask import Flask, render_template_string, request, redirect, url_for
import threading


class ZoomWebInterface:
    def __init__(
        self,
        meeting_url,
        host="0.0.0.0",
        port=5000,
        command_callback=None,
        state_callback=None
    ):
        self.meeting_url = meeting_url
        self.host = host
        self.port = port
        self.command_callback = command_callback
        self.state_callback = state_callback

        self.started = False
        self.web_thread = None
        self.app = Flask(__name__)

        self.setup_routes()

    def get_state(self):
        if self.state_callback is not None:
            return self.state_callback()

        return {
            "current_state": "waiting",
            "status_message": "Waiting for nurse command",
            "target_room": None,
            "patient_verified": False,
            "zoom_ready": False
        }

    def render_home_page(self):
        state = self.get_state()

        current_state = state.get("current_state", "waiting")
        status_message = state.get("status_message", "Waiting for nurse command")
        target_room = state.get("target_room", None)
        zoom_ready = state.get("zoom_ready", False)

        room_label = "None"
        if target_room == "room1":
            room_label = "Room 1"
        elif target_room == "room2":
            room_label = "Room 2"
        elif target_room == "nurse_station":
            room_label = "Nurse Station"

        zoom_button_html = ""
        if current_state == "ready_for_zoom" and zoom_ready:
            zoom_button_html = f"""
            <form action="/start_zoom" method="post">
                <button class="btn btn-zoom" type="submit">Tap to Join Zoom</button>
            </form>
            """

        return render_template_string(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Hospital Robot Interface</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="2">
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #eaf4ff, #f7fbff);
            text-align: center;
            padding: 30px;
            margin: 0;
        }}

        .card {{
            background: white;
            border-radius: 20px;
            padding: 30px;
            max-width: 760px;
            margin: auto;
            box-shadow: 0 8px 24px rgba(0,0,0,0.10);
        }}

        .badge {{
            display: inline-block;
            background: #d9f2e6;
            color: #146c43;
            padding: 8px 14px;
            border-radius: 999px;
            font-size: 0.95rem;
            margin-bottom: 18px;
            font-weight: bold;
        }}

        h1 {{
            font-size: 2.1rem;
            color: #113b67;
            margin-bottom: 10px;
        }}

        p {{
            font-size: 1.15rem;
            color: #444;
            margin-bottom: 22px;
        }}

        .status-box {{
            background: #f4f8fc;
            border: 1px solid #d8e6f3;
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 24px;
            font-size: 1.05rem;
            color: #1f3b57;
            line-height: 1.6;
        }}

        .btn {{
            display: block;
            width: 90%;
            margin: 14px auto;
            padding: 18px 24px;
            border: none;
            border-radius: 14px;
            font-size: 1.15rem;
            font-weight: bold;
            cursor: pointer;
            color: white;
        }}

        .btn-room1 {{
            background: #007aff;
        }}

        .btn-room2 {{
            background: #34a853;
        }}

        .btn-home {{
            background: #6c757d;
        }}

        .btn-zoom {{
            background: #0b5ed7;
        }}

        .btn:hover {{
            opacity: 0.93;
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="badge">Autonomous Telepresence Robot</div>
        <h1>Hospital Robot Interface</h1>
        <p>Where would you like to send the robot?</p>

        <div class="status-box">
            <strong>Status:</strong> {status_message}<br>
            <strong>State:</strong> {current_state}<br>
            <strong>Destination:</strong> {room_label}
        </div>

        <form action="/send_command" method="post">
            <button class="btn btn-room1" type="submit" name="destination" value="room1">
                Go to Room 1
            </button>

            <button class="btn btn-room2" type="submit" name="destination" value="room2">
                Go to Room 2
            </button>

            <button class="btn btn-home" type="submit" name="destination" value="nurse_station">
                Back to Nurse Station
            </button>
        </form>

        {zoom_button_html}
    </div>
</body>
</html>
        """)

    def setup_routes(self):
        @self.app.route("/")
        def home():
            return self.render_home_page()

        @self.app.route("/send_command", methods=["POST"])
        def send_command():
            destination = request.form.get("destination")

            if self.command_callback is not None and destination is not None:
                self.command_callback(destination)

            return redirect(url_for("home"))

        @self.app.route("/start_zoom", methods=["POST"])
        def start_zoom():
            if self.command_callback is not None:
                self.command_callback("start_zoom")

            return redirect(self.meeting_url)

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
