import zmq
import threading
import time
import json
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ai_module import train_model, predict  # ✅ Use single AI module
import sys

# Add Flask and SocketIO imports for dashboard
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import logging
import signal

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("master")

# Load configuration
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    print("✅ Loaded configuration from config.json")
except Exception as e:
    print(f"⚠️ Error loading config.json: {e}")
    # Default configuration
    config = {
        "master": {
            "ip": "localhost",
            "task_port": "5555",
            "result_port": "5556",
            "heartbeat_port": "5557"
        },
        "dashboard": {
            "host": "localhost",
            "port": "5051"
        }
    }

# Get ports from config
TASK_PORT = config["master"]["task_port"]
RESULT_PORT = config["master"]["result_port"]
HEARTBEAT_PORT = config["master"]["heartbeat_port"]
DASHBOARD_HOST = config["dashboard"]["host"]
DASHBOARD_PORT = config["dashboard"]["port"]

# Global Configuration
context = zmq.Context()

# Initialize Flask app for dashboard
app = Flask(__name__)
app.config['SECRET_KEY'] = 'distributed-training-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class MasterNode:
    def __init__(self, mode="train"):
        self.mode = mode  # ✅ 'train' or 'test'
        self.workers = {}
        self.unfinished_tasks = []
        self.results = {}
        self.total_tasks = 0
        self.completed_tasks = 0
        self.running = True  # Flag for graceful shutdown
        self.worker_stats = {}  # Store worker resource stats
        self.start_time = time.time()  # Record start time for performance metrics
        self.completion_time = None  # Store completion time when all tasks are done
        
        # ✅ Load dataset based on mode
        if self.mode == "train":
            self.load_train_data()
        else:
            self.load_test_data()

        # Task Distributor (ROUTER)
        self.task_socket = context.socket(zmq.ROUTER)
        self.task_socket.bind(f"tcp://*:{TASK_PORT}")

        # Result Collector (PULL)
        self.result_socket = context.socket(zmq.PULL)
        self.result_socket.bind(f"tcp://*:{RESULT_PORT}")

        # Heartbeat Checker (SUB)
        self.heartbeat_socket = context.socket(zmq.SUB)
        self.heartbeat_socket.bind(f"tcp://*:{HEARTBEAT_PORT}")
        self.heartbeat_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(f"✅ Master: {self.mode.upper()} Server Started... Waiting for workers.")

    def load_train_data(self):
        """Loads preprocessed Titanic dataset for training."""
        df = pd.read_csv(os.path.join("Data", "augmented_dataset.csv"))
        
        # Extract PassengerId before dropping it
        self.passenger_ids = df["PassengerId"].tolist()

        # Drop PassengerId and Survived columns
        X = df.drop(["PassengerId", "Survived"], axis=1)
        y = df["Survived"]

        # ✅ Feature Scaling
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # ✅ Store tasks
        self.unfinished_tasks = [
            {"task_id": i, "passenger_id": self.passenger_ids[i], "data": X.iloc[i].tolist(), "label": int(y.iloc[i])}
            for i in range(len(y))
        ]

        self.total_tasks = len(self.unfinished_tasks)
        print(f"✅ Loaded {self.total_tasks} training samples.")

    def load_test_data(self):
        """Loads preprocessed Titanic dataset for testing."""
        df = pd.read_csv(os.path.join("Data", "processed_test.csv"))
        # Extract PassengerId before dropping it
        self.passenger_ids = df["PassengerId"].tolist()

        # Drop PassengerId (NO Survived labels in test set)
        X_test = df.drop(["PassengerId"], axis=1)

        # ✅ Feature Scaling (Same as in training)
        scaler = StandardScaler()
        X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

        # ✅ Store test tasks
        self.unfinished_tasks = [
            {"task_id": i, "passenger_id": self.passenger_ids[i], "data": X_test.iloc[i].tolist()}
            for i in range(len(X_test))
        ]

        self.total_tasks = len(self.unfinished_tasks)
        print(f"✅ Loaded {self.total_tasks} test samples.")

    def estimate_completion_time(self, elapsed_time):
        """Estimate total completion time based on progress so far."""
        if self.completed_tasks == 0 or self.total_tasks == 0:
            return {
                "elapsed": elapsed_time,
                "estimated_total": 0,
                "remaining": 0
            }
        
        progress = self.completed_tasks / self.total_tasks
        if progress > 0:
            total_estimated_time = elapsed_time / progress
            remaining_time = total_estimated_time - elapsed_time
            return {
                "elapsed": elapsed_time,
                "estimated_total": total_estimated_time,
                "remaining": max(0, remaining_time)  # Ensure it's not negative
            }
        return {
            "elapsed": elapsed_time,
            "estimated_total": 0,
            "remaining": 0
        }

    def handle_workers(self):
        """Handle heartbeat monitoring from workers."""
        last_update_time = 0  # Track the last time we sent an update
        
        while self.running:
            try:
                # Use NOBLOCK to prevent blocking the thread
                worker_info = json.loads(self.heartbeat_socket.recv_string(flags=zmq.NOBLOCK))
                worker_id = worker_info['worker_id']
                
                # Store full worker info
                self.workers[worker_id] = worker_info
                
                # Format and store stats for dashboard
                self.worker_stats[worker_id] = {
                    "cpu": worker_info.get("cpu", 0),
                    "memory": worker_info.get("memory", 0),
                    "tasks_processed": worker_info.get("tasks_processed", 0),
                    "last_seen": worker_info.get("last_seen", time.time()),
                    "current_task": worker_info.get("current_task", None),
                    "processing_rate": worker_info.get("processing_rate", 0),
                    "uptime": worker_info.get("uptime", 0)
                }
                
                current_time = time.time()
                
                # Only send updates every 0.5 seconds to avoid flooding
                if current_time - last_update_time >= 0.5:
                    # Calculate worker performance metrics
                    workers_count = len(self.worker_stats)  # Count by unique worker IDs
                    if workers_count > 0:
                        total_cpu = sum(w.get("cpu", 0) for w in self.worker_stats.values())
                        total_memory = sum(w.get("memory", 0) for w in self.worker_stats.values())
                        
                        # Calculate elapsed time since start
                        elapsed_time = time.time() - self.start_time
                        
                        # Update dashboard with stats via SocketIO
                        try:
                            stats_data = {
                                "workers": self.worker_stats,
                                "completed_tasks": self.completed_tasks,
                                "total_tasks": self.total_tasks,
                                "avg_cpu": total_cpu / max(1, workers_count),
                                "avg_memory": total_memory / max(1, workers_count),
                                "workers_count": workers_count,
                                "elapsed_time": elapsed_time
                            }
                            
                            # Add either estimated completion time or actual completion time
                            if self.completion_time:
                                stats_data["completion_time"] = self.completion_time
                            else:
                                stats_data["estimated_completion_time"] = self.estimate_completion_time(elapsed_time)
                            
                            print(f"📊 Sending update to dashboard: {workers_count} workers, {self.completed_tasks}/{self.total_tasks} tasks")
                            socketio.emit('update_stats', stats_data)
                            
                            # Save stats to file for persistence
                            try:
                                with open("last_stats.json", "w") as f:
                                    json.dump(stats_data, f)
                            except Exception as e:
                                print(f"⚠️ Error saving stats: {e}")
                                
                        except Exception as e:
                            print(f"⚠️ Error sending stats to dashboard: {e}")
                        
                        last_update_time = current_time
                    
            except zmq.Again:
                pass  # No data available at the moment
            except Exception as e:
                print(f"⚠️ Error in handle_workers: {e}")
            
            # Remove inactive workers
            current_time = time.time()
            inactive_workers = [worker for worker, info in list(self.worker_stats.items())
                               if current_time - info.get("last_seen", current_time) > 10]

            for worker_id in inactive_workers:
                print(f"➖ Worker disconnected: {worker_id}")
                if worker_id in self.workers:
                    del self.workers[worker_id]
                if worker_id in self.worker_stats:
                    del self.worker_stats[worker_id]
                    
            time.sleep(0.1)  # Small sleep to avoid excessive CPU usage

    def distribute_tasks(self):
        """Assign tasks (train or test) to available workers."""
        while self.unfinished_tasks and self.running:
            try:
                worker_id, _, request = self.task_socket.recv_multipart(flags=zmq.NOBLOCK)
                worker_id_str = worker_id.decode() if isinstance(worker_id, bytes) else worker_id

                if self.unfinished_tasks:
                    task = self.unfinished_tasks.pop(0)
                    self.task_socket.send_multipart([worker_id, b"", json.dumps(task).encode()])
                    print(f"📤 Assigned Task ID {task['task_id']} to {worker_id_str}")
            except zmq.Again:
                time.sleep(0)

    def collect_results(self):
        """Wait for all workers to send back results."""
        received_task_ids = set()
        start_time = time.time()

        while len(received_task_ids) < self.total_tasks and self.running:
            try:
                result = self.result_socket.recv_json(flags=zmq.NOBLOCK)
                print(f"📥 Master Received: {result}")

                received_task_ids.add(result["task_id"])
                self.completed_tasks += 1
                progress = (self.completed_tasks / self.total_tasks) * 100
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Check if we've completed all tasks
                if self.completed_tasks == self.total_tasks:
                    self.completion_time = elapsed_time
                    print(f"✅ All tasks completed in {self.completion_time:.2f} seconds")
                
                print(f"📊 Progress: {self.completed_tasks}/{self.total_tasks} tasks completed ({progress:.1f}%)")
                
                self.save_results(result)
            except zmq.Again:
                time.sleep(0)
                
                # Periodically update stats with elapsed time every second
                if hasattr(self, 'last_stats_update'):
                    if time.time() - self.last_stats_update >= 1.0:
                        elapsed_time = time.time() - start_time
                        self.last_stats_update = time.time()
                else:
                    self.last_stats_update = time.time()

        if not self.running:
            print("⚠️ Processing interrupted by user.")
        else:
            final_time = self.completion_time or (time.time() - start_time)
            print(f"🎉 All {self.total_tasks} tasks completed successfully in {final_time:.2f} seconds!")
            
            # Save the completion time to a file for reference
            with open("completion_time.txt", "a") as f:
                worker_count = len(self.worker_stats)
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {worker_count} workers - {self.total_tasks} tasks - {final_time:.2f} seconds\n")

    def save_results(self, result):
        """Save predictions into submission file based on mode."""
        prediction = result['prediction'][0] if result['prediction'] and isinstance(result['prediction'], list) else -1

        df = pd.DataFrame({"PassengerId": [result['passenger_id']], "Survived": [prediction]})

        # ✅ Save to different files for train & test
        filename = os.path.join("Data", "submission_train.csv") if self.mode == "train" else os.path.join("Data", "submission_test.csv")        
        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

        print(f"✅ Saved Passenger ID {result['passenger_id']} with prediction: {prediction}")

    def monitor_keyboard(self):
        """Listen for keyboard input to shut down master gracefully."""
        while self.running:
            cmd = input()
            if cmd.lower() == 'q':
                print("🛑 Master: Initiating shutdown...")
                self.running = False
                break
        self.cleanup()

    def run(self):
        # Start dashboard thread first
        dashboard_thread = threading.Thread(target=run_dashboard)
        dashboard_thread.daemon = True  # Make sure it closes when main thread exits
        dashboard_thread.start()
        
        # Start worker handler
        threading.Thread(target=self.handle_workers, daemon=True).start()
        threading.Thread(target=self.monitor_keyboard, daemon=True).start()
        
        print("💡 Dashboard is now running internally")
        print(f"📊 Dashboard available at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
        
        # Give a moment for everything to initialize
        time.sleep(1)

        distribution_thread = threading.Thread(target=self.distribute_tasks)
        distribution_thread.start()
        
        try:
            self.collect_results()
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user. Shutting down...")
            self.running = False
        finally:
            distribution_thread.join(timeout=1)
            self.cleanup()
            
    def cleanup(self):
        """Gracefully terminate the master node."""
        print("👋 Master node shutting down...")
        time.sleep(1)

        # ✅ Close all sockets before terminating
        self.task_socket.close()
        self.result_socket.close()
        self.heartbeat_socket.close()

        context.term()  # ✅ Terminate ZeroMQ context

        print("✅ Master shutdown complete.")
        os._exit(0)  # ✅ Ensure proper exit

# Flask routes for dashboard
@app.route('/')
def index():
    """Serve the dashboard HTML page."""
    print("📊 Serving dashboard page")
    return render_template('dashboard.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"🔌 Client connected to dashboard: {request.sid}")
    
    # Send current stats to newly connected client
    try:
        if os.path.exists("last_stats.json"):
            with open("last_stats.json", "r") as f:
                stats = json.load(f)
                socketio.emit('update_stats', stats, room=request.sid)
    except Exception as e:
        print(f"⚠️ Error loading last stats: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"🔌 Client disconnected from dashboard: {request.sid}")

def run_dashboard():
    """Run the Flask dashboard."""
    try:
        print(f"🌐 Starting dashboard at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
        socketio.run(app, host=DASHBOARD_HOST, port=int(DASHBOARD_PORT), 
                     debug=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"⚠️ Error in dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 Checking for templates directory...")
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    if not os.path.exists(templates_dir):
        print(f"⚠️ Templates directory not found at {templates_dir}! Creating it...")
        try:
            os.makedirs(templates_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Error creating templates directory: {e}")
    
    dashboard_file = os.path.join(templates_dir, "dashboard.html")
    if not os.path.exists(dashboard_file):
        print(f"⚠️ Dashboard HTML file not found at {dashboard_file}!")
        print("💡 Make sure you've created the dashboard.html file in the templates directory.")
    
    print("🚀 Starting Master Node")
    print("💡 Select mode:")
    print("   1) Train (default)")
    print("   2) Test")
    
    mode_choice = input("Enter choice (1-2) or press Enter for default: ")
    
    if mode_choice == "2":
        mode = "test"
    else:
        mode = "train"
    
    try:
        MasterNode(mode).run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Shutting down...")
        context.term()
        sys.exit(0)