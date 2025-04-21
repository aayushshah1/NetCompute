import zmq
import threading
import time
import json
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ai_module import train_model, predict
import sys
import queue
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import socket

# Load configuration
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
except Exception as e:
    config = {
        "master": {
            "ip": "0.0.0.0",  # Use 0.0.0.0 to bind to all interfaces
            "task_port": "5555",
            "result_port": "5556",
            "heartbeat_port": "5557"
        },
        "dashboard": {
            "host": "0.0.0.0",
            "port": "5051"
        },
        "batch_size": {
            "initial": 10,
            "min": 5,
            "max": 50
        },
        "load_balancing": {
            "enabled": True,
            "fairness_weight": 0.7
        }
    }

# Get master configuration
MASTER_IP = config["master"]["ip"]
TASK_PORT = config["master"]["task_port"]
RESULT_PORT = config["master"]["result_port"]
HEARTBEAT_PORT = config["master"]["heartbeat_port"]
DASHBOARD_HOST = config.get("dashboard", {}).get("host", "0.0.0.0")
DASHBOARD_PORT = config.get("dashboard", {}).get("port", "5051")

# Get batch configuration
INITIAL_BATCH_SIZE = config.get("batch_size", {}).get("initial", 10)
MIN_BATCH_SIZE = config.get("batch_size", {}).get("min", 5)
MAX_BATCH_SIZE = config.get("batch_size", {}).get("max", 50)
LOAD_BALANCING_ENABLED = config.get("load_balancing", {}).get("enabled", True)
FAIRNESS_WEIGHT = config.get("load_balancing", {}).get("fairness_weight", 0.7)

# Global Configuration
context = zmq.Context()

# Initialize Flask app for dashboard
app = Flask(__name__)
app.config['SECRET_KEY'] = 'distributed-training-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class MasterNode:
    def __init__(self, mode="train"):
        self.mode = mode
        self.workers = {}
        self.worker_stats = {}
        self.unfinished_tasks = []
        self.results = {}
        self.total_tasks = 0
        self.completed_tasks = 0
        self.running = True
        self.start_time = time.time()
        self.completion_time = None
        
        # Initialize task queue, worker tracking for batch processing
        self.task_queue = queue.Queue()
        self.worker_batches = {}
        self.worker_efficiency = {}
        self.stats = {}
        self.execution_summary = {
            "start_time": self.start_time,
            "end_time": None,
            "total_tasks": 0,
            "worker_stats": {},
            "distribution_imbalance": 0,
            "avg_time_per_task": 0
        }
        
        # Load dataset based on mode
        if self.mode == "train":
            self.load_train_data()
        else:
            self.load_test_data()

        # Task Distributor (ROUTER)
        self.task_socket = context.socket(zmq.ROUTER)
        self.task_socket.bind(f"tcp://{MASTER_IP}:{TASK_PORT}")

        # Result Collector (PULL)
        self.result_socket = context.socket(zmq.PULL)
        self.result_socket.bind(f"tcp://{MASTER_IP}:{RESULT_PORT}")

        # Heartbeat Checker (SUB)
        self.heartbeat_socket = context.socket(zmq.SUB)
        self.heartbeat_socket.bind(f"tcp://{MASTER_IP}:{HEARTBEAT_PORT}")
        self.heartbeat_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(f"✅ Master: {self.mode.upper()} Server Started on ports {TASK_PORT}/{RESULT_PORT}/{HEARTBEAT_PORT}")

    def load_train_data(self):
        """Loads preprocessed Titanic dataset for training."""
        df = pd.read_csv(os.path.join("Data", "augmented_dataset.csv"))
        
        # Extract PassengerId before dropping it
        self.passenger_ids = df["PassengerId"].tolist()

        # Drop PassengerId and Survived columns
        X = df.drop(["PassengerId", "Survived"], axis=1)
        y = df["Survived"]

        # Feature Scaling
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Store tasks
        self.unfinished_tasks = [
            {"task_id": i, "passenger_id": self.passenger_ids[i], "data": X.iloc[i].tolist(), "label": int(y.iloc[i])}
            for i in range(len(y))
        ]
        
        # Load tasks into queue for batch processing
        for task in self.unfinished_tasks:
            self.task_queue.put(task)

        self.total_tasks = len(self.unfinished_tasks)
        print(f"✅ Loaded {self.total_tasks} training samples.")

    def load_test_data(self):
        """Loads preprocessed Titanic dataset for testing."""
        df = pd.read_csv(os.path.join("Data", "processed_test.csv"))
        # Extract PassengerId before dropping it
        self.passenger_ids = df["PassengerId"].tolist()

        # Drop PassengerId (NO Survived labels in test set)
        X_test = df.drop(["PassengerId"], axis=1)

        # Feature Scaling (Same as in training)
        scaler = StandardScaler()
        X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

        # Store test tasks
        self.unfinished_tasks = [
            {"task_id": i, "passenger_id": self.passenger_ids[i], "data": X_test.iloc[i].tolist()}
            for i in range(len(X_test))
        ]
        
        # Load tasks into queue for batch processing
        for task in self.unfinished_tasks:
            self.task_queue.put(task)

        self.total_tasks = len(self.unfinished_tasks)
        print(f"✅ Loaded {self.total_tasks} test samples.")

    def calculate_batch_size(self, worker_id):
        """Calculate batch size based on worker efficiency and fairness."""
        if not LOAD_BALANCING_ENABLED:
            return INITIAL_BATCH_SIZE
        
        # Default to initial batch size for new workers
        if worker_id not in self.worker_efficiency:
            return INITIAL_BATCH_SIZE
        
        # Count active workers
        current_time = time.time()
        active_workers = sum(1 for w in self.workers.values() 
                          if current_time - w.get("last_seen", 0) < 10)
        
        if active_workers <= 1:
            return INITIAL_BATCH_SIZE
        
        # Calculate base batch size from worker efficiency
        worker_eff = self.worker_efficiency[worker_id]
        avg_eff = sum(self.worker_efficiency.values()) / max(1, len(self.worker_efficiency))
        eff_ratio = worker_eff / max(0.1, avg_eff)  # Avoid division by zero
        
        # Adjust for tasks processed so far (fairness)
        tasks_processed = self.workers.get(worker_id, {}).get("tasks_processed", 0)
        avg_tasks = self.completed_tasks / max(1, len(self.workers))
        fairness_ratio = min(1.5, avg_tasks / max(1, tasks_processed))  # Cap at 1.5x
        
        # Combine efficiency and fairness based on fairness weight
        combined_factor = (eff_ratio * (1 - FAIRNESS_WEIGHT)) + (fairness_ratio * FAIRNESS_WEIGHT)
        
        # Calculate batch size with constraints
        batch_size = int(INITIAL_BATCH_SIZE * combined_factor)
        batch_size = max(MIN_BATCH_SIZE, min(batch_size, MAX_BATCH_SIZE))
        
        # Limit by remaining tasks
        remaining_tasks = max(0, self.total_tasks - self.completed_tasks)
        if active_workers > 0:
            max_fair_share = max(MIN_BATCH_SIZE, remaining_tasks // active_workers)
            batch_size = min(batch_size, max_fair_share)
        
        return batch_size

    def get_worker_batch(self, worker_id):
        """Get a batch of tasks for a worker."""
        if self.task_queue.empty():
            return []
        
        batch_size = self.calculate_batch_size(worker_id)
        batch = []
        
        for _ in range(batch_size):
            if self.task_queue.empty():
                break
            try:
                task = self.task_queue.get_nowait()
                batch.append(task)
            except queue.Empty:
                break
        
        # Record the batch for tracking
        if worker_id not in self.worker_batches:
            self.worker_batches[worker_id] = []
        self.worker_batches[worker_id].extend([task["task_id"] for task in batch])
        
        return batch

    def handle_workers(self):
        """Thread for handling worker heartbeats."""
        last_update_time = 0
        
        while self.running:
            try:
                # Use NOBLOCK to prevent blocking the thread
                worker_info = json.loads(self.heartbeat_socket.recv_string(flags=zmq.NOBLOCK))
                worker_id = worker_info['worker_id']
                
                # Store worker info
                self.workers[worker_id] = worker_info
                
                # Store stats for dashboard
                self.worker_stats[worker_id] = {
                    "cpu": worker_info.get("cpu", 0),
                    "memory": worker_info.get("memory", 0),
                    "tasks_processed": worker_info.get("tasks_processed", 0),
                    "last_seen": worker_info.get("last_seen", time.time()),
                    "current_task": worker_info.get("current_task", None),
                    "processing_rate": worker_info.get("processing_rate", 0)
                }
                
                current_time = time.time()
                
                # Only send updates every 1 second to avoid flooding
                if current_time - last_update_time >= 1.0:
                    # Calculate stats
                    workers_count = len(self.worker_stats)
                    if workers_count > 0:
                        # Send stats to dashboard
                        try:
                            elapsed_time = time.time() - self.start_time
                            stats_data = {
                                "workers": self.worker_stats,
                                "completed_tasks": self.completed_tasks,
                                "total_tasks": self.total_tasks,
                                "workers_count": workers_count,
                                "elapsed_time": elapsed_time
                            }
                            
                            # Add completion time if available
                            if self.completion_time:
                                stats_data["completion_time"] = self.completion_time
                            
                            # Save stats for persistence
                            with open("last_stats.json", "w") as f:
                                json.dump(stats_data, f)
                                
                            socketio.emit('update_stats', stats_data)
                        except Exception:
                            pass
                        
                        last_update_time = current_time
                    
            except zmq.Again:
                pass  # No data available
            except Exception:
                pass  # Ignore other errors
            
            # Remove inactive workers
            current_time = time.time()
            inactive_workers = [worker for worker, info in list(self.worker_stats.items())
                              if current_time - info.get("last_seen", current_time) > 10]

            for worker_id in inactive_workers:
                if worker_id in self.workers:
                    del self.workers[worker_id]
                if worker_id in self.worker_stats:
                    del self.worker_stats[worker_id]
                    
            time.sleep(0.01)

    def start_task_distribution(self):
        """Thread for distributing tasks to workers."""
        while self.running:
            try:
                # Use poll with timeout to avoid blocking indefinitely
                poller = zmq.Poller()
                poller.register(self.task_socket, zmq.POLLIN)
                
                socks = dict(poller.poll(100))  # 100ms timeout
                if self.task_socket in socks and socks[self.task_socket] == zmq.POLLIN:
                    worker_id_bytes, empty, request = self.task_socket.recv_multipart()
                    worker_id = worker_id_bytes.decode('utf-8')
                    
                    # Process the request for tasks
                    if request == b"request_task":
                        batch = self.get_worker_batch(worker_id)
                        
                        # Send the batch to the worker
                        batch_json = json.dumps({"batch": batch}).encode()
                        self.task_socket.send_multipart([
                            worker_id_bytes, 
                            b"", 
                            batch_json
                        ])
                
                # Check if job is done
                if self.total_tasks > 0 and self.completed_tasks >= self.total_tasks:
                    if not self.completion_time:
                        self.completion_time = time.time() - self.start_time
                        print(f"✅ Job completed in {self.completion_time:.2f} seconds")
                        
                        # Save the completion time to a file
                        worker_count = len(self.worker_stats)
                        with open("completion_time.txt", "a") as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {worker_count} workers - {self.total_tasks} tasks - {self.completion_time:.2f} seconds\n")
                
                time.sleep(0.001)  # Minimal sleep
                
            except Exception:
                time.sleep(0.01)  # Wait before retrying

    def process_results(self):
        """Thread for processing results from workers."""
        while self.running:
            try:
                # Use poll with timeout to avoid blocking
                poller = zmq.Poller()
                poller.register(self.result_socket, zmq.POLLIN)
                
                socks = dict(poller.poll(100))  # 100ms timeout
                if self.result_socket in socks and socks[self.result_socket] == zmq.POLLIN:
                    # Process the result
                    result_data = self.result_socket.recv_json()
                    
                    # Update task tracking
                    self.completed_tasks += 1
                    
                    # Update worker efficiency if processing_time provided
                    worker_id = result_data.get("worker_id", "unknown")
                    if worker_id and "processing_time" in result_data:
                        processing_time = result_data["processing_time"]
                        if processing_time > 0:
                            # Use exponential moving average for efficiency
                            eff = 1.0 / processing_time  # tasks per second
                            if worker_id not in self.worker_efficiency:
                                self.worker_efficiency[worker_id] = eff
                            else:
                                # 80% old value, 20% new value
                                self.worker_efficiency[worker_id] = 0.8 * self.worker_efficiency[worker_id] + 0.2 * eff
                    
                    # Save predictions if any
                    prediction = result_data.get('prediction', [None])[0]
                    passenger_id = result_data.get('passenger_id')
                    
                    if passenger_id is not None and prediction is not None:
                        # Save to file
                        df = pd.DataFrame({"PassengerId": [passenger_id], "Survived": [prediction]})
                        filename = os.path.join("Data", f"submission_{self.mode}.csv")
                        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
                
                time.sleep(0.001)  # Minimal sleep
                
            except Exception:
                time.sleep(0.01)  # Wait before retrying

    def run(self):
        """Start all processing threads."""
        # Start dashboard first
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        # Start worker threads
        threads = [
            threading.Thread(target=self.handle_workers, daemon=True),
            threading.Thread(target=self.start_task_distribution, daemon=True),
            threading.Thread(target=self.process_results, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        print(f"📊 Dashboard available at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
        
        # Process keyboard input for graceful shutdown
        try:
            while self.running:
                cmd = input()
                if cmd.lower() == 'q':
                    print("🛑 Master: Initiating shutdown...")
                    self.running = False
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user. Shutting down...")
            self.running = False
        finally:
            # Clean up
            self.task_socket.close()
            self.result_socket.close()
            self.heartbeat_socket.close()
            context.term()
            print("✅ Master shutdown complete.")
            os._exit(0)

# Flask routes for dashboard
@app.route('/')
def index():
    """Serve the dashboard HTML page."""
    return render_template('dashboard.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection to dashboard."""
    # Send current stats to newly connected client
    try:
        if os.path.exists("last_stats.json"):
            with open("last_stats.json", "r") as f:
                stats = json.load(f)
                socketio.emit('update_stats', stats, room=request.sid)
    except Exception:
        pass

def run_dashboard():
    """Run the Flask dashboard."""
    try:
        socketio.run(app, host=DASHBOARD_HOST, port=int(DASHBOARD_PORT), 
                     debug=False, allow_unsafe_werkzeug=True, log_output=False)
    except Exception:
        print("⚠️ Could not start dashboard. Continuing without it.")

if __name__ == "__main__":
    # Check for templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    if not os.path.exists(templates_dir):
        print(f"⚠️ Templates directory not found. Creating it...")
        try:
            os.makedirs(templates_dir, exist_ok=True)
        except Exception:
            pass
    
    # Check for dashboard file
    dashboard_file = os.path.join(templates_dir, "dashboard.html")
    if not os.path.exists(dashboard_file):
        print(f"⚠️ Dashboard HTML file not found.")
    
    print("🚀 Starting Master Node")
    print("Select mode: 1) Train (default), 2) Test")
    
    mode_choice = input("Enter choice (1-2) or press Enter for default: ")
    
    mode = "test" if mode_choice == "2" else "train"
    
    try:
        MasterNode(mode).run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Shutting down...")
        context.term()
        sys.exit(0)