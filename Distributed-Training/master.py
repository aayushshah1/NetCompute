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
import queue
import datetime
import socket
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("master.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger("MasterNode")

# Load configuration
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    logger.info("✅ Loaded configuration from config.json")
except Exception as e:
    logger.warning(f"⚠️ Error loading config.json: {e}")
    config = {
        "master": {
            "ip": "0.0.0.0",  # Use 0.0.0.0 to bind to all interfaces
            "task_port": "5555",
            "result_port": "5556",
            "heartbeat_port": "5557"
        },
        "batch_size": {
            "initial": 10,
            "min": 5,
            "max": 50
        },
        "load_balancing": {
            "enabled": True,
            "fairness_weight": 0.7  # 0.0 = pure efficiency, 1.0 = pure fairness
        }
    }

# Get master configuration
MASTER_IP = config["master"]["ip"]
TASK_PORT = config["master"]["task_port"]
RESULT_PORT = config["master"]["result_port"]
HEARTBEAT_PORT = config["master"]["heartbeat_port"]

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
        self.task_socket.bind(f"tcp://{MASTER_IP}:{TASK_PORT}")

        # Result Collector (PULL)
        self.result_socket = context.socket(zmq.PULL)
        self.result_socket.bind(f"tcp://{MASTER_IP}:{RESULT_PORT}")

        # Heartbeat Checker (SUB)
        self.heartbeat_socket = context.socket(zmq.SUB)
        self.heartbeat_socket.bind(f"tcp://{MASTER_IP}:{HEARTBEAT_PORT}")
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

    def calculate_batch_size(self, worker_id):
        """Calculate batch size based on worker efficiency and fairness."""
        if not LOAD_BALANCING_ENABLED:
            return INITIAL_BATCH_SIZE
        
        # Count active workers (seen in the last 10 seconds)
        current_time = time.time()
        active_workers = sum(1 for w in self.workers.values() 
                           if current_time - w.get("last_seen", 0) < 10)
        
        # If this is a new worker or we have very few workers, use initial size
        if worker_id not in self.worker_efficiency or active_workers <= 1:
            return INITIAL_BATCH_SIZE
        
        # Calculate remaining tasks
        remaining_tasks = max(0, self.total_tasks - self.completed_tasks)
        
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
        
        # Limit by min/max batch size
        batch_size = max(MIN_BATCH_SIZE, min(batch_size, MAX_BATCH_SIZE))
        
        # Limit by remaining tasks and number of active workers
        if active_workers > 0 and remaining_tasks > 0:
            max_fair_share = max(MIN_BATCH_SIZE, remaining_tasks // active_workers)
            batch_size = min(batch_size, max_fair_share)
        
        logger.debug(f"Calculated batch size for {worker_id}: {batch_size} " +
                    f"(eff={eff_ratio:.2f}, fair={fairness_ratio:.2f})")
        
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
        
        logger.info(f"Prepared batch of {len(batch)} tasks for worker {worker_id}")
        return batch

    def start_task_distribution(self):
        """Thread for distributing tasks to workers."""
        logger.info("Starting task distribution thread")
        
        while True:
            try:
                # Use poll with timeout to avoid blocking indefinitely
                poller = zmq.Poller()
                poller.register(self.task_socket, zmq.POLLIN)
                
                socks = dict(poller.poll(1000))  # 1 second timeout
                if self.task_socket in socks and socks[self.task_socket] == zmq.POLLIN:
                    worker_id_bytes, empty, request = self.task_socket.recv_multipart()
                    worker_id = worker_id_bytes.decode('utf-8')
                    
                    logger.info(f"Received task request from worker {worker_id}")
                    
                    # Process the request
                    if request == b"request_task":
                        # Get a batch of tasks for this worker
                        batch = self.get_worker_batch(worker_id)
                        
                        if batch:
                            logger.info(f"Sending batch of {len(batch)} tasks to worker {worker_id}")
                            # Send the batch to the worker
                            batch_json = json.dumps({"batch": batch}).encode()
                            self.task_socket.send_multipart([
                                worker_id_bytes, 
                                b"", 
                                batch_json
                            ])
                            logger.info(f"Batch sent to {worker_id}, size: {len(batch_json)} bytes")
                        else:
                            # No tasks left, send empty response
                            logger.info(f"No tasks available for worker {worker_id}")
                            self.task_socket.send_multipart([
                                worker_id_bytes, 
                                b"", 
                                json.dumps({"batch": []}).encode()
                            ])
                
                # Check if job is done
                if self.total_tasks > 0 and self.completed_tasks >= self.total_tasks:
                    if not self.completion_time:  # Only record completion time once
                        self.completion_time = time.time() - self.start_time
                        self.record_completion()
                        logger.info(f"✅ Job completed in {self.completion_time:.2f} seconds")
                        self.stats["completion_time"] = self.completion_time
                
                # Update stats for current state
                self.update_stats()
                time.sleep(0.01)  # Small sleep to reduce CPU usage
                
            except Exception as e:
                logger.error(f"⚠️ Error in task distribution: {str(e)}", exc_info=True)
                time.sleep(1)  # Wait before retrying

    def process_results(self):
        """Thread for processing results from workers."""
        logger.info("Starting result processing thread")
        
        while True:
            try:
                # Use poll with timeout to avoid blocking indefinitely
                poller = zmq.Poller()
                poller.register(self.result_socket, zmq.POLLIN)
                
                socks = dict(poller.poll(1000))  # 1 second timeout
                if self.result_socket in socks and socks[self.result_socket] == zmq.POLLIN:
                    # Process the result
                    result_data = self.result_socket.recv_json()
                    
                    # Update task tracking
                    self.completed_tasks += 1
                    
                    # Log the received result
                    worker_id = result_data.get("worker_id", "unknown")
                    task_id = result_data.get("task_id", "unknown")
                    logger.info(f"Received result for task {task_id} from worker {worker_id}")
                    
                    # Update worker tracking if worker_id provided
                    if worker_id and worker_id in self.worker_batches:
                        # Remove task from worker's batch
                        if task_id in self.worker_batches[worker_id]:
                            self.worker_batches[worker_id].remove(task_id)
                    
                    # Update worker efficiency if processing_time provided
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
                
                time.sleep(0.01)  # Small sleep to reduce CPU usage
                
            except Exception as e:
                logger.error(f"⚠️ Error processing results: {str(e)}", exc_info=True)
                time.sleep(1)  # Wait before retrying

    def process_heartbeats(self):
        """Thread for processing heartbeat messages from workers."""
        logger.info("Starting heartbeat processing thread")
        
        while True:
            try:
                # Use poll with timeout to avoid blocking indefinitely
                poller = zmq.Poller()
                poller.register(self.heartbeat_socket, zmq.POLLIN)
                
                socks = dict(poller.poll(1000))  # 1 second timeout
                if self.heartbeat_socket in socks and socks[self.heartbeat_socket] == zmq.POLLIN:
                    # Process the heartbeat
                    heartbeat_msg = self.heartbeat_socket.recv_string()
                    heartbeat_data = json.loads(heartbeat_msg)
                    worker_id = heartbeat_data.get("worker_id")
                    hostname = heartbeat_data.get("hostname", "unknown")
                    
                    if worker_id:
                        # Check if this is a new worker
                        if worker_id not in self.workers:
                            logger.info(f"New worker connected: {worker_id} from {hostname}")
                        
                        # Store worker information
                        self.workers[worker_id] = heartbeat_data
                
                time.sleep(0.01)  # Small sleep to reduce CPU usage
                
            except Exception as e:
                logger.error(f"⚠️ Error processing heartbeat: {str(e)}", exc_info=True)
                time.sleep(1)  # Wait before retrying

    def update_stats(self):
        """Update statistics for the dashboard."""
        if not self.start_time:
            return
            
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        
        # Calculate average CPU and memory usage
        workers_count = len(self.workers)
        if workers_count > 0:
            cpu_sum = sum(worker.get("cpu", 0) for worker in self.workers.values())
            memory_sum = sum(worker.get("memory", 0) for worker in self.workers.values())
            avg_cpu = cpu_sum / workers_count
            avg_memory = memory_sum / workers_count
        else:
            avg_cpu = 0
            avg_memory = 0
        
        # Calculate estimated completion time
        estimated_completion = None
        if self.completed_tasks > 0 and self.total_tasks > self.completed_tasks:
            # Calculate tasks per second
            tasks_per_second = self.completed_tasks / elapsed_time
            if tasks_per_second > 0:
                remaining_tasks = self.total_tasks - self.completed_tasks
                estimated_seconds = remaining_tasks / tasks_per_second
                estimated_completion = {
                    "remaining": estimated_seconds,
                    "estimated_total": elapsed_time + estimated_seconds
                }
        
        # Prepare worker stats for execution summary
        worker_stats = {}
        for worker_id, worker in self.workers.values():
            tasks_processed = worker.get("tasks_processed", 0)
            avg_task_time = worker.get("avg_task_time_ms", 0)
            worker_stats[worker_id] = {
                "tasks_processed": tasks_processed,
                "avg_time": avg_task_time,
            }
        
        # Calculate task distribution imbalance
        if workers_count > 1:
            tasks_per_worker = [w.get("tasks_processed", 0) for w in self.workers.values()]
            if tasks_per_worker:
                max_tasks = max(tasks_per_worker)
                min_tasks = min(tasks_per_worker)
                avg_tasks = sum(tasks_per_worker) / workers_count
                
                if avg_tasks > 0:
                    # Calculate percentage difference between max and avg
                    imbalance = ((max_tasks - min_tasks) / avg_tasks) * 100
                else:
                    imbalance = 0
            else:
                imbalance = 0
        else:
            imbalance = 0
        
        # Calculate average time per task across all workers
        total_processing_time = 0
        total_processed = 0
        for worker in self.workers.values():
            if "avg_task_time_ms" in worker and "tasks_processed" in worker:
                total_processing_time += worker["avg_task_time_ms"] * worker["tasks_processed"]
                total_processed += worker["tasks_processed"]
        
        avg_time_per_task = total_processing_time / max(1, total_processed)
        
        # Update execution summary
        self.execution_summary.update({
            "total_tasks": self.total_tasks,
            "worker_stats": worker_stats,
            "distribution_imbalance": imbalance,
            "avg_time_per_task": avg_time_per_task
        })
        
        # Update stats
        self.stats.update({
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "workers": self.workers,
            "workers_count": workers_count,
            "avg_cpu": avg_cpu,
            "avg_memory": avg_memory,
            "elapsed_time": elapsed_time,
            "estimated_completion_time": estimated_completion,
            "time_data": {"elapsed": elapsed_time, "timestamp": time.time()},
            "execution_summary": self.execution_summary
        })

    def record_completion(self):
        """Record job completion details to a file."""
        try:
            # Count active workers (seen in the last 10 seconds)
            current_time = time.time()
            active_workers = sum(1 for w in self.workers.values() 
                              if current_time - w.get("last_seen", 0) < 10)
            
            # Format the completion record
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = f"{timestamp}: {active_workers} workers - {self.total_tasks} tasks - {self.completion_time:.2f} seconds\n"
            
            # Append to completion_time.txt
            with open("completion_time.txt", "a") as f:
                f.write(record)
                
            logger.info(f"✅ Recorded completion time: {record.strip()}")
        except Exception as e:
            logger.error(f"⚠️ Error recording completion time: {str(e)}")

    def load_tasks(self, tasks):
        """Load tasks into the queue."""
        if self.start_time is None:
            self.start_time = time.time()
            self.execution_summary["start_time"] = self.start_time
            
        # Reset tracking for a new job
        self.total_tasks = len(tasks)
        self.completed_tasks = 0
        self.completion_time = None
        self.worker_batches = {}
        
        # Reset execution summary for new job
        self.execution_summary.update({
            "start_time": self.start_time,
            "end_time": None,
            "total_tasks": self.total_tasks,
            "worker_stats": {},
            "distribution_imbalance": 0
        })
        
        # Add tasks to the queue
        for task in tasks:
            self.task_queue.put(task)
            
        logger.info(f"✅ Loaded {self.total_tasks} tasks")

    def run(self):
        """Start all processing threads."""
        # Start threads for task distribution, result processing, and heartbeat monitoring
        threads = [
            threading.Thread(target=self.start_task_distribution, daemon=True),
            threading.Thread(target=self.process_results, daemon=True),
            threading.Thread(target=self.process_heartbeats, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
            
        logger.info("✅ Master node running - all threads started")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("👋 Shutting down master node")
            self.context.term()

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