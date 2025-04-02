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


# Import dashboard functions - moved outside of the class
try:
    from dashboard import update_stats
    has_dashboard = True
    print("✅ Successfully imported dashboard module")
except ImportError as e:
    print(f"⚠️ Error importing dashboard module: {e}")
    has_dashboard = False
    
    # Define fallback function if dashboard module is not available
    def update_stats(data):
        print(f"📊 Stats update (dashboard not available): {data}")

# Global Configuration
TASK_PORT = "5555"
RESULT_PORT = "5556"
HEARTBEAT_PORT = "5557"

context = zmq.Context()

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
        # df = pd.read_csv("processed_train.csv")
        df = pd.read_csv(os.path.join("Data", "processed_train.csv"))
        
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
                    "processing_rate": worker_info.get("processing_rate", 0)
                }
                
                current_time = time.time()
                
                # Only send updates every 0.5 seconds to avoid flooding
                if current_time - last_update_time >= 0.5:
                    # Calculate worker performance metrics
                    workers_count = len(self.workers)
                    if workers_count > 0:
                        total_cpu = sum(w.get("cpu", 0) for w in self.workers.values())
                        total_memory = sum(w.get("memory", 0) for w in self.workers.values())
                        total_tasks_processed = sum(w.get("tasks_processed", 0) for w in self.workers.values())
                        
                        # Update dashboard with stats
                        print(f"📊 Sending stats to dashboard: {workers_count} workers, {self.completed_tasks}/{self.total_tasks} tasks")
                        
                        update_stats({
                            "workers": self.worker_stats,
                            "completed_tasks": self.completed_tasks,
                            "total_tasks": self.total_tasks,
                            "workers_count": workers_count,
                            "avg_cpu": total_cpu / max(1, workers_count),
                            "avg_memory": total_memory / max(1, workers_count),
                            "total_tasks_processed": total_tasks_processed,
                            "elapsed_time": time.time() - self.start_time
                        })
                        
                        last_update_time = current_time
                
            except zmq.Again:
                pass  # No data available at the moment
            except Exception as e:
                print(f"⚠️ Error in handle_workers: {e}")
                import traceback
                traceback.print_exc()
                
            # Remove inactive workers
            current_time = time.time()
            inactive_workers = [worker for worker, info in self.workers.items() 
                            if current_time - info.get("last_seen", current_time) > 10]

            for worker_id in inactive_workers:
                print(f"➖ Worker disconnected: {worker_id}")
                del self.workers[worker_id]
                if worker_id in self.worker_stats:
                    del self.worker_stats[worker_id]
            
            # Even if no heartbeats received, periodically update dashboard
            current_time = time.time()
            if current_time - last_update_time >= 2.0:  # Send update every 2 seconds regardless
                workers_count = len(self.workers)
                if workers_count > 0:
                    total_cpu = sum(w.get("cpu", 0) for w in self.workers.values())
                    total_memory = sum(w.get("memory", 0) for w in self.workers.values())
                    total_tasks_processed = sum(w.get("tasks_processed", 0) for w in self.workers.values())
                    
                    update_stats({
                        "workers": self.worker_stats,
                        "completed_tasks": self.completed_tasks,
                        "total_tasks": self.total_tasks,
                        "workers_count": workers_count,
                        "avg_cpu": total_cpu / max(1, workers_count),
                        "avg_memory": total_memory / max(1, workers_count),
                        "total_tasks_processed": total_tasks_processed,
                        "elapsed_time": time.time() - self.start_time
                    })
                    
                    last_update_time = current_time
                    
            time.sleep(0.1)  # Avoid excessive CPU usage
        
        
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

    # def collect_results(self):
    #     """Wait for all workers to send back results."""
    #     received_task_ids = set()

    #     while len(received_task_ids) < self.total_tasks and self.running:
    #         try:
    #             result = self.result_socket.recv_json(flags=zmq.NOBLOCK)
    #             print(f"📥 Master Received: {result}")

    #             received_task_ids.add(result["task_id"])
    #             self.completed_tasks += 1
    #             progress = (self.completed_tasks / self.total_tasks) * 100
    #             print(f"📊 Progress: {self.completed_tasks}/{self.total_tasks} tasks completed ({progress:.1f}%)")
                
    #             self.save_results(result)

    #         except zmq.Again:
    #             time.sleep(0)
                
    #             # Print status update every 5 seconds
    #             if self.completed_tasks > 0 and self.completed_tasks % 10 == 0:
    #                 progress = (self.completed_tasks / self.total_tasks) * 100
    #                 print(f"📊 Progress update: {self.completed_tasks}/{self.total_tasks} tasks completed ({progress:.1f}%)")

    #     if not self.running:
    #         print("⚠️ Processing interrupted by user.")
    #     else:
    #         print(f"🎉 All {self.total_tasks} tasks completed successfully!")

    #         # ✅ If any task IDs were never received, save them as `-1`
    #         missing_task_ids = set(range(self.total_tasks)) - received_task_ids
    #         for task_id in missing_task_ids:
    #             print(f"⚠️ No result received for Task ID {task_id}, marking as -1.")
    #             self.save_results({"task_id": task_id, "passenger_id": task_id + 1, "prediction": None})

    # Update the collect_results function in master.py to include the following changes:

    def collect_results(self):
        """Wait for all workers to send back results."""
        received_task_ids = set()

        while len(received_task_ids) < self.total_tasks and self.running:
            try:
                result = self.result_socket.recv_json(flags=zmq.NOBLOCK)
                print(f"📥 Master Received: {result}")

                received_task_ids.add(result["task_id"])
                self.completed_tasks += 1
                progress = (self.completed_tasks / self.total_tasks) * 100
                print(f"📊 Progress: {self.completed_tasks}/{self.total_tasks} tasks completed ({progress:.1f}%)")
                
                # Update dashboard with latest task completion info
                workers_count = len(self.workers)
                if workers_count > 0:
                    total_cpu = sum(w.get("cpu", 0) for w in self.workers.values())
                    total_memory = sum(w.get("memory", 0) for w in self.workers.values())
                    total_tasks_processed = sum(w.get("tasks_processed", 0) for w in self.workers.values())
                else:
                    total_cpu = total_memory = total_tasks_processed = 0
                    
                update_stats({
                    "workers": self.worker_stats,
                    "completed_tasks": self.completed_tasks,
                    "total_tasks": self.total_tasks,
                    "workers_count": workers_count,
                    "avg_cpu": total_cpu / max(1, workers_count),
                    "avg_memory": total_memory / max(1, workers_count),
                    "total_tasks_processed": total_tasks_processed,
                    "elapsed_time": time.time() - self.start_time if hasattr(self, 'start_time') else 0
                })
                
                self.save_results(result)

            except zmq.Again:
                time.sleep(0)

        if not self.running:
            print("⚠️ Processing interrupted by user.")
        else:
            print(f"🎉 All {self.total_tasks} tasks completed successfully!")
            
            # Final update to dashboard
            workers_count = len(self.workers)
            if workers_count > 0:
                total_cpu = sum(w.get("cpu", 0) for w in self.workers.values())
                total_memory = sum(w.get("memory", 0) for w in self.workers.values())
                total_tasks_processed = sum(w.get("tasks_processed", 0) for w in self.workers.values())
            else:
                total_cpu = total_memory = total_tasks_processed = 0
                
            update_stats({
                "workers": self.worker_stats,
                "completed_tasks": self.total_tasks,
                "total_tasks": self.total_tasks,
                "workers_count": workers_count,
                "avg_cpu": total_cpu / max(1, workers_count),
                "avg_memory": total_memory / max(1, workers_count),
                "total_tasks_processed": total_tasks_processed,
                "elapsed_time": time.time() - self.start_time if hasattr(self, 'start_time') else 0
            })

            # Process any missing tasks
            missing_task_ids = set(range(self.total_tasks)) - received_task_ids
            for task_id in missing_task_ids:
                print(f"⚠️ No result received for Task ID {task_id}, marking as -1.")
                self.save_results({"task_id": task_id, "passenger_id": task_id + 1, "prediction": None})
                    
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
        threading.Thread(target=self.handle_workers, daemon=True).start()
        threading.Thread(target=self.monitor_keyboard, daemon=True).start()
        
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



if __name__ == "__main__":
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