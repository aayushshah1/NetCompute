import zmq
import threading
import time
import json
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ai_module import train_model, predict  # ✅ Use single AI module

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
        df = pd.read_csv("processed_train.csv")

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
        df = pd.read_csv("processed_test.csv")

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
        while True:
            try:
                worker_info = json.loads(self.heartbeat_socket.recv_string(flags=zmq.NOBLOCK))
                self.workers[worker_info['worker_id']] = worker_info
            except zmq.Again:
                pass
            time.sleep(1)

    def distribute_tasks(self):
        """Assign tasks (train or test) to available workers."""
        while self.unfinished_tasks:
            worker_id, _, request = self.task_socket.recv_multipart()

            if self.unfinished_tasks:
                task = self.unfinished_tasks.pop(0)
                self.task_socket.send_multipart([worker_id, b"", json.dumps(task).encode()])
                print(f"📤 Assigned Task ID {task['task_id']} to {worker_id}")

    def collect_results(self):
        """Wait for all workers to send back results."""
        received_task_ids = set()

        while len(received_task_ids) < self.total_tasks:
            try:
                result = self.result_socket.recv_json(flags=zmq.NOBLOCK)
                print(f"📥 Master Received: {result}")

                received_task_ids.add(result["task_id"])
                self.save_results(result)

            except zmq.Again:
                time.sleep(0.5)

        # ✅ If any task IDs were never received, save them as `-1`
        missing_task_ids = set(range(self.total_tasks)) - received_task_ids
        for task_id in missing_task_ids:
            print(f"⚠️ No result received for Task ID {task_id}, marking as -1.")
            self.save_results({"task_id": task_id, "passenger_id": task_id + 1, "prediction": None})

    def save_results(self, result):
        """Save predictions into submission file based on mode."""
        prediction = result['prediction'][0] if result['prediction'] and isinstance(result['prediction'], list) else -1

        df = pd.DataFrame({"PassengerId": [result['passenger_id']], "Survived": [prediction]})

        # ✅ Save to different files for train & test
        filename = "submission_train.csv" if self.mode == "train" else "submission_test.csv"
        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

        print(f"✅ Saved Passenger ID {result['passenger_id']} with prediction: {prediction}")

    def run(self):
        threading.Thread(target=self.handle_workers, daemon=True).start()
        self.distribute_tasks()
        self.collect_results()

if __name__ == "__main__":
    mode = "test"  # Change to "test" for inference
    MasterNode(mode).run()
