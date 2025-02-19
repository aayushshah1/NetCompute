import zmq
import threading
import random
import time
import json

# Global Configuration
TASK_PORT = "5555"
RESULT_PORT = "5556"
HEARTBEAT_PORT = "5557"

context = zmq.Context()

class MasterNode:
    def __init__(self):
        self.tasks = [{"num1": random.randint(1, 100), "num2": random.randint(1, 100), "operation": random.choice(["+", "-", "*", "/"])} for _ in range(20)]
        self.workers = {}
        self.unfinished_tasks = self.tasks.copy()  # Tasks that still need processing
        
        # Task Distributor (ROUTER)
        self.task_socket = context.socket(zmq.ROUTER)
        self.task_socket.bind(f"tcp://*:{TASK_PORT}")
        
        # Result Collector (PULL)
        self.result_socket = context.socket(zmq.PULL)
        self.result_socket.bind(f"tcp://*:{RESULT_PORT}")
        
        # Heartbeat Checker (SUB)
        self.heartbeat_socket = context.socket(zmq.SUB)
        self.heartbeat_socket.bind(f"tcp://*:{HEARTBEAT_PORT}")
        self.heartbeat_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all heartbeats

        print("Master: Server started... Waiting for workers.")

    def handle_workers(self):
        """Track available workers via heartbeats"""
        while True:
            try:
                worker_ip = self.heartbeat_socket.recv_string(flags=zmq.NOBLOCK)
                self.workers[worker_ip] = time.time()  # Update last heartbeat timestamp
            except zmq.Again:
                pass  # No heartbeat received
            time.sleep(1)

    def distribute_tasks(self):
        """Assign tasks dynamically to workers when they request it"""
        while self.unfinished_tasks:
            worker_id, _, request = self.task_socket.recv_multipart()
            if self.unfinished_tasks:
                task = self.unfinished_tasks.pop(0)
                self.task_socket.send_multipart([worker_id, b"", json.dumps(task).encode()])
                print(f"Master: Assigned task {task} to worker {repr(worker_id)}")
            else:
                print("Master: No tasks left.")

    def collect_results(self):
        """Receive results from workers"""
        completed = 0
        while completed < len(self.tasks):
            result = self.result_socket.recv_json()
            print(f"Master: Received result -> {result}")
            completed += 1
        print("Master: All tasks completed!")

    def check_worker_status(self):
        """Reassign tasks if a worker fails (checks heartbeats)"""
        while True:
            current_time = time.time()
            dead_workers = [worker for worker, last_seen in self.workers.items() if current_time - last_seen > 5]
            
            for worker in dead_workers:
                print(f"Master: Worker {worker} is unresponsive. Reassigning tasks...")
                del self.workers[worker]  # Remove from active list
                
                # Put unfinished tasks back to queue (this assumes each worker got only 1 task at a time)
                self.unfinished_tasks.append({"num1": random.randint(1, 100), "num2": random.randint(1, 100), "operation": random.choice(["+", "-", "*", "/"])})

            time.sleep(3)

    def run(self):
        """Start server threads"""
        threading.Thread(target=self.handle_workers, daemon=True).start()
        threading.Thread(target=self.check_worker_status, daemon=True).start()
        
        self.distribute_tasks()
        self.collect_results()

if __name__ == "__main__":
    MasterNode().run()