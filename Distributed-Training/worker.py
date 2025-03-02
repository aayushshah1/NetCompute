import zmq
import time
import random
import json
import psutil
import threading
from ai_module import train_model, predict  # ✅ Use single ai_module.py

# Master Node Configuration
MASTER_IP = "192.168.29.14"  # Change this if running on multiple machines
TASK_PORT = "5555"
RESULT_PORT = "5556"
HEARTBEAT_PORT = "5557"

context = zmq.Context()

# Task Request (DEALER)
task_socket = context.socket(zmq.DEALER)
task_socket.connect(f"tcp://{MASTER_IP}:{TASK_PORT}")

# Result Sending (PUSH)
result_socket = context.socket(zmq.PUSH)
result_socket.connect(f"tcp://{MASTER_IP}:{RESULT_PORT}")

# Heartbeat (PUB)
heartbeat_socket = context.socket(zmq.PUB)
heartbeat_socket.connect(f"tcp://{MASTER_IP}:{HEARTBEAT_PORT}")

# Unique Worker ID
worker_id = f"worker-{random.randint(1000, 9999)}"
print(f"[Worker {worker_id}] Started and ready to receive tasks.")

def send_heartbeat():
    """Send heartbeat with system resource stats."""
    while True:
        usage = {
            "worker_id": worker_id,
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
        }
        heartbeat_socket.send_string(json.dumps(usage))
        time.sleep(2)

# Start heartbeat thread
threading.Thread(target=send_heartbeat, daemon=True).start()

while True:
    task_socket.send_multipart([b"", b"request_task"])  # ✅ Request task
    
    try:
        _, task_data = task_socket.recv_multipart()
        task = json.loads(task_data.decode())
    except Exception as e:
        print(f"[Worker {worker_id}] Error receiving task: {e}")
        continue

    print(f"[Worker {worker_id}] Processing Task ID {task['task_id']}")

    # ✅ Check whether this is training or testing
    if "label" in task:
        result = train_model(task)  # ✅ Training mode
    else:
        result = predict(task)  # ✅ Prediction mode

    # ✅ Send result back to master
    result_socket.send_json(result)
    print(f"[Worker {worker_id}] Sent result for Task ID {task['task_id']}: {result}")
