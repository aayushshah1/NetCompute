import zmq
import time
import random
import json

# Master Node Configuration
MASTER_IP = "192.168.29.201"  # Change to the actual master IP
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
    """Send heartbeat to the master every 2 seconds"""
    while True:
        heartbeat_socket.send_string(worker_id)
        time.sleep(2)

# Start heartbeat thread
import threading
threading.Thread(target=send_heartbeat, daemon=True).start()

while True:
    # Request a task from the master
    task_socket.send_multipart([b"", b"request_task"])
    
    # Receive the task
    try:
        _, task_data = task_socket.recv_multipart()
        task = json.loads(task_data.decode())
    except Exception as e:
        print(f"[Worker {worker_id}] Error receiving task: {e}")
        continue

    # Extract task details
    operation = task.get("operation")
    num1 = task.get("num1")
    num2 = task.get("num2")
    print(f"[Worker {worker_id}] Received Task: {num1} {operation} {num2}")

    # Perform computation
    if operation == "+":
        result = num1 + num2
    elif operation == "-":
        result = num1 - num2
    elif operation == "*":
        result = num1 * num2
    elif operation == "/":
        result = num1 / num2 if num2 != 0 else "Error: Division by Zero"
    else:
        result = "Error: Invalid Operation"

    time.sleep(random.uniform(0.5, 2))  # Simulate processing delay

    # Send result back
    result_data = {"worker_id": worker_id, "task": task, "result": result}
    result_socket.send_json(result_data)
    print(f"[Worker {worker_id}] Task Completed: {num1} {operation} {num2} = {result}")