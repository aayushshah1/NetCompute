import zmq
import time
import random
import json
import psutil  # Import psutil for system monitoring
import threading

# Master Node Configuration
MASTER_IP = "192.168.29.201"  # Change to actual master IP
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
    """Send heartbeat along with resource usage to the master every 2 seconds."""
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory().percent
        net_io = psutil.net_io_counters()
        network_sent = net_io.bytes_sent
        network_received = net_io.bytes_recv

        # Construct resource data
        resource_data = {
            "worker_id": worker_id,
            "cpu": cpu_usage,
            "memory": memory_info,
            "network_sent": network_sent,
            "network_received": network_received
        }

        heartbeat_socket.send_json(resource_data)  # Send as JSON
        time.sleep(2)

# Start heartbeat thread
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

    # Send result back with resource usage
    result_data = {
        "worker_id": worker_id,
        "task": task,
        "result": result,
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent
    }
    result_socket.send_json(result_data)
    print(f"[Worker {worker_id}] Task Completed: {num1} {operation} {num2} = {result}")
