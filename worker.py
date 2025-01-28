import socket
import pickle

def fibonacci(n):
    """Simple Fibonacci function."""
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def process_task(task):
    """Processes the given task."""
    func = task["function"]
    args = task["args"]
    if func == "fibonacci":
        return fibonacci(*args)

def worker(server_host="127.0.0.1", server_port=5001):
    """Connects to the server and executes tasks."""
    worker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    worker_socket.connect((server_host, server_port))

    print("Connected to the server, waiting for tasks...")
    
    while True:
        try:
            # Receive task from server
            task = pickle.loads(worker_socket.recv(1024))
            
            # If server sends a stop signal, exit
            if task.get("status") == "stop":
                print("Server has stopped. Worker shutting down...")
                break

            print(f"Received task: {task}")

            # Process task and send result
            result = process_task(task)
            worker_socket.sendall(pickle.dumps({"task_id": task["task_id"], "result": result}))

            # Check if the worker wants the next task
            status = pickle.loads(worker_socket.recv(1024))
            if status.get("status") == "ready":
                ready = input("Ready for next task? - Yes or No: ").strip().lower()
                if ready == "yes":
                    worker_socket.sendall(pickle.dumps({"ready": "yes"}))
                else:
                    worker_socket.sendall(pickle.dumps({"ready": "no"}))
                    print("Disconnecting from server...")
                    break

        except (EOFError, ConnectionResetError):
            print("Server disconnected.")
            break

    worker_socket.close()

if __name__ == "__main__":
    worker()