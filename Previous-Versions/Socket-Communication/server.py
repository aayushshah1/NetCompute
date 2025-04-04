import socket
import threading
import queue
import pickle

# Task queue to manage tasks
task_queue = queue.Queue()
results = {}
server_running = True  # Global flag to control the server state

def handle_worker(client_socket, address):
    """Handles communication with a worker."""
    try:
        while server_running:  # Keep running until server is stopped
            # Send task to worker
            if not task_queue.empty():
                task = task_queue.get()
                client_socket.sendall(pickle.dumps(task))
                print(f"Assigned task {task['task_id']} to {address}")

                # Receive result
                result = pickle.loads(client_socket.recv(1024))
                results[task["task_id"]] = result
                print(f"Received result from {address}: {result}")

                # Ask the worker if they are ready for another task
                client_socket.sendall(pickle.dumps({"status": "ready"}))
                ready_response = pickle.loads(client_socket.recv(1024))
                
                if ready_response.get("ready") == "no":
                    print(f"Worker {address} disconnected after completing tasks.")
                    break

        # Notify worker to stop when the server stops
        client_socket.sendall(pickle.dumps({"status": "stop"}))
    except (ConnectionResetError, EOFError):
        print(f"Worker {address} disconnected.")
    finally:
        client_socket.close()

def server(host="0.0.0.0", port=5001):
    """Server to distribute tasks and collect results."""
    global server_running  # Use the global variable

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)

    print("Server is running and waiting for workers to connect...")

    def stop_server_input():
        """Listens for the 'stop server' input from the admin."""
        global server_running  # Modify the global variable
        while True:
            command = input("Enter 'stop server' to shut down the server: ").strip().lower()
            if command == "stop server":
                print("Shutting down the server...")
                server_running = False
                server_socket.close()
                break

    # Start the stop server input listener
    threading.Thread(target=stop_server_input, daemon=True).start()

    while server_running:
        try:
            client_socket, address = server_socket.accept()
            print(f"Worker connected: {address}")
            threading.Thread(target=handle_worker, args=(client_socket, address)).start()
        except OSError:  # This happens when the server socket is closed
            break

    print("Server has been stopped.")

# Example tasks to distribute
def create_tasks():
    task_id = 0
    while server_running:  # Continuously generate tasks while the server is running
        for i in range(10):  # Generate 10 Fibonacci tasks at a time
            task_queue.put({"task_id": task_id, "function": "fibonacci", "args": (i + 20,)})
            task_id += 1

if __name__ == "__main__":
    threading.Thread(target=create_tasks, daemon=True).start()
    server()