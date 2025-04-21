import os
import zmq
import time
import random
import json
import psutil
import threading
import sys
import uuid
import traceback
from ai_module import train_model, predict

# Load configuration
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
except Exception as e:
    print(f"Error loading config: {e}")
    config = {
        "master": {
            "ip": "localhost",
            "task_port": "5555",
            "result_port": "5556",
            "heartbeat_port": "5557"
        }
    }

# Get master configuration
MASTER_IP = config["master"]["ip"]
TASK_PORT = config["master"]["task_port"]
RESULT_PORT = config["master"]["result_port"]
HEARTBEAT_PORT = config["master"]["heartbeat_port"]

context = zmq.Context()

# Generate a unique Worker ID using UUID to ensure uniqueness across machines
worker_id = f"worker-{str(uuid.uuid4())[:8]}"

# Store hostname for identification
hostname = os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', 'unknown')

class WorkerNode:
    def __init__(self):
        self.running = True
        self.tasks_processed = 0
        self.task_times = []  # Track processing time for each task
        self.total_processing_time = 0
        self.current_task = None
        self.current_batch_size = 0
        self.batch_processed = 0
        
        try:
            # Task Request (DEALER)
            self.task_socket = context.socket(zmq.DEALER)
            self.task_socket.setsockopt_string(zmq.IDENTITY, worker_id)
            task_url = f"tcp://{MASTER_IP}:{TASK_PORT}"
            self.task_socket.connect(task_url)

            # Result Sending (PUSH)
            self.result_socket = context.socket(zmq.PUSH)
            result_url = f"tcp://{MASTER_IP}:{RESULT_PORT}"
            self.result_socket.connect(result_url)

            # Heartbeat (PUB)
            self.heartbeat_socket = context.socket(zmq.PUB)
            heartbeat_url = f"tcp://{MASTER_IP}:{HEARTBEAT_PORT}"
            self.heartbeat_socket.connect(heartbeat_url)
            
            print(f"🚀 Worker {worker_id} started on {hostname}")
            print("Press 'q' and Enter to quit gracefully")
        except Exception as e:
            print(f"Error initializing worker: {e}")
            raise

    def send_heartbeat(self):
        """Send minimal heartbeat with system resource stats."""
        start_time = time.time()
        
        while self.running:
            try:
                # Gather system resource stats (minimal computation)
                cpu_usage = psutil.cpu_percent(interval=None)
                memory_usage = psutil.virtual_memory().percent
                
                # Calculate processing rate
                elapsed_time = time.time() - start_time
                processing_rate = self.tasks_processed / max(1, elapsed_time)
                
                # Calculate avg task processing time (in milliseconds)
                avg_task_time = 0
                if self.tasks_processed > 0:
                    avg_task_time = (self.total_processing_time / self.tasks_processed) * 1000
                
                # Send heartbeat with minimal information
                heartbeat_message = {
                    "worker_id": worker_id,
                    "hostname": hostname,
                    "cpu": cpu_usage,
                    "memory": memory_usage,
                    "tasks_processed": self.tasks_processed,
                    "last_seen": time.time(),
                    "current_task": self.current_task,
                    "current_batch_size": self.current_batch_size,
                    "batch_processed": self.batch_processed,
                    "processing_rate": processing_rate,
                    "avg_task_time_ms": avg_task_time
                }
                
                self.heartbeat_socket.send_string(json.dumps(heartbeat_message))
                time.sleep(2)  # Send heartbeat every 2 seconds to reduce overhead
            except Exception:
                time.sleep(2)

    def monitor_keyboard(self):
        """Listen for keyboard input to shut down worker gracefully."""
        try:
            while self.running:
                cmd = input()
                if cmd.lower() == 'q':
                    print("🛑 Initiating graceful shutdown...")
                    self.running = False
                    self.cleanup()
                    break
        except Exception:
            pass

    def process_tasks(self):
        """Main loop to process tasks from the master."""
        while self.running:
            try:
                # Request a batch of tasks
                self.task_socket.send_multipart([b"", b"request_task"])
                
                # Wait for task with timeout
                poller = zmq.Poller()
                poller.register(self.task_socket, zmq.POLLIN)
                
                socks = dict(poller.poll(2000))  # 2 second timeout
                if self.task_socket in socks and socks[self.task_socket] == zmq.POLLIN:
                    parts = self.task_socket.recv_multipart()
                    if len(parts) != 2:
                        time.sleep(0.1)
                        continue
                        
                    _, batch_data = parts
                    
                    try:
                        batch_info = json.loads(batch_data.decode())
                        task_batch = batch_info.get("batch", [])
                        
                        if not task_batch:
                            time.sleep(0.5)  # No tasks, small wait
                            continue
                            
                        # Process each task in the batch
                        self.current_batch_size = len(task_batch)
                        self.batch_processed = 0
                        
                        for task in task_batch:
                            if not self.running:
                                break
                                
                            task_id = task.get("task_id", "unknown")
                            self.current_task = task_id
                            self.batch_processed += 1
                            
                            # Process the task
                            start_time = time.time()
                            
                            # Determine if training or testing
                            if "label" in task:
                                result = train_model(task)  # Training mode
                            else:
                                result = predict(task)  # Prediction mode
                            
                            # Calculate processing time
                            processing_time = time.time() - start_time
                            
                            # Track task times for analytics
                            self.task_times.append(processing_time)
                            self.total_processing_time += processing_time
                            
                            # Keep only the last 100 task times to limit memory usage
                            if len(self.task_times) > 100:
                                self.task_times.pop(0)
                            
                            # Add processing stats and worker identification to result
                            result['processing_time'] = processing_time
                            result['worker_id'] = worker_id
                            result['hostname'] = hostname
                            result['task_id'] = task_id
                            
                            # Send result back to master
                            self.result_socket.send_json(result)
                            
                            # Update statistics
                            self.tasks_processed += 1
                            
                        # Clear current task when batch is done
                        self.current_task = None
                        self.current_batch_size = 0
                        self.batch_processed = 0
                    except json.JSONDecodeError:
                        time.sleep(0.1)
                else:
                    # No response, retry after delay
                    time.sleep(0.5)
                        
            except Exception as e:
                print(f"Error processing tasks: {e}")
                time.sleep(1)  # Wait before retrying
    
    def run(self):
        """Start all worker threads and manage shutdown."""
        heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        heartbeat_thread.start()

        keyboard_thread = threading.Thread(target=self.monitor_keyboard, daemon=True)
        keyboard_thread.start()

        try:
            self.process_tasks()
        except KeyboardInterrupt:
            print(f"\n🛑 [Worker {worker_id}] Interrupted by user")
            self.running = False
        except Exception as e:
            print(f"⚠️ Critical error: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources before shutting down."""
        print(f"👋 [Worker {worker_id}] Cleaning up...")

        try:
            self.task_socket.close()
            self.result_socket.close() 
            self.heartbeat_socket.close()
            context.term()
            print(f"✅ [Worker {worker_id}] Shutdown complete.")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

        os._exit(0)  # Force exit


if __name__ == "__main__":
    try:
        worker = WorkerNode()
        worker.run()
    except Exception as e:
        print(f"⚠️ Critical error: {e}")
        context.term()
        sys.exit(1)