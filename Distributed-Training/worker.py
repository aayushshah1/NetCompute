import os
import zmq
import time
import random
import json
import psutil
import threading
import sys
import uuid
from ai_module import train_model, predict

# Load configuration
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    print("✅ Loaded configuration from config.json")
except Exception as e:
    print(f"⚠️ Error loading config.json: {e}")
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
hostname = os.uname().nodename

class WorkerNode:
    def __init__(self):
        self.running = True
        self.tasks_processed = 0
        self.task_times = []  # Track processing time for each task
        self.total_processing_time = 0
        self.current_task = None
        self.current_batch_size = 0
        self.batch_processed = 0
        
        # Task Request (DEALER)
        self.task_socket = context.socket(zmq.DEALER)
        self.task_socket.setsockopt_string(zmq.IDENTITY, worker_id)
        self.task_socket.connect(f"tcp://{MASTER_IP}:{TASK_PORT}")

        # Result Sending (PUSH)
        self.result_socket = context.socket(zmq.PUSH)
        self.result_socket.connect(f"tcp://{MASTER_IP}:{RESULT_PORT}")

        # Heartbeat (PUB)
        self.heartbeat_socket = context.socket(zmq.PUB)
        self.heartbeat_socket.connect(f"tcp://{MASTER_IP}:{HEARTBEAT_PORT}")
        
        print(f"🚀 Worker {worker_id} started on {hostname} and ready to receive tasks.")
        print("📝 Press 'q' and Enter at any time to quit gracefully")

    def send_heartbeat(self):
        """Send heartbeat with system resource stats."""
        start_time = time.time()
        
        while self.running:
            # Gather system resource stats
            cpu_usage = psutil.cpu_percent(interval=0.5)  # Use shorter interval
            memory_usage = psutil.virtual_memory().percent
            
            # Calculate processing rate and uptime
            elapsed_time = time.time() - start_time
            processing_rate = self.tasks_processed / max(1, elapsed_time)  # tasks per second
            
            # Calculate avg task processing time (in milliseconds)
            avg_task_time = 0
            if self.tasks_processed > 0:
                avg_task_time = (self.total_processing_time / self.tasks_processed) * 1000
            
            # Get the last 10 task times
            recent_task_times = self.task_times[-10:] if self.task_times else []
            
            # Send heartbeat message
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
                "uptime": elapsed_time,
                "avg_task_time_ms": avg_task_time,
                "recent_task_times": [t * 1000 for t in recent_task_times],  # Convert to milliseconds
                "system_info": {
                    "hostname": hostname,
                    "cpu_count": psutil.cpu_count(),
                    "mem_total": psutil.virtual_memory().total / (1024 * 1024),  # MB
                }
            }
            
            self.heartbeat_socket.send_string(json.dumps(heartbeat_message))
            time.sleep(1)  # Send heartbeat every second

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
        except Exception as e:
            print(f"⚠️ Critical error in monitor_keyboard: {e}")

    def process_tasks(self):
        """Main loop to process tasks from the master."""
        while self.running:
            # Request a task
            self.task_socket.send_multipart([b"", b"request_task"])
            
            try:
                # Wait for task with timeout
                poller = zmq.Poller()
                poller.register(self.task_socket, zmq.POLLIN)
                
                socks = dict(poller.poll(1000))  # 1 second timeout
                if self.task_socket in socks and socks[self.task_socket] == zmq.POLLIN:
                    _, batch_data = self.task_socket.recv_multipart()
                    batch_info = json.loads(batch_data.decode())
                    task_batch = batch_info.get("batch", [])
                    
                    if not task_batch:
                        time.sleep(0.1)  # No tasks, small wait
                        continue
                        
                    # Process the batch of tasks
                    self.current_batch_size = len(task_batch)
                    self.batch_processed = 0
                    
                    print(f"📥 [Worker {worker_id}] Received batch of {len(task_batch)} tasks")
                    
                    # Process each task in the batch
                    for task in task_batch:
                        if not self.running:
                            break
                            
                        self.current_task = task.get("task_id", "unknown")
                        self.batch_processed += 1
                        
                        print(f"📥 [Worker {worker_id}] Processing Task ID {task['task_id']} ({self.batch_processed}/{self.current_batch_size})")
                        
                        # Process the task
                        start_time = time.time()
                        
                        # Check whether this is training or testing
                        if "label" in task:
                            result = train_model(task)  # Training mode
                        else:
                            result = predict(task)  # Prediction mode
                        
                        # Calculate processing time
                        processing_time = time.time() - start_time
                        
                        # Track task times for analytics
                        self.task_times.append(processing_time)
                        self.total_processing_time += processing_time
                        
                        # Keep only the last 100 task times
                        if len(self.task_times) > 100:
                            self.task_times.pop(0)
                        
                        # Add processing stats and worker identification to result
                        result['processing_time'] = processing_time
                        result['worker_id'] = worker_id
                        result['hostname'] = hostname
                        result['task_id'] = task.get('task_id')
                        
                        # Send result back to master
                        self.result_socket.send_json(result)
                        
                        # Update statistics
                        self.tasks_processed += 1
                        prediction_str = str(result.get('prediction', None)) if result.get('prediction') else "None"
                        print(f"📤 [Worker {worker_id}] Completed Task ID {task['task_id']} in {processing_time:.3f}s - Result: {prediction_str}")
                        print(f"📊 Total tasks processed: {self.tasks_processed}, Avg time: {(self.total_processing_time/self.tasks_processed)*1000:.2f}ms")
                    
                    # Clear current task when batch is done
                    self.current_task = None
                    self.current_batch_size = 0
                    self.batch_processed = 0
                        
            except Exception as e:
                print(f"⚠️ [Worker {worker_id}] Error: {e}")
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
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources before shutting down."""
        print(f"👋 [Worker {worker_id}] Cleaning up resources...")
        time.sleep(1)

        try:
            # Close sockets before terminating context
            self.task_socket.close()
            self.result_socket.close()
            self.heartbeat_socket.close()

            context.term()  # Properly terminate ZeroMQ context
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

        print(f"✅ [Worker {worker_id}] Successfully shut down.")
        os._exit(0)  # Force exit to prevent lingering issues


if __name__ == "__main__":
    try:
        worker = WorkerNode()
        worker.run()
    except Exception as e:
        print(f"⚠️ Critical error: {e}")
        context.term()
        sys.exit(1)