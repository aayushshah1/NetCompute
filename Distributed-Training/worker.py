import os
import zmq
import time
import random
import json
import psutil
import threading
import sys
from ai_module import train_model, predict  # ✅ Use single ai_module.py


# Master Node Configuration
MASTER_IP = "192.168.29.117"  # Change this if running on multiple machines
TASK_PORT = "5555"
RESULT_PORT = "5556"
HEARTBEAT_PORT = "5557"

context = zmq.Context()

# Generate a unique Worker ID
worker_id = f"worker-{random.randint(1000, 9999)}"

class WorkerNode:
    def __init__(self):
        self.running = True
        self.tasks_processed = 0
        
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
        
        print(f"🚀 Worker {worker_id} started and ready to receive tasks.")
        print("📝 Press 'q' and Enter at any time to quit gracefully")

    # def send_heartbeat(self):
    #     """Send heartbeat with system resource stats."""
    #     while self.running:
    #         usage = {
    #             "worker_id": worker_id,
    #             "cpu": psutil.cpu_percent(),
    #             "memory": psutil.virtual_memory().percent,
    #             "last_seen": time.time(),
    #             "tasks_processed": self.tasks_processed
    #         }
    #         self.heartbeat_socket.send_string(json.dumps(usage))
    #         # time.sleep(0.01)
    
   
    def send_heartbeat(self):
        """Send heartbeat with system resource stats."""
        start_time = time.time()
        
        while self.running:
            # Gather system resource stats
            cpu_usage = psutil.cpu_percent(interval=0.5)  # Use shorter interval
            memory_usage = psutil.virtual_memory().percent
            
            # Calculate processing rate
            elapsed_time = time.time() - start_time
            processing_rate = self.tasks_processed / max(1, elapsed_time)  # tasks per second
            
            # Send heartbeat message
            heartbeat_message = {
                "worker_id": worker_id,
                "cpu": cpu_usage,
                "memory": memory_usage,
                "tasks_processed": self.tasks_processed,
                "last_seen": time.time(),
                "current_task": self.current_task if hasattr(self, 'current_task') else None,
                "processing_rate": processing_rate,
                "uptime": elapsed_time
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
                    _, task_data = self.task_socket.recv_multipart()
                    task = json.loads(task_data.decode())
                    
                    print(f"📥 [Worker {worker_id}] Processing Task ID {task['task_id']}")
                    
                    # Process the task
                    start_time = time.time()
                    
                    # ✅ Check whether this is training or testing
                    if "label" in task:
                        result = train_model(task)  # ✅ Training mode
                    else:
                        result = predict(task)  # ✅ Prediction mode
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # ✅ Send result back to master
                    self.result_socket.send_json(result)
                    
                    # Update statistics
                    self.tasks_processed += 1
                    prediction_str = str(result['prediction']) if result['prediction'] else "None"
                    print(f"📤 [Worker {worker_id}] Completed Task ID {task['task_id']} in {processing_time:.3f}s - Result: {prediction_str}")
                    print(f"📊 Total tasks processed: {self.tasks_processed}")
                    
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
            # ✅ Close sockets before terminating context
            self.task_socket.close()
            self.result_socket.close()
            self.heartbeat_socket.close()

            context.term()  # ✅ Properly terminate ZeroMQ context
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

        print(f"✅ [Worker {worker_id}] Successfully shut down.")
        os._exit(0)  # ✅ Force exit to prevent lingering issues



if __name__ == "__main__":
    try:
        worker = WorkerNode()
        worker.run()
    except Exception as e:
        print(f"⚠️ Critical error: {e}")
        context.term()
        sys.exit(1)