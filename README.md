# NetCompute
NetCompute aims to create a distributed computing framework that utilizes unused computational resources in a network of computers. This system will dynamically assign tasks to connected nodes, pooling their collective computational power to execute resource-intensive workloads.
------

# MVP

A simple distributed computing prototype using Python sockets to demonstrate task distribution between a server and multiple workers. The server assigns tasks from a queue, and workers execute the tasks and return results.

## How to Run

### Prerequisites
1. Ensure you have **Python 3.8+** installed on your system.
2. Install required dependencies (if any). For this basic setup, no additional libraries are required.

### Steps to Clone and Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/aayushshah1/NetCompute
   cd NetCompute

2. **Start the Server**
   ```bash
   python server.py

3. **Start a Worker**
   ```bash
   python worker.py

4. **Additional Workers**
  To connect more workers, repeat Step 3 in separate terminal windows. Each worker will be assigned tasks sequentially.

