# NetCompute - A Distributed Computing Framework

## **1️⃣ Problem Statement**
Modern AI and data workloads require vast computing power, but most organizations have **idle PCs**. Cloud computing is **expensive**, and setting up distributed systems is **complex**. **NetCompute** enables organizations to use existing machines for distributed computing.

### **Key Challenges:**
- **Idle PCs:** Wasted potential in offices, labs, and research centers.
- **Cloud Costs:** Expensive and scales poorly for long-term AI workloads.
- **Complexity:** Traditional distributed systems require admin access & setup.

## **2️⃣ Solution - NetCompute**
NetCompute is a **lightweight, scalable distributed computing system** that connects multiple PCs to share workloads. It works by:
✅ Running a **Master Server** that distributes tasks and collects results.  
✅ Workers connecting to the Master via **ZeroMQ** and executing computations.  
✅ Supporting **multiple workers dynamically**, ensuring efficient load balancing.  
✅ Machine Learning Model Training & Prediction to distribute AI workloads efficiently.

## **3️⃣ Implementation Perspective**
Inspired by **BIONIC (UC Berkeley)** cause, NetCompute applies distributed computing to everyday hardware. It can:
🔹 Distribute ML workloads in AI labs.  
🔹 Accelerate simulations in research.  
🔹 Provide a scalable alternative to cloud-based computing.  

---

## **🚀 Getting Started**
### Prerequisites
- **Python 3.8+** installed on your system
- Required libraries:
  - pyzmq (ZeroMQ for distributed communication)
  - psutil (system monitoring)
  - pandas, numpy, scikit-learn (data analysis & ML)
  - flask, flask-socketio (dashboard interface)
  - requests (API communication)

### Steps to Clone and Run
1. **Clone the Repository**

   ```bash
   git clone https://github.com/aayushshah1/NetCompute
   cd NetCompute
   code .  # To open folder in code editor
   
   git checkout branch-name  # (optional) Switch to your branch
   pip install -r requirements.txt  # Install dependencies
   ```

2. **Set Up Configuration**
   * Find your system's IP address (run this on the master machine):

   ```bash
   ipconfig getifaddr en0  # macOS/Linux
   ipconfig  # Windows (look for IPv4 Address)
   ```

   * Update the `config.json` file with your master machine's IP address:

   ```json
   {
     "master": {
       "ip": "YOUR_MASTER_IP",
       "task_port": "5555",
       "result_port": "5556",
       "heartbeat_port": "5557"
     },
     "dashboard": {
       "host": "YOUR_MASTER_IP",
       "port": "5051"
     }
   }
   ```

3. **Start the Dashboard Server**

   ```bash
   cd Distributed-Training
   ```

4. **Start the Master Node**

   ```bash
   python master.py
   ```
   
   * Select mode when prompted:
     * Enter `1` or press Enter for Training mode
     * Enter `2` for Testing/Prediction mode

5. **Start Worker Nodes**
   * Copy the `config.json` file to each worker machine (or use the same machine)
   * Run on each worker:

   ```bash
   python worker.py
   ```

6. **Monitor Progress**
   * Open a web browser and navigate to: `http://YOUR_MASTER_IP:5051`
   * The dashboard will show real-time progress, worker stats, and performance metrics

## **System Architecture**

NetCompute uses a distributed architecture with:

1. **ZeroMQ Communication Patterns**:
   * ROUTER/DEALER pattern for task distribution
   * PUSH/PULL pattern for result collection
   * PUB/SUB pattern for heartbeats and system stats

2. **Components**:
   * **Master Node**: Central coordinator that distributes tasks, collects results, and monitors the system
   * **Worker Nodes**: Process tasks and send results back to the master
   * **Dashboard**: Web interface for monitoring system performance and progress

## Training and Prediction

### Training the Model
* Start the master node in training mode (`python master.py` and select option 1)
* Workers will automatically process training tasks
* Results will be collected and stored in `Data/submission_train.csv`

### Running Predictions on Test Data
* Start the master node in testing mode (`python master.py` and select option 2)
* Workers will process the test dataset and generate predictions
* Results will be saved in `Data/submission_test.csv`

### Resetting for Re-Training
Before retraining, delete:

```bash
rm Data/submission_train.csv
rm Data/submission_test.csv
```

Then, restart the training process.

## Monitoring and Debugging
* **Master Node Logs**:
   * Monitor task distribution and results collection
   * Check for worker connection events
* **Worker Node Logs**:
   * View task processing status and any errors
* **Dashboard**:
   * Real-time visualization of system performance
   * Worker stats and task completion progress
* **Common Issues**:
   * If workers aren't connecting, check that the IP in `config.json` is correct
   * Ensure all required ports are open (5555, 5556, 5557, 5051)
   * Verify that all dependencies are installed on both master and worker machines

## Contributions
Fork the repository, create a new branch, and submit a pull request.

## License
This project is licensed under the MIT License.
