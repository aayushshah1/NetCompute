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
✅ Workers connecting to the Master via **WebSockets** and executing computations.  
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
**Python 3.8+** installed on your system.

### Steps to Clone and Run
1. **Clone the Repository**

   ```bash
   git clone https://github.com/aayushshah1/NetCompute
   cd NetCompute
   code .  # To open folder in code editor
   
   git checkout branch-name  # (optional) Switch to the your branch
   pip install -r requirements.txt  # In terminal
   ```

2. **Set Up Master Node**
   * Find your system's IP address (run this on the master machine):

   ```bash
   ipconfig getifaddr en0  # macOS/Linux
   ipconfig  # Windows (look for IPv4 Address)
   ```

   * Update `MASTER_IP` in `worker.py`:

   ```python
   MASTER_IP = "YOUR_MASTER_IP"
   ```

3. **Start the Master Node**

   ```bash
   python master.py  # For training enter 1 / for test enter 2
   ```

4. **Start a Worker Node**
   * Run the following on a separate machine or terminal:

   ```bash
   python worker.py
   ```

5. **Start Multiple Workers**
   * To utilize more computing power, start additional workers:

   ```bash
   python worker.py
   ```

## Training and Prediction

### Training the Model
* Start the master node (`master.py`) to train the model on distributed worker nodes.
* Ensure the model is trained before running test mode.

### Running Predictions on Test Data
* Run the test master node:

   ```bash
   python master.py
   ```

* Workers will process the test dataset and generate predictions.
* Results will be saved in `submission_test.csv`.

### Resetting for Re-Training
Before retraining, delete:

```bash
rm Model/trained_model.pkl
rm Data/submission_test.csv
```

Then, restart the training process.

## Monitoring and Debugging
* **Master Node Logs**:
   * Assigned tasks and received results are logged in `master.py`.
* **Worker Node Logs**:
   * Each worker prints task completion status and any errors.
* **Common Issues**:
   * If a worker is not receiving tasks, check if `MASTER_IP` is correctly set.
   * If predictions are all `-1`, ensure the model was trained successfully before running `master_test.py`.

## Contributions
Fork the repository, create a new branch, and submit a pull request.

## License
This project is licensed under the MIT License.
