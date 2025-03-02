# NetCompute - Distributed Computing Framework

NetCompute is a distributed computing framework that dynamically assigns tasks to connected nodes, utilizing their computational power to execute resource-intensive workloads efficiently.

## Features
- Distributed Task Execution
- Dynamic Worker Assignment
- Machine Learning Model Training & Prediction
- Supports Multiple Workers Across Different Machines

---

## Installation and Setup

### Prerequisites
1. **Python 3.8+** installed on your system.
2. **Required Dependencies**:
   - Install the dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

### Steps to Clone and Run
1. **Clone the Repository**

   ```bash
   git clone https://github.com/aayushshah1/NetCompute
   cd NetCompute
   git checkout Distributed-Training  # Switch to the correct branch
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
   python master.py  # For training
   python master_test.py  # For test mode (predictions)
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
   python master_test.py
   ```

* Workers will process the test dataset and generate predictions.
* Results will be saved in `submission_test.csv`.

### Resetting for Re-Training
Before retraining, delete:

```bash
rm trained_model.pkl
rm data/submission_test.csv
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
Feel free to contribute to NetCompute! Fork the repository, create a new branch, and submit a pull request.

## License
This project is licensed under the MIT License.