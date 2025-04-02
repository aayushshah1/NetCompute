from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import time
import threading
import json

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'distributed-training-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Store for statistics
stats = {
    "workers": {},
    "completed_tasks": 0,
    "total_tasks": 0,
    "elapsed_time": 0,
    "start_time": time.time(),
    "workers_count": 0,
    "avg_cpu": 0,
    "avg_memory": 0,
    "total_tasks_processed": 0
}

# Track single vs multiple worker performance
performance_metrics = {
    "single_worker": {
        "processing_time": None,
        "tasks_completed": 0
    },
    "multi_worker": {
        "processing_time": None,
        "tasks_completed": 0,
        "worker_count": 0
    }
}

@app.route('/')
def index():
    """Serve the dashboard HTML page."""
    return render_template('dashboard.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"🔌 Client connected to dashboard")
    # Send current stats to newly connected client
    emit('update_stats', stats)

def update_stats(new_stats):
    """Update the stats store with new data."""
    global stats, performance_metrics
    
    # Update worker counts
    workers_count = len(new_stats.get("workers", {}))
    if workers_count > 0:
        # Track performance metrics for comparison
        if workers_count == 1 and performance_metrics["single_worker"]["processing_time"] is None:
            if new_stats.get("completed_tasks", 0) >= new_stats.get("total_tasks", 0) > 0:
                # Single worker completed all tasks
                performance_metrics["single_worker"]["processing_time"] = new_stats.get("elapsed_time", 0)
                performance_metrics["single_worker"]["tasks_completed"] = new_stats.get("completed_tasks", 0)
                print(f"📊 Single worker performance recorded: {performance_metrics['single_worker']}")
        elif workers_count > 1 and performance_metrics["multi_worker"]["processing_time"] is None:
            if new_stats.get("completed_tasks", 0) >= new_stats.get("total_tasks", 0) > 0:
                # Multiple workers completed all tasks
                performance_metrics["multi_worker"]["processing_time"] = new_stats.get("elapsed_time", 0)
                performance_metrics["multi_worker"]["tasks_completed"] = new_stats.get("completed_tasks", 0)
                performance_metrics["multi_worker"]["worker_count"] = workers_count
                print(f"📊 Multi-worker performance recorded: {performance_metrics['multi_worker']}")
    
    # Update stats
    stats.update(new_stats)
    
    # Add performance comparison data
    stats["performance_comparison"] = performance_metrics
    
    try:
        # Broadcast updated stats to all clients
        socketio.emit('update_stats', stats)
    except Exception as e:
        print(f"Error emitting stats: {e}")

def start_dashboard(port=5050):
    """Start the Flask-SocketIO server."""
    print(f"🌐 Starting dashboard at http://localhost:{port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    # Start in main thread for testing
    start_dashboard()