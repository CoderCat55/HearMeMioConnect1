#!/usr/bin/env python3
"""
Raspberry Pi 5 Web Server - Gesture Recognition API
Provides gesture recognition endpoints for mobile app access
comminicates with json
"""

from flask import Flask, jsonify, request
import numpy as np
import glob

app = Flask(__name__)

# Configuration
SSID = "hearme"
PASSWORD = "123456789"
PORT = 5002

# Global system reference (will be injected from main.py)
_system = None

def inject_system(system):
    """Inject the GestureSystem instance from main.py"""
    global _system
    _system = system
    print("✓ GestureSystem injected into webserver")


# Sensor data structure (for compatibility with existing mobile app)
sensor_data = {
    'ituemg0': 0, 'ituemg1': 0, 'ituemg2': 0, 'ituemg3': 0, 
    'ituemg4': 0, 'ituemg5': 0, 'ituemg6': 0, 'ituemg7': 0,
    'ituax': 0, 'ituay': 0, 'ituaz': 0, 'itugx': 0, 'itugy': 0, 
    'itugz': 0, 'ituroll': 0, 'ituyaw': 0, 'itupitch': 0,
    'maremg0': 0, 'maremg1': 0, 'maremg2': 0, 'maremg3': 0, 
    'maremg4': 0, 'maremg5': 0, 'maremg6': 0, 'maremg7': 0,
    'marax': 0, 'maray': 0, 'maraz': 0, 'margx': 0, 'margy': 0, 
    'margz': 0, 'marroll': 0, 'maryaw': 0, 'marpitch': 0,
    #'cr': 0,      # Classification result
    'calword': 0,
    'connections': 0 #add if connections are available with myo armbands 
}
# Global variable to store latest classification result
_latest_result = {
    'cr': None,
    'timestamp': None
}

# Result history for status endpoint
_result_history = []  # List of dicts: [{gesture, timestamp}, ...]
MAX_HISTORY_SIZE = 10  # Keep last 10 results

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Root endpoint"""
    sensor_data['connections'] += 1
    return jsonify({
        "status": "ok", 
        "message": "Raspberry Pi Gesture Recognition Server",
        "data_acquisition_running": _system.is_data_acquisition_running() if _system else False
    })

@app.route('/data')
def data():
    """Get current sensor data from shared memory"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    if not _system.is_data_acquisition_running():
        return jsonify({
            "status": "error",
            "message": "Data acquisition not running. Call /connect first."
        }), 400
    
    # Read last sample from circular buffer (NO modification to shared memory)
    current_idx = _system.stream_index.value
    if current_idx == 0:
        return jsonify({
            "status": "no_data",
            "message": "No data available yet. Wait for sensors to start streaming."
        })
    
    # Get the most recent sample (use modulo for circular buffer)
    from main import STREAM_BUFFER_SIZE
    buffer_idx = (current_idx - 1) % STREAM_BUFFER_SIZE
    latest_sample = _system.stream_buffer[buffer_idx].copy()  # .copy() to avoid shared memory issues
    
    # Parse into mobile app format (alphabetically: MyoITU first, MyoMarmara second)
    sensor_data = {
        # First Myo (index 0-16)
        'ituemg0': float(latest_sample[0]), 'ituemg1': float(latest_sample[1]),
        'ituemg2': float(latest_sample[2]), 'ituemg3': float(latest_sample[3]),
        'ituemg4': float(latest_sample[4]), 'ituemg5': float(latest_sample[5]),
        'ituemg6': float(latest_sample[6]), 'ituemg7': float(latest_sample[7]),
        'ituroll': float(latest_sample[8]), 'itupitch': float(latest_sample[9]),
        'ituyaw': float(latest_sample[10]),
        'ituax': float(latest_sample[11]), 'ituay': float(latest_sample[12]),
        'ituaz': float(latest_sample[13]),
        'itugx': float(latest_sample[14]), 'itugy': float(latest_sample[15]),
        'itugz': float(latest_sample[16]),
        
        # Second Myo (index 17-33)
        'maremg0': float(latest_sample[17]), 'maremg1': float(latest_sample[18]),
        'maremg2': float(latest_sample[19]), 'maremg3': float(latest_sample[20]),
        'maremg4': float(latest_sample[21]), 'maremg5': float(latest_sample[22]),
        'maremg6': float(latest_sample[23]), 'maremg7': float(latest_sample[24]),
        'marroll': float(latest_sample[25]), 'marpitch': float(latest_sample[26]),
        'maryaw': float(latest_sample[27]),
        'marax': float(latest_sample[28]), 'maray': float(latest_sample[29]),
        'maraz': float(latest_sample[30]),
        'margx': float(latest_sample[31]), 'margy': float(latest_sample[32]),
        'margz': float(latest_sample[33]),
        
        'calword': 0,
        'connections': 0 #add if connections are available with myo armbands 
    }
    
    return jsonify(sensor_data)

@app.route('/connect')
def connect():
    """Start data acquisition process """
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    if _system.is_data_acquisition_running():
        return jsonify({
            "status": "already_connected",
            "message": "Data acquisition already running"
        })
    
    success = _system.start_data_acquisition()
    
    if success:
        return jsonify({
            "status": "connecting",
            "message": "Data acquisition started successfully"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Failed to start data acquisition"
        }), 500

@app.route('/disconnect')
def disconnect():
    """Stop data acquisition process"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    _system.stop_data_acquisition()
    
    return jsonify({
        "status": "disconnected",
        "message": "Data acquisition stopped"
    })

@app.route('/startcf')
def startcf():
    """Start classification"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    if not _system.is_data_acquisition_running():
        return jsonify({
            "status": "error",
            "message": "Data acquisition not running. Call /connect first."
        }), 400
    
    _system.start_classification()
    
    return jsonify({
        "status": "success",
        "message": "Classification started"
    })

@app.route('/stopcf')
def stopcf():
    """Stop classification"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    _system.stop_classification()
    
    return jsonify({
        "status": "success",
        "message": "Classification stopped"
    })

@app.route('/startPcf')
def startPcf():
    """Start personal classification"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    if not _system.is_data_acquisition_running():
        return jsonify({
            "status": "error",
            "message": "Data acquisition not running. Call /connect first."
        }), 400
    
    _system.start_personal_classification()
    
    return jsonify({
        "status": "success",
        "message": "Personal classification started"
    })

@app.route('/stopPcf')
def stopPcf():
    """Stop personal classification"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    _system.stop_personal_classification()
    
    return jsonify({
        "status": "success",
        "message": "Personal classification stopped"
    })

@app.route('/result')
def result():
    global _latest_result, _result_history
    
    result_value = request.args.get('value', type=str)
    
    if result_value:
        import time
        current_time = time.time()
        
        # Store as latest result
        _latest_result = {
            'cr': result_value,
            'timestamp': current_time
        }
        
        # Add to history
        _result_history.append({
            'cr': result_value,
            'timestamp': current_time
        })
        
        # Trim history if too large
        if len(_result_history) > MAX_HISTORY_SIZE:
            _result_history.pop(0)
        
        print(f"🔥 Received classification result: {result_value}")
        return jsonify({
            "status": "success",
            "message": f"Result '{result_value}' received"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Missing 'value' parameter"
        }), 400

@app.route('/get_result')
def get_result():
    """Get latest classification result (polled by mobile app) - AUTO-RESETS after read"""
    global _latest_result
    
    if _latest_result['cr'] is None:
        return jsonify({
            "status": "no_result",
            "message": "No classification result available yet",
            "cr": "none"
        })
    
    # Capture result before resetting
    result_to_return = {
        "status": "success",
        "cr": _latest_result['cr'],
        "timestamp": _latest_result['timestamp']
    }
    
    # RESET after reading (one-time consumption)
    _latest_result = {
        'cr': None,
        'timestamp': None
    }
    
    return jsonify(result_to_return)

@app.route('/train') 
def train():
    """Train both models (RestModel + GestureModel)"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    print("Training models...")
    success = _system.train_models()
    
    if success:
        return jsonify({
            "status": "success",
            "message": "Models trained successfully"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Training failed. Check logs."
        }), 500

@app.route('/Ptrain') 
def Personaltrain():
    """Train the personal model"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    print("Training personal model...")
    success = _system.train_personal_model()
    
    if success:
        return jsonify({
            "status": "success",
            "message": "Personal model trained successfully"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Personal training failed. Check logs for details."
        }), 500
    
@app.route('/setcw')
def setcw():
    """Start calibration in background (non-blocking)"""
    if _system is None:
        return jsonify({"status": "error", "message": "System not initialized"}), 500
    
    value = request.args.get('value', type=int)
    gesture_name = request.args.get('name', type=str)
    
    if gesture_name is None and value is not None:
        gesture_name = f"gesture_{value}"
    elif gesture_name is None:
        return jsonify({"status": "error", "message": "Missing 'value' or 'name' parameter"}), 400
    
    if not _system.is_data_acquisition_running():
        return jsonify({"status": "error", "message": "Data acquisition not running"}), 400
    
    # Note: calibrate() now uses rest-to-rest detection (30s timeout)
    # Start async calibration
    success = _system.calibrate(gesture_name)
    
    if success:
        return jsonify({
            "status": "started",
            "message": f"Calibration started for '{gesture_name}'. Poll /calibration_status for updates.",
            "gesture_name": gesture_name
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Calibration already running or failed to start"
        }), 400
    
@app.route('/deletecw')
def deletecw():
    """Delete all npy files starting with 'name' in current calibration folder"""
    if _system is None:
        return jsonify({"status": "error", "message": "System not initialized"}), 500
    
    name = request.args.get('name', type=str)
    if not name:
        return jsonify({"status": "error", "message": "Missing 'name' parameter"}), 400
    
    deleted_count = _system.delete_gesture_samples(name)
    
    if deleted_count == 0:
        return jsonify({
            "status": "success",
            "message": f"This file doesn't exist. There is no file to be deleted.",
            "deleted_count": 0
        })
    else:
        return jsonify({
            "status": "success",
            "message": f"Deleted {deleted_count} samples for '{name}'",
            "deleted_count": deleted_count
        })

@app.route('/stopcb')
def stopcb():
    """Stop ongoing calibration"""
    if _system is None:
        return jsonify({"status": "error", "message": "System not initialized"}), 500
    
    success = _system.stop_calibration()
    
    if success:
        return jsonify({
            "status": "success",
            "message": "Calibration stopped"
        })
    else:
        return jsonify({
            "status": "info",
            "message": "No calibration was running"
        })
    
@app.route('/Gsetcw')
def Generalsetcw():
    """Save calibration sample for general model (timed recording)"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    # Get gesture name/number from query params
    value = request.args.get('value', type=int)
    gesture_name = request.args.get('name', type=str)
    
    # Support both numeric IDs and string names
    if gesture_name is None and value is not None:
        gesture_name = f"gesture_{value}"
    elif gesture_name is None:
        return jsonify({
            "status": "error",
            "message": "Missing 'value' or 'name' parameter"
        }), 400
    
    if not _system.is_data_acquisition_running():
        return jsonify({
            "status": "error",
            "message": "Data acquisition not running. Call /connect first."
        }), 400
    
    success = _system.general_calibrate(gesture_name)
    
    if success:
        # Count samples in 'user' folder (general calibration saves here)
        generalcalibrationfolder="user"
        matching_files = glob.glob(f'{generalcalibrationfolder}/{gesture_name}_*.npy')
        sample_count = len(matching_files)
        
        return jsonify({
            "status": "success",
            "message": f"General calibration sample saved for '{gesture_name}'",
            "gesture_name": gesture_name,
            "total_samples": sample_count,
            "method": "timed_recording",
            "nextcal": "ok"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "General calibration failed",
            "nextcal": "notok"
        }), 500

@app.route('/pf')
def set_personal_folder():
    """Set personal folder for user-specific training and calibration"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    name = request.args.get('name', type=str)
    
    if not name:
        return jsonify({
            "status": "error",
            "message": "Missing 'name' parameter"
        }), 400
    
    # Validate name (no special characters for safety)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        return jsonify({
            "status": "error",
            "message": "Invalid name. Use only letters, numbers, underscore and dash."
        }), 400
    
    folder = _system.set_personal_folder(name)
    
    return jsonify({
        "status": "success",
        "message": f"Personal folder set to '{folder}'",
        "folder": folder,
        "exists": True  # Folder is created automatically
    })

@app.route('/calibration_status')
def calibration_status():
    """Get real-time calibration status (polled by mobile app)"""
    if _system is None:
        return jsonify({"status": "error", "message": "System not initialized"}), 500
    
    status = _system.get_calibration_status()
    
    if not status['active'] and not status['messages']:
        return jsonify({
            "status": "no_calibration",
            "message": "No calibration in progress or recent history"
        })
    
    return jsonify({
        "status": "active" if status['active'] else "complete",
        "active": status['active'],
        "messages": status['messages'],
        "latest": status['latest']
    })

@app.route('/status')
def status():
    """Get system status"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    # Check if models are trained
    rest_model_trained = (
        _system.rest_model is not None and 
        _system.rest_model.model is not None
    )
    gesture_model_trained = (
        _system.gesture_model is not None and 
        _system.gesture_model.model is not None
    )
    personal_model_trained = (
        hasattr(_system, 'Pgesture_model') and
        _system.Pgesture_model is not None and 
        _system.Pgesture_model.model is not None
    )
    
    # Get gesture labels if available
    gesture_labels = []
    if gesture_model_trained:
        gesture_labels = _system.gesture_model.gesture_labels
    
    personal_gesture_labels = []
    if personal_model_trained:
        personal_gesture_labels = _system.Pgesture_model.gesture_labels
    
    # Determine which model is currently active
    active_model = "none"
    if _system.is_running_flag.value == 1:
        active_model = "general"
    elif _system.Pis_running_flag.value == 1:
        active_model = "personal"
    
    return jsonify({
        "status": "success",
        "data": {
            "data_acquisition_running": _system.is_data_acquisition_running(),
            "classification_running": _system.is_classification_running(),
            "active_model": active_model,
            "current_personal_folder": _system.current_user_folder,
            "rest_model_trained": rest_model_trained,
            "gesture_model_trained": gesture_model_trained,
            "personal_model_trained": personal_model_trained,
            "available_gestures": gesture_labels,
            "personal_gestures": personal_gesture_labels,
            "latest_result": _latest_result['cr'],
            "result_history": _result_history[-10:],  # Last 10 results
            "total_classifications": len(_result_history)
        }
    })

# ==================== Main ====================

if __name__ == '__main__':
    print("=" * 50)
    print("ERROR: Do not run webserver.py directly!")
    print("Please run main.py instead.")
    print("=" * 50)