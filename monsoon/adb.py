import os
import subprocess
import re
import pandas as pd
import time
import stat
from threading import Thread, Event
import Monsoon.HVPM as HVPM
import Monsoon.sampleEngine as sampleEngine
import Monsoon.pmapi as pmapi
import numpy as np

def run_command(command):
    print(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
        print(f"Output: {result}")
        return True, result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output}")
        return False, e.output
    
def extract_avg(output):
    timings_pattern = re.compile(r'Timings \(microseconds\):\s+count=\d+\s+first=\d+\s+curr=\d+\s+min=\d+\s+max=\d+\s+avg=(\d+(\.\d+)?)\s+std=\d+')
    match = timings_pattern.search(output)
    if match:
        return match.group(1)
    return None

def extract_all_timestamps(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        if not lines:
            return []
        start_time = float(lines[0].strip().split(": ")[1])
        end_time = float(lines[-1].strip().split(": ")[1])
    return [(start_time, end_time)]

def power_monitor_setup(serialno, protocol):
    HVMON = HVPM.Monsoon()
    HVMON.setup_usb(serialno, protocol)
    print("HVPM Serial Number: " + repr(HVMON.getSerialNumber()))

    try:
        HVMON.stopSampling()
    except Exception as e:
        print(f"Error while stopping previous sampling: {e}")

    HVMON.setPowerUpCurrentLimit(8)
    HVMON.setRunTimeCurrentLimit(8)
    HVMON.fillStatusPacket()
    HVMON.setVout(4.2)
    HVengine = sampleEngine.SampleEngine(HVMON)
    HVengine.ConsoleOutput(True)
    HVengine.enableChannel(sampleEngine.channels.MainCurrent)
    HVengine.enableChannel(sampleEngine.channels.MainVoltage)
    return HVMON, HVengine

def baseline_sampling(HVMON, HVengine, duration):
    print("Starting baseline sampling...")
    numSamples = sampleEngine.triggers.SAMPLECOUNT_INFINITE
    HVengine.setStartTrigger(sampleEngine.triggers.GREATER_THAN, 0)
    HVengine.setStopTrigger(sampleEngine.triggers.GREATER_THAN, duration)
    HVengine.setTriggerChannel(sampleEngine.channels.timeStamp)

    HVengine.enableCSVOutput("baseline.csv")
    HVengine.startSampling(numSamples, 1)
    time.sleep(duration)  # Baseline sampling for the specified duration
    HVengine.disableCSVOutput()
    HVMON.stopSampling()  # Ensure the power monitor exits sample mode
    print("Baseline sampling complete.")

def inference_sampling(HVMON, HVengine, start_event, stop_event, model_name, inference_dir):
    start_event.wait()  # Wait for the signal to start inference sampling
    print(f"Starting inference sampling for {model_name}...")
    numSamples = sampleEngine.triggers.SAMPLECOUNT_INFINITE
    csv_file = os.path.join(inference_dir, f"inference_{model_name}.csv")
    HVengine.enableCSVOutput(csv_file)
    HVengine.startSampling(numSamples, 1)

    stop_event.wait()  # Wait for the signal to stop inference sampling
    HVengine.disableCSVOutput()
    HVMON.stopSampling()  # Ensure the power monitor exits sample mode
    print(f"Inference sampling for {model_name} complete.")

def calculate_average_current_voltage(csv_file, start_time, end_time):
    df = pd.read_csv(csv_file)
    df_filtered = df[(df['Time(ms)'] >= start_time) & (df['Time(ms)'] <= end_time)]
    avg_current = df_filtered['Main(mA)'].mean()
    avg_voltage = df_filtered['Main Voltage(V)'].mean()
    return avg_current, avg_voltage

def run_inference(models_dir, results_file, start_event, stop_event, baseline_event, inference_event, HVMON, HVengine, inference_script, baseline_dir, inference_dir, timestamp_dir):
    models = [f for f in os.listdir(models_dir) if f.endswith('.tflite')]

    # baseline_event.wait()  # Wait for baseline sampling to complete

    # baseline_current, _ = calculate_average_current_voltage(os.path.join(baseline_dir,"baseline.csv"), 0, float('inf'))
    # print(f"Baseline Current: {baseline_current} mA")

    for model_name in models:
        model_path = os.path.join(models_dir, model_name)

        # Start inference sampling in a separate thread for each model
        # sampling_thread = Thread(target=inference_sampling, args=(HVMON, HVengine, start_event, stop_event, model_name, inference_script))
        sampling_thread = Thread(
            target=inference_sampling,
            args=(HVMON, HVengine, start_event, stop_event, model_name, inference_dir)
        )

        sampling_thread.start()

        start_event.set()
        time.sleep(1)  # Ensure sampling starts 1 second before inference
        inference_event.set()  # Notify that inference is about to start

        # success, output = run_command(["bash", os.path.join(inference_script, "inference.sh"), model_path, model_name, timestamp_dir])
        inference_script_path = inference_script  # 直接传入完整路径，比如 "monsoon/inference.sh"
        success, output = run_command(["bash", inference_script_path, model_path, model_name, timestamp_dir])

        if not success:
            print(f"Inference run failed for {model_name}.")
            stop_event.set()  # Stop the sampling
            sampling_thread.join()  # Wait for the sampling thread to finish
            continue

        avg_time = extract_avg(output)
        avg_time_str = f"{avg_time} microseconds" if avg_time else "N/A"
        print(f"Average Inference Time for {model_name}: {avg_time_str}")

        timestamps_file = os.path.join(timestamp_dir, f"timestamps_{model_name}.txt")
        timestamps = extract_all_timestamps(timestamps_file)

        csv_file = os.path.join(inference_dir, f"inference_{model_name}.csv")
        avg_current, avg_voltage = calculate_average_current_voltage(csv_file, timestamps[0][0], timestamps[-1][1])

        inference_current = avg_current
        results_file.write(f"{model_name}, {avg_time_str}, {inference_current} mA, {avg_voltage} V\n")
        print(f"Average Inference Current for {model_name}: {inference_current} mA")
        print(f"Average Voltage for {model_name}: {avg_voltage} V")

        stop_event.set()  # Notify power monitoring to stop
        sampling_thread.join()  # Wait for the sampling thread to finish

        # Reset events for the next model
        start_event.clear()
        stop_event.clear()
        inference_event.clear()

        time.sleep(1)  # Wait 1 second before starting the next model inference

def check_device_connection(adb_connect_script):
    try:
        devices_output = subprocess.check_output(["adb", "devices"], text=True).strip()
        device_list = devices_output.split("\n")[1:]  # Skip the first line which is just the header
        connected_devices = [line for line in device_list if "\tdevice" in line]

        if not connected_devices:
            # If no device is connected, run adb_connect_script
            if not os.path.exists(adb_connect_script):
                print(f"{adb_connect_script} not found.")
                return False

            st = os.stat(adb_connect_script)
            os.chmod(adb_connect_script, st.st_mode | stat.S_IEXEC)

            retcode = subprocess.call(["bash", adb_connect_script])
            if retcode != 0:
                print("Failed to connect to device via ADB.")
                return False
            print("ADB connection successful")

            devices_output = subprocess.check_output(["adb", "devices"], text=True).strip()
            device_list = devices_output.split("\n")[1:]
            connected_devices = [line for line in device_list if "\tdevice" in line]

            if not connected_devices:
                print("No device found. Please ensure the device is connected and try again.")
                return False
            print("Device verified as connected")
        else:
            print("Device already connected")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False

def main(args):
    models_dir = args.models_dir
    inference_script = args.inference_script
    adb_connect_script = args.adb_connect_script
    result_file_path = args.result_file_path
    baseline_dir = args.baseline_dir
    inference_dir = args.inference_dir
    timestamp_dir = args.timestamp_dir

    baseline_duration = 10  # seconds

    baseline_event = Event()
    start_event = Event()
    stop_event = Event()
    inference_event = Event()

    HVMON, HVengine = power_monitor_setup(26589, pmapi.USB_protocol())

    # Ensure phone is connected and press enter to confirm
    # input("Ensure the phone is powered on, then press Enter to confirm...")
    # time.sleep(2)

    # Start baseline sampling in the main thread
    # baseline_sampling(HVMON, HVengine, baseline_duration)
    # baseline_event.set()

    print("Monsoon power monitor thread started")

    if not check_device_connection(adb_connect_script):
        return None  # If the device isn't connected, return None

    power_results = {}

    with open(result_file_path, "w") as results_file:
        results_file.write("Model Name, Average Inference Time (microseconds), Inference Current (mA), Average Voltage (V)\n")

        run_inference(models_dir, results_file, start_event, stop_event, baseline_event, inference_event, HVMON, HVengine, inference_script, baseline_dir, inference_dir, timestamp_dir)

    with open(result_file_path, "r") as results_file:
        next(results_file)  # Skip the header line
        for line in results_file:
            parts = line.strip().split(", ")
            model_name = parts[0]
            power_consumption = float(parts[2].split(" ")[0])  # Extract the current in mA
            power_results[model_name] = power_consumption

    return power_results  # Return a dictionary of power consumption keyed by model name

# if __name__ == "__main__":
#     import sys
#     models_dir = sys.argv[1] if len(sys.argv) > 1 else None
#     results = main(models_dir)
#     print(results)  # Or handle it as needed

