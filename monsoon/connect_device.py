import subprocess
import time
import os

def run(cmd, shell=False):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=shell, check=True)

def prepare_android_device(device_ip="192.168.0.112:5555",
                                benchmark_cpu="/home/xiaolong/Downloads/GreenAuto/benchmark_tools/benchmark_model",
                                benchmark_gpu="/home/xiaolong/Downloads/GreenAuto/benchmark_tools/benchmark_model_gpu",
                                benchmark_remote="/data/local/tmp/benchmark_model"):
    # Step 0: USB plug-in prompt
    input("Please plug in the USB cable, then press Enter to continue...")

    print("\n[1] Killing adb server...")
    run(["adb", "kill-server"])

    print("[2] Restarting adb in USB mode...")
    run(["adb", "usb"])
    time.sleep(10)

    print("[3] Restarting adb in TCP/IP mode on port 5555...")
    run(["adb", "tcpip", "5555"])
    time.sleep(10)

    # Step 4: USB unplug prompt
    input("Now remove the USB cable, then press Enter to continue...")

    print(f"[4] Connecting to device at {device_ip}...")
    run(["adb", "connect", device_ip])

    print("[5] Verifying ADB connection...")
    run(["adb", "devices"])

    # Step 6: Push benchmark binary
    print(f"[6] Pushing {benchmark_cpu} to {benchmark_remote}...")
    run(["adb", "push", benchmark_cpu, benchmark_remote])

    print(f"[6] Pushing {benchmark_gpu} to {benchmark_remote}...")
    run(["adb", "push", benchmark_gpu, benchmark_remote])

    print("[7] Making benchmark_model executable...")
    run(["adb", "shell", f"chmod +x {benchmark_remote}"])

if __name__ == "__main__":
    prepare_android_device("192.168.0.112:5555")
