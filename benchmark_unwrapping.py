import os
import csv
import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import dhm

def benchmark_unwrapping(folder_path, output_csv='unwrapping_benchmark.csv'):
    results = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'angle.csv':
                csv_path = os.path.join(root, file)
                try:
                    # Read CSV as 2D array (assuming comma-separated values)
                    data = np.loadtxt(csv_path, delimiter=',')
                    if data.ndim == 1:
                        data = data.reshape(1, -1)  # Handle 1D case
                    elif data.ndim > 2:
                        continue  # Skip if not 2D

                    pic = dhm.dhmPic()
                    pic.angle = data

                    # CPU unwrapping
                    start_time = time.time()
                    unwrapped_cpu = pic.dhm_unwrap(pic.angle)
                    cpu_time = time.time() - start_time

                    # Warm up GPU kernels and FFT plans for this input shape.
                    pic.dhm_unwrap_gpu(pic.angle)
                    cp.cuda.get_current_stream().synchronize()

                    # GPU unwrapping
                    start_time = time.time()
                    unwrapped_gpu = pic.dhm_unwrap_gpu(pic.angle)
                    cp.cuda.get_current_stream().synchronize()
                    gpu_time = time.time() - start_time

                    # Calculate improvements
                    time_improvement = cpu_time - gpu_time
                    percent_improvement = (time_improvement / cpu_time) * 100 if cpu_time > 0 else 0

                    # Save visualization images
                    cpu_dir = root
                    gpu_dir = root

                    # CPU visualization
                    plt.figure(figsize=(10, 8))
                    plt.imshow(unwrapped_cpu, cmap='viridis', aspect='auto')
                    plt.colorbar(label='Phase (radians)')
                    plt.title('CPU Phase Unwrapping')
                    plt.tight_layout()
                    cpu_output_path = os.path.join(cpu_dir, 'phase_CPU.png')
                    plt.savefig(cpu_output_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    # GPU visualization
                    plt.figure(figsize=(10, 8))
                    plt.imshow(unwrapped_gpu, cmap='viridis', aspect='auto')
                    plt.colorbar(label='Phase (radians)')
                    plt.title('GPU Phase Unwrapping')
                    plt.tight_layout()
                    gpu_output_path = os.path.join(gpu_dir, 'phase_GPU.png')
                    plt.savefig(gpu_output_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    results.append({
                        'file': csv_path,
                        'cpu_time': cpu_time,
                        'gpu_time': gpu_time,
                        'time_improvement_sec': time_improvement,
                        'percent_improvement': percent_improvement
                    })

                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['file', 'cpu_time', 'gpu_time', 'time_improvement_sec', 'percent_improvement']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Benchmark results saved to {output_csv}")

if __name__ == "__main__":
    # Example usage: replace with your folder path
    folder_path = "X:/expData/260108/"
    benchmark_unwrapping(folder_path)
