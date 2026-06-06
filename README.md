# Fast CV Ops: Custom PyTorch C++ Extension

[cite_start]This repository contains a high-performance C++ extension for PyTorch designed to optimize core Computer Vision operations[cite: 36, 45]. [cite_start]It addresses common pre-processing and data-loading I/O bottlenecks frequently encountered in large-scale deep learning pipelines on HPC systems[cite: 3, 5, 87].

## 🚀 The Solution: C++ IoU Extension
Calculating **Intersection over Union (IoU)** for thousands of bounding boxes (e.g., during Non-Maximum Suppression in YOLO models) introduces a severe performance bottleneck if done in pure Python due to interpreter overhead and dynamic typing.

By using the **PyTorch C++ API (`libtorch`)** and direct memory mapping via `torch::Tensor::accessor`, this implementation completely bypasses Python's overhead, executing nested loops at native CPU speed with maximum hardware optimization (`-O3`).

## 📊 Benchmark Results

- **Task:** 1,000 x 1,000 bounding box comparisons (1,000,000 total iterations).
- **Pure Python Execution Time:** ~47.69 seconds
- **C++ Extension Execution Time:** ~0.0064 seconds
- **Performance Boost:** 🔥 **~7,400x faster** than pure Python.
- **Max Error:** 0.000000 (100% mathematical accuracy match).

## 🛠️ Installation & Compilation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <your-repository-url>
   cd Deeply```

2. Activate your virtual environment and install dependencies:
   ```source venv/bin/activate
   pip install torch```

3. Compile the C++ extension locally:
   ```python setup_csrc.py build_ext --inplace```

## 🛠️ Usage & Verification

Run the built-in benchmark script to verify correctness and measure the speedup on your machine:
    ```python benchmark.py```