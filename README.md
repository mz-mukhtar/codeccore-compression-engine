# 🗜️ CodecCore: Multimedia Compression Engine

CodecCore is an educational and highly functional communications systems sandbox. It allows users to track, analyze, and experiment with lossy and lossless data compression algorithms natively, without relying on opaque third-party compression libraries. By building fundamental algorithms from scratch, CodecCore provides unparalleled insight into Information Theory and Digital Signal Processing (DSP).

## Core Features

*   **Text/Binary (Lossless):**
    *   Huffman Coding (Entropy Coding)
    *   LZ78 Dictionary Parsing
    *   Chained (LZ78 + Huffman)
*   **Images (Lossy):**
    *   Spatial Quantization
    *   RGB-to-YUV Color Space Conversions
    *   Run-Length Encoding (RLE)
*   **Audio Waveforms (DSP):**
    *   Standard Pulse-Code Modulation (PCM)
    *   Delta Modulation (DM)
    *   Adaptive DM (ADM via Jayant's Logic)
    *   Differential PCM (DPCM) with a closed-loop linear predictor

## System Architecture & Requirements

CodecCore operates within a securely locked `python:3.11-slim` container environment. This robust architectural choice guarantees cross-OS compatibility and completely eliminates host runtime conflicts (such as the removal of `audioop` in newer Python versions).

**Docker** is the only dependency required on the host system. All mathematical logic is executed purely within the container using `numpy`, `math`, and explicitly defined codec structures.

## Multi-OS Installation & Deployment Guide

Follow the steps below to download dependencies, clone the repository, build the image, and launch the CodecCore container on your specific platform.

### 0. Clone the Repository

First, clone the repository to your local machine and navigate into the project directory:

```bash
git clone https://github.com/mz-mukhtar/codeccore-compression-engine.git
cd codeccore-compression-engine
```

### 1. Windows Setup (PowerShell / Command Prompt)

*   **Prerequisite:** Install Docker Desktop via Winget:
    ```powershell
    winget install Docker.DockerDesktop
    ```
    *(Alternatively, download it directly from the official Docker website). Ensure the WSL2 backend is enabled in your Docker Desktop settings.*
*   **Build Command:**
    ```powershell
    docker build -t codeccore:latest .
    ```
*   **Run Command:**
    ```powershell
    docker run --rm -p 8501:8501 --name codeccore-app codeccore:latest
    ```

### 2. macOS Setup (Terminal)

*   **Prerequisite:** Install Docker Desktop via Homebrew:
    ```bash
    brew install --cask docker
    ```
    *(Alternatively, download the DMG matching your Intel or Apple Silicon architecture from the official Docker website).*
*   **Build Command:**
    ```bash
    docker build -t codeccore:latest .
    ```
*   **Run Command:**
    ```bash
    docker run --rm -p 8501:8501 --name codeccore-app codeccore:latest
    ```

### 3. Linux Setup (Ubuntu/Debian Bash)

*   **Prerequisite:** Install Docker Engine:
    ```bash
    sudo apt-get update && sudo apt-get install docker.io -y
    sudo systemctl enable --now docker
    sudo usermod -aG docker $USER
    ```
    *(Note: You must log out and log back in for the user group changes to take effect).*
*   **Build Command:**
    ```bash
    docker build -t codeccore:latest .
    ```
*   **Run Command:**
    ```bash
    docker run --rm -p 8501:8501 --name codeccore-app codeccore:latest
    ```

## How to Use the Application

Once the container is running, navigate to `http://localhost:8501` in your web browser.

*   **Compress Tab:**
    *   Upload any supported file (Text, Binary, Image, or Audio).
    *   Use the interactive UI sliders to tweak algorithm-specific parameters (such as Quantization Bit-Depth, Delta Step Size, or Predictor Weights).
    *   Observe the real-time mathematical analytics (Shannon Entropy, SQNR, PSNR).
    *   Download the mathematically compressed custom `.abc` archive format.
*   **Decompress Tab:**
    *   Upload your generated `.abc` archive.
    *   The engine automatically parses the custom `ARCH` binary header.
    *   Preview the seamlessly reconstructed file and download the decoded output (e.g., restored `.wav` or `.png`).
