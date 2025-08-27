# Image Classification Web Demo

This project provides a simple **web demo** for image classification using both a **local PyTorch model** and the **Alibaba Cloud Image Recognition API**.  
The demo allows you to upload an image, view predictions from both models, and compare the results in a web interface powered by [Gradio](https://gradio.app).


## 📂 Project Structure

```bash
├── README.md # Project documentation
├── examples/ # Example images for testing
├── labels.txt # ImageNet class labels (1000 categories)
└── web.py # Main Gradio app
```

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-repo/image-classification-webdemo.git
cd image-classification-webdemo
```

### 2. Install dependencies

```bash
pip install torch torchvision gradio alibabacloud_imagerecog20190930
```

### 3. Configure environment variables

You need to set your Alibaba Cloud credentials:
```bash
export ALIBABA_CLOUD_ACCESS_KEY_ID="your_access_key_id"
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="your_access_key_secret"
```

▶️ Usage

Run the Gradio app:
```bash
python web.py
```

This will launch a local web server (default: http://127.0.0.1:7860).
Open it in your browser to upload images and view results.