import os
import torch
from torchvision import transforms
import gradio as gr
from alibabacloud_imagerecog20190930.client import Client
from alibabacloud_imagerecog20190930.models import TaggingImageAdvanceRequest
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions

from io import BytesIO

def pil_to_fileobj(img, format="JPEG"):
    """
    Convert PIL.Image to open(..., 'rb') like object
    """
    buf = BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf

# Load a pretrained model from torchvision
# Can change to other models, e.g., resnet50, vit_b_16, etc.
# See https://pytorch.org/vision/stable/models.html for more options
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).eval()

# human-readable labels for ImageNet.
# Use Local file labels.txt here
response = open("labels.txt", "r")
labels = [line.strip() for line in response.readlines()]

config = Config(
  access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
  access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
  endpoint='imagerecog.cn-shanghai.aliyuncs.com',
  region_id='cn-shanghai'
)

def predict(inp):
    # local model
    inp_tensor = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp_tensor)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    # aliyun api
    tagging_image_request = TaggingImageAdvanceRequest()
    tagging_image_request.image_urlobject = pil_to_fileobj(inp)
    runtime = RuntimeOptions()
    try:
        client = Client(config)
        response = client.tagging_image_advance(tagging_image_request, runtime)
        response_conf = response.body.data.to_map()['Tags']
        confidences_api = {d['Value']: round(d['Confidence'] / 100, 2) for d in response_conf}
    except Exception as error:
        print(error)
        print(error.code)
    return confidences, confidences_api

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=[gr.Label(label="Local Model", num_top_classes=3), gr.Label(label="Aliyun API", num_top_classes=3)],
             examples=["./examples/36_985_original.png", "./examples/140_388_original.png", "./examples/53_919_original.png", "./examples/93_468_original.png"]).launch()