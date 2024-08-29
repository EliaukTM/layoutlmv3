import time
time.sleep(150)

from structure.model_init import Layoutlmv3_Predictor
from flask import Flask, request
from waitress import serve
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

model = Layoutlmv3_Predictor("models/model_final.pth")


def process_base64_image(base64_string, max_size=3000):
    # 解码 base64 字符串
    image_data = base64.b64decode(base64_string)

    # 将二进制数据转换为 PIL Image 对象
    image = Image.open(io.BytesIO(image_data))

    # 检查图像尺寸
    if image.width > max_size or image.height > max_size:
        # 如果图像过大，进行等比例缩放
        image.thumbnail((max_size, max_size), Image.LANCZOS)

    # 将 PIL Image 转换为 NumPy 数组，并反转颜色通道顺序（RGB 到 BGR）
    image_array = np.array(image)[:, :, ::-1]

    return image_array


@app.route("/layoutlmv3", methods=['POST', 'GET'])
def translate():
    image_base64 = request.json['image']
    image = process_base64_image(image_base64)
    img_H, img_W = image.shape[0], image.shape[1]
    layout_res = model(image, ignore_catids=[])
    return {
        'width': img_W,
        'height': img_H,
        'layout': layout_res
    }


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8099)
