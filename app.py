import os
import io
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename

import torch
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image

import network

app = Flask(__name__)

UPLOAD_FOLDER = './tmp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMAGE_WIDTH = 640
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

if os.path.exists(UPLOAD_FOLDER):
    pass
else:
    os.mkdir(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        img_file = request.files['img_file']
        style = request.form["style"]

        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
        else:
            return ''' <p>許可されていない拡張子です</p> '''

        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # リサイズ
        raw_img = cv2.resize(img, (IMAGE_WIDTH, int(IMAGE_WIDTH*img.shape[0]/img.shape[1])))

        # リサイズ画像の保存
        raw_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_'+filename)
        cv2.imwrite(raw_img_url, raw_img)

        #------ ここから推論処理 -------------------
        device = "cpu"

        # 前処理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
        inputs = transform(raw_img).unsqueeze(0).to(device)

        # モデルの読み込み
        model_map = {
            'deeplabv3_resnet50': network.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': network.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
        }
        model = model_map["deeplabv3plus_mobilenet"](num_classes=21)
        model.load_state_dict(torch.load("./weights/best_deeplabv3plus_mobilenet_voc_os16.pth", map_location=torch.device('cpu'))["model_state"])
        model.to(device)

        # 推論
        model.eval()
        pred = model(inputs)

        # 後処理
        mask = torch.argmax(pred[0], 0)
        mask = np.stack([mask, mask, mask], 2)
        res = np.where(mask != 15, raw_img , 0)

        # スタイル画像を読み込み
        mosaic = cv2.imread("./static/images/{}.jpg".format(style))
        mosaic = cv2.resize(mosaic, (res.shape[1], res.shape[0]))

        out = np.where(res == 0, mosaic, res)

        # 結果を保存
        out_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'out_'+filename)
        cv2.imwrite(out_img_url, out)

        return render_template('result.html', raw_img_url=raw_img_url, out_img_url=out_img_url)

    else:
        return redirect(url_for('index'))

@app.route('/tmp/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)

'''
host="0.0.0.0",
port=int(os.environ.get("PORT", 5000))
'''
    