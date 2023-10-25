from flask import Flask, request, jsonify, render_template, send_from_directory

import os
import glob
import cv2
import base64
import numpy as np
import urllib
from lib.mizutama import Mizutama
from lib.mizutama import process_images

app = Flask(__name__)
app.debug = True

@app.route('/api')
def api():
    url = request.args.get('url')
    if url is None:
        return jsonify(error='"url" is required.')
    try:
        data = urllib.urlopen(url).read()
    except Exception:
        return jsonify(error='urlopen failed.')

    buf = np.fromstring(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error='read image failed.')

    mizutama = Mizutama(img)
    img = mizutama.collage()
    return jsonify(image='data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', img)[1]), url=url)


@app.route('/originImages')
def origin_image_list():
    # 获取root/test/下的所有图片文件名
    image_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
    image_extensions = ["*.png", "*.jpg", "*.jpeg"]

    # 使用glob来获取所有图片
    image_files = [os.path.basename(img) for ext in image_extensions for img in
                   glob.glob(os.path.join(image_directory, ext))]

    return jsonify(image_files)
@app.route('/processedImages')
def processed_image_list():
    # 获取root/test/下的所有图片文件名
    image_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test2')
    image_extensions = ["*.png", "*.jpg", "*.jpeg"]

    # 使用glob来获取所有图片
    image_files = [os.path.basename(img) for ext in image_extensions for img in
                   glob.glob(os.path.join(image_directory, ext))]

    return jsonify(image_files)


@app.route('/originImages/<filename>')
def serve_origin_image(filename):
    image_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
    return send_from_directory(image_directory, filename)
@app.route('/processedImages/<filename>')
def serve_processed_image(filename):
    image_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test2')
    return send_from_directory(image_directory, filename)

@app.route('/handleProcess')
def handle_process():
    try:
        process_images()
        return jsonify({"status": "success", "message": "Images processed successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/')
def main():
    return render_template('index.html')

app.run()
