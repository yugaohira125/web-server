import cv2
import numpy as np
from flask import Flask, request

app = Flask(__name__)

#画像を読み込んで加工する関数
def process_image(image_path):
    # 画像を読み込む
    img = cv2.imread(image_path)

    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ぼかしを適用して影を軽減
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # 影の割合を計算して元画像から引く
    img2 = cv2.divide(gray, blur, scale=255)

    # 二値化する
    _, binary = cv2.threshold(img2, 198, 255, cv2.THRESH_BINARY)

    # 輪郭を検出
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 全て白の画像を作成
    img_blank = np.ones_like(img2) * 255

    # 輪郭だけを描画（黒色で描画）
    cv2.drawContours(img_blank, contours, -1, (0,0,0), 3)

    # 黒色で描画された部分のみを切り抜く
    processed_image = cv2.bitwise_and(img2, img_blank)

    return processed_image


#webアプリケーションのルート関数
@app.route('/', methods=['POST'])
def upload_image():
    image_file = request.files['file']
    if image_file:
        # 画像を加工する関数を実行
        processed_image = process_image(image_file)
        #画像から文字を抽出する関数を実行
        
    else:
        return 'No image file received'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

