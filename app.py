import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from io import BytesIO

app = FastAPI()

# 画像を読み込んで加工する関数
def process_image(image):
    # 画像を読み込む
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

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


# 画像を受け取るエンドポイント
@app.get("/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    processed_image = process_image(contents)

    # ここに画像から文字を抽出する処理を追加する
    # 例：text = extract_text(processed_image)

    # 処理結果を返す
    return {"message": "Image processed successfully!"}

# 画像から文字を抽出する関数
# この関数はまだ実装されていません。必要に応じて追加してください。
