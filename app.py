import os
import mysql.connector as conn
from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import uuid
import numpy as np

app = Flask(__name__)

model_toba = load_model('model/toba-classification.h5')

model_toba.make_predict_function()

model_karo = load_model('model/karo-classification.h5')

model_karo.make_predict_function()

def connection():
    koneksi = conn.connect(host='localhost', user='root', password='')
    kursor = koneksi.cursor()
    kursor.execute('use db_api_tenun')
    return koneksi, kursor


def predict_label_toba(img_path):
    names = ['Ulos Bintang Maratur', 'Ulos Harungguan', 'Ulos Mangiring', 'Ulos Ragi Hidup', 'Ulos Ragi Hotang',
             'Ulos Sadum', 'Ulos Sibolang', 'Ulos Sitolutuho']
    img_ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_test = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    img_test = cv2.resize(img_test, (150, 150))
    img_test = img_test.reshape(1, 150, 150, 3)
    # class_pred = model_toba.predict(img_test)
    class_pred = np.argmax(model_toba.predict(img_test), axis=1)
    print(class_pred)
    p = names[class_pred[0]]
    save_path = "static/toba/" + p.lower() + "/" + str(uuid.uuid4()) + ".png"
    cv2.imwrite(save_path, img_ori)
    return p, save_path


def predict_label_karo(img_path):
    names = ['Bulang', 'Gara', 'Julu', 'Tudung', 'Uis Nipes']
    img_ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_test = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    img_test = cv2.resize(img_test, (150, 150))
    img_test = img_test.reshape(1, 150, 150, 3)
    # class_pred = model_karo.predict(img_test)
    class_pred = np.argmax(model_karo.predict(img_test), axis=1)
    print(class_pred)
    p = names[class_pred[0]]
    save_path = "static/karo/" + p.lower() + "/" + str(uuid.uuid4()) + ".png"
    cv2.imwrite(save_path, img_ori)
    return p, save_path


# routes
@app.route("/toba", methods=['GET', 'POST'])
@app.route("/", methods=['GET', 'POST'])
def maintoba():
    return render_template("index.html", typesubmit="/submittoba", jenis="Toba")


@app.route("/karo", methods=['GET', 'POST'])
def mainkaro():
    return render_template("index.html", typesubmit="/submitkaro", jenis="Karo")


@app.route("/submittoba", methods=['GET', 'POST'])
def get_output_toba():
    if request.method == 'POST':
        img = request.files['my_image']
        name = img.filename
        img_path = "static/" + str(uuid.uuid4()) + ".png"
        print(img_path)
        img.save(img_path)
        p, save_path = predict_label_toba(img_path)
        os.remove(img_path)
        koneksi, kursor = connection()
        kursor.execute(
            f'''insert into classification (original_name,image_path,classification_result) values (\'{name}\', \'{save_path}\', \'{p}\');''')
        koneksi.commit()
        koneksi.close()
        print(save_path)

    return render_template("index.html", prediction=p, img_path=save_path, jenis="Toba")


@app.route("/submitkaro", methods=['GET', 'POST'])
def get_output_karo():
    if request.method == 'POST':
        img = request.files['my_image']
        name = img.filename
        img_path = "static/" + str(uuid.uuid4()) + ".png"
        img.save(img_path)
        p, save_path = predict_label_karo(img_path)
        os.remove(img_path)
        koneksi, kursor = connection()
        kursor.execute(
            f'''insert into classification (original_name,image_path,classification_result) values (\'{name}\', \'{save_path}\', \'{p}\');''')
        koneksi.commit()
        koneksi.close()
        print(save_path)

    return render_template("index.html", prediction=p, img_path=save_path, jenis="Karo")


if __name__ == '__main__':
    app.run(debug=True)
