import os
from flask import Flask, flash, request, render_template
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
import numpy as np
import cloudinary.uploader
from urllib.request import urlopen
from PIL import Image
import joblib

app = Flask(__name__)

app.static_folder = 'static'
filename = None

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CLOUD_NAME = 'dx8k8cjdq'
API_KEY = '495478988912747'
API_SECRET = '90xd1xw4Ck3qlCaitFDUBMzM4e4'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('intro.html')

@app.route('/insect')
def insect():
    return render_template('insect_recognition.html')

@app.route('/flower')
def flower():
    return render_template('flower_recognition.html')

@app.route('/sign_language')
def sign_language():
    return render_template('sign_language_recognition.html')

@app.route('/weather')
def weather():
    return render_template('weather_recognition.html')

@app.route('/insect_recogition', methods=['POST'])
def insect_recogition():
    if 'file' not in request.files:
        flash('Không tìm thấy file!')
        return render_template('insect_recognition.html')

    file = request.files['file']

    if file.filename == '':
        flash('Không có ảnh nào được tải lên!')
        return render_template('insect_recognition.html')

    if file and allowed_file(file.filename):
        # Upload ảnh lên cloudinary
        cloudinary.config(cloud_name = CLOUD_NAME, api_key=API_KEY, 
            api_secret=API_SECRET)
        upload_result = None

        upload_result = cloudinary.uploader.upload(file)

        filename = upload_result['url']

        # Load model có độ chính xác cao nhất
        loaded_best_model = keras.models.load_model("model_insect.h5")

        # Load ảnh từ url
        img = Image.open(urlopen(filename))

        # Đổi cỡ ảnh
        img = img.resize((300, 300))

        # Chuyển ảnh sang mảng
        img = image.img_to_array(img, dtype=np.uint8)

        # Căn chỉnh ảnh
        img = np.array(img)/255.0

        # Dự đoán
        p = loaded_best_model.predict(img[np.newaxis, ...])

        # Tên côn trùng
        labels = {0: 'Bươm bướm', 1: 'Chuồn chuồn',
                  2: 'Châu chấu', 3: 'Bọ rùa', 4: 'Muỗi'}

        probality = np.max(p[0], axis=-1)
        name = labels[np.argmax(p[0], axis=-1)].upper()

        classes = []
        prob = []

        for i, j in enumerate(p[0], 0):
            classes.append(labels[i])
            prob.append(round(j*100, 2))

        return render_template('insect_recognition.html', image=filename, insectname=name, probality=probality, prediction=zip(classes, prob))
    else:
        flash('Định dạng ảnh hỗ trợ là png, jpg, jpeg, gif!')
        return render_template('insect_recognition.html')


@app.route('/flower_recognition', methods=['POST'])
def flower_recognition():
    if 'imgfile' not in request.files:
        flash('Không tìm thấy file!')
        return render_template('flower_recognition.html')

    imgfile = request.files['imgfile']

    if imgfile.filename == '':
        flash('Không có ảnh nào được tải lên!')
        return render_template('flower_recognition.html')

    if imgfile and allowed_file(imgfile.filename):
        # Upload ảnh lên cloudinary
        cloudinary.config(cloud_name = CLOUD_NAME, api_key=API_KEY, 
            api_secret=API_SECRET)
        upload_result = None

        upload_result = cloudinary.uploader.upload(imgfile)

        filename = upload_result['url']

        cnn = joblib.load('model_flower.pkl')

        # Tiền xử lý hình ảnh mới
        
        # Load ảnh từ url
        test_image = Image.open(urlopen(filename))

        # Đổi cỡ ảnh
        test_image = test_image.resize((64, 64))

        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image)
        # training_set.class_indices

        prediction = ""
        if result[0][0] == 1:
            prediction = 'hoa cúc'
        elif result[0][1] == 1:
            prediction = 'bồ công anh'
        elif result[0][2] == 1:
            prediction = 'hoa hồng'
        elif result[0][3] == 1:
            prediction = 'hoa hướng dương'
        elif result[0][4] == 1:
            prediction = 'hoa tu líp'

        return render_template('flower_recognition.html', prediction=prediction, pathOfImg=filename)
    else:
        flash('Định dạng ảnh hỗ trợ là png, jpg, jpeg, gif!')
        return render_template('flower_recognition.html')


@app.route('/sign_language_recogition', methods=['POST'])
def sign_language_recogition():
    if 'file' not in request.files:
        flash('Không tìm thấy file!')
        return render_template('sign_language_recognition.html')

    file = request.files['file']

    if file.filename == '':
        flash('Không có ảnh nào được tải lên!')
        return render_template('sign_language_recognition.html')

    if file and allowed_file(file.filename):
        # Upload ảnh lên cloudinary
        cloudinary.config(cloud_name = CLOUD_NAME, api_key=API_KEY, 
            api_secret=API_SECRET)
        upload_result = None

        upload_result = cloudinary.uploader.upload(file)

        filename = upload_result['url']

        loaded_best_model = keras.models.load_model("model_sign_language.h5")

        # Tên ký hiệu
        dict_labels = {0:"a",1:"b",2:"c",3:"d",4:"e",5:"f",6:"g",7:"h",8:"i",9:"j"
        ,10:"k",11:"l",12:"m",13:"n",14:"o",15:"p",16:"q",17:"r",18:"s",19:"t",20:"u",
        21:"unkowen",22:"v",23:"w",24:"x",25:"y",26:"z"}


        # Load ảnh từ url
        img = Image.open(urlopen(filename))

        # Đổi cỡ ảnh
        img = img.resize((50, 50))

        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Bảng dự đoán phần trăm
        p = loaded_best_model.predict(img)

        # Số tương ứng
        label = np.argmax(p[0],axis=-1)
        name = dict_labels[label].upper()
        # Mức độ trùng khớp
        probality = np.max(p[0],axis=-1)
        
        return render_template('sign_language_recognition.html', image=filename, signname=name, probality=probality)
    else:
        flash('Định dạng ảnh hỗ trợ là png, jpg, jpeg, gif!')
        return render_template('sign_language_recognition.html')

@app.route('/weather_recognition', methods=['POST'])
def weather_recognition():
    if 'imgfile' not in request.files:
        flash('Không tìm thấy file!')
        return render_template('weather_recognition.html')

    imgfile = request.files['imgfile']

    if imgfile.filename == '':
        flash('Không có ảnh nào được tải lên!')
        return render_template('weather_recognition.html')


    if imgfile and allowed_file(imgfile.filename):
        # Upload ảnh lên cloudinary
        cloudinary.config(cloud_name = CLOUD_NAME, api_key=API_KEY, 
            api_secret=API_SECRET)
        upload_result = None

        upload_result = cloudinary.uploader.upload(imgfile)

        filename = upload_result['url']

        cnn = joblib.load("model_weather.pkl")

        # Tiền xử lý hình ảnh mới
        # Load ảnh từ url
        test_image = Image.open(urlopen(filename))

        # Đổi cỡ ảnh
        test_image = test_image.resize((64, 64))

        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image)
        # training_set.class_indices

        prediction = ""
        if result[0][0] == 1:
            prediction = 'sương mù'
        elif result[0][1] == 1:
            prediction = 'mưa đá'
        elif result[0][2] == 1:
            prediction = 'sấm chớp'
        elif result[0][3] == 1:
            prediction = 'mưa'
        elif result[0][4] == 1:
            prediction = 'cầu vồng'
        elif result[0][5] == 1:
            prediction = 'tuyết'

        return render_template('weather_recognition.html', prediction=prediction, pathOfImg=filename)
    else:
        flash('Định dạng ảnh hỗ trợ là png, jpg, jpeg, gif!')
        return render_template('weather_recognition.html')

if __name__ == "__main__":
    port = os.environ.get('FLASK_PORT') or 8080
    port = int(port)

    app.run(port=port,host='0.0.0.0')

