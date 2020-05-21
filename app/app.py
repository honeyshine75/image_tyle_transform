import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow import keras
import g_model
from flask import Flask, send_file
from flask_restful import Resource, Api, reqparse
from werkzeug import secure_filename
from werkzeug.datastructures import FileStorage
import logging
import time
import logging.handlers


# 初始化logging
logger = logging.getLogger(__name__)
handler = logging.handlers.RotatingFileHandler('../mount/log/app.log',
                                                maxBytes=5*1024*1024,
                                                backupCount=5,
                                                encoding='utf-8')
logger.setLevel(logging.DEBUG)
# 定义格式器,添加到处理器中
fmt = '%(asctime)s , %(levelname)s , %(filename)s %(funcName)s line %(lineno)s , %(message)s'
datefmt = '%Y-%m-%d %H:%M:%S %a'
log_fmt = logging.Formatter(fmt=fmt, datefmt=datefmt)
handler.setFormatter(log_fmt)
logger.addHandler(handler)

# 所有文件的路径
model_weights_path = "../static/model/g_model.h5"
UPLOAD_FOLDER = '../mount/uploads'

app = Flask(__name__)
api = Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# 定义允许的参数为task，类型为int，以及错误时的提示
parser = reqparse.RequestParser()
parser.add_argument('contentUrl', type=str, required=True, help='Please set a url task content!')
parser.add_argument('styleUrl', type=str, required=True, help='Please set a url task content!')

# 运行上传的文件类型
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# 导入模型
model = g_model.get_Net()
model.load_weights(model_weights_path)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 真正处理请求的地方
class StyleImage(Resource):
    def get(self):
        try:
            args = parser.parse_args()
            contentUrl = args['contentUrl']
            styleUrl = args['styleUrl']
            assert contentUrl.lower().endswith('jpg') or contentUrl.lower().endswith('png') or contentUrl.lower().endswith('jpeg')
            assert styleUrl.lower().endswith('jpg') or styleUrl.lower().endswith('png') or styleUrl.lower().endswith('jpeg')
            content_path = os.path.abspath(
                            os.path.join(
                                os.path.dirname(__file__), app.config['UPLOAD_FOLDER'], contentUrl))
            style_path = os.path.abspath(
                            os.path.join(
                                os.path.dirname(__file__), app.config['UPLOAD_FOLDER'], styleUrl))
            content, style = g_model.precession(content_path, style_path)
            output_image = model([content, style], training=False)[0]
            # 保存生成图片
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], contentUrl+styleUrl+'.jpg')
            keras.preprocessing.image.save_img(save_path, output_image)

            return send_file(save_path, mimetype="image/jpeg")
            # return resp, 200
        except:
            logger.exception('Exception occur!')
            return 'error', 400

    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('content', type=FileStorage, location='files')
            parser.add_argument('style', type=FileStorage, location='files')
            args = parser.parse_args()
            content = args['content']
            style = args['style']
            if content and style and allowed_file(content.filename.lower()) and allowed_file(style.filename.lower()):
                filename = secure_filename(content.filename)
                filename = filename.rsplit('.')[0] + str(time.time()) + '.' + filename.rsplit('.')[-1]
                content.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                stylename = secure_filename(style.filename)
                stylename = stylename.rsplit('.')[0] + str(time.time()) + '.' + stylename.rsplit('.')[-1]
                style.save(os.path.join(app.config['UPLOAD_FOLDER'], stylename))
                return [filename, stylename], 200
            else:
                logger.error('Not allowed file or not safe!')
                return {'msg':'not allow'}, 400
        except:
            logger.exception('Exception occur!')
            return '''
                <!doctype html>
                <title>Upload new File</title>
                <h1>Upload new File</h1>
                <form action="" method=post enctype=multipart/form-data>
                  <p><input type=file name=file>
                     <input type=submit value=Upload>
                </form>
                ''', 400

# 实际定义路由的地方
api.add_resource(StyleImage, '/get', '/post')

if __name__ == '__main__':
     app.run(host='0.0.0.0')