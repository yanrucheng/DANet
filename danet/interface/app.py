from flask import Flask
import os

UPLOAD_FOLDER = os.path.abspath( os.path.join(os.path.dirname(__file__), '..', 'datasets', 'cityscapes'))

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
