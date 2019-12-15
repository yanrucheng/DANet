import os
#import magic
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('file-upload.html')

@app.route('/python-flask-files-upload', methods=['POST'])
def upload_file():
	# check if the post request has the file part
	if 'files[]' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	
	files = request.files.getlist('files[]')
	
	errors = {}
	success = False
	
	for file in files:
		if file and allowed_file(file.filename):
			import uuid
			uid = str(uuid.uuid1())
			filename = uid + secure_filename(file.filename)
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'upload', filename)
			file.save(filepath)
			with open(os.path.join(app.config['UPLOAD_FOLDER'], 'test.txt'), "w") as f:
                            f.write('{}\t{}'.format(filepath, filepath))

			os.system('./visualize.sh')
			success = True
			from flask import send_file
			return send_file(os.path.splitext('cityscapes/danet_vis/'+filename)[0]+'.png', as_attachment=True)
		else:
			errors[file.filename] = 'File type is not allowed'
	
	if success and errors:

		errors['message'] = 'File(s) successfully uploaded'
		resp = jsonify(errors)
		resp.status_code = 206
		return resp
	if success:
		resp = jsonify({'message' : 'Files successfully uploaded'})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify(errors)
		resp.status_code = 400
		return resp

if __name__ == "__main__":
    app.run('0.0.0.0', port='9982')
