from flask import Flask, render_template, request, flash, redirect
import os
import re

# 安装flask
# pip install flask==3.1.1
from main import *

# 创建app对象
app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'docx', 'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# ------------------------------------------------------------------------- #
# 集合名称
collection_name = 'demo'
name_list = os.listdir(UPLOAD_FOLDER)
if name_list:
    collection_name = name_list[0]
# ------------------------------------------------------------------------- #



# ------------------------------------------------------------------------- #
# -----------------------------   1. 上传文档  ----------------------------- #
# ------------------------------------------------------------------------- #
# 上传文档界面
@app.route('/document_upload/', methods=['GET', 'POST'])
def document_upload():
    # 进入页面
    if request.method == 'GET':
        return render_template('document_upload.html')
    # 接收提交上来的文档
    elif request.method == 'POST':
        # 检查是否有文件被上传
        if 'file' not in request.files:
            flash('没有选择文件')
            return redirect(request.url)

        file = request.files['file']

        # 如果用户没有选择文件，浏览器可能会提交一个空文件
        if file.filename == '':
            flash('没有选择文件')
            return redirect(request.url)

        # 检查文件类型是否允许
        if file and allowed_file(file.filename):
            # 存储文件到uploads文件夹
            filename = file.filename
            print("filename:", filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("file_path:", file_path)
            file.save(file_path)

            # 上传成功后，将文档存入向量数据库
            global collection_name  # 更改全局变量collection_name
            collection_name = re.split(r'[/\\]', filename)[-1]
            save_to_db(file_path, collection_name=collection_name)

        return redirect(request.url)

    else:
        return render_template('document_upload.html')


# ------------------------------------------------------------------------- #
# ------------------------------   2. 聊天    ------------------------------ #
# ------------------------------------------------------------------------- #

# 聊天界面
@app.route('/')
@app.route('/chat/', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return render_template('chat.html')
    elif request.method == 'POST':
        message = request.json.get('message')
        print("message:", message)
        # 检索
        if message:
            # 查对应集合的数据
            response = rag_chat(message, collection_name=collection_name, n_results=10)
            print('response:', response)

            return response
        else:
            return "不知道"

    return redirect(request.url)

# 聊天界面：文档名称切换
@app.route('/collection/', methods=['GET', 'POST'])
def collection():
    # 使用全局变量 collection_name
    global collection_name

    if request.method == 'GET':
        # 进入页面，默认显示所有文档
        name_list = os.listdir(UPLOAD_FOLDER)
        if name_list:
            return {'name_list': name_list, 'collection_name': collection_name}
        return {'name_list': [], 'collection_name': collection_name}

    elif request.method == 'POST':
        # 前端改变文档，后端修改文档名称，方便后面针对不同文档进行检索
        collection_name = request.json.get('collection_name')
        print('文档切换成了：', collection_name)
        # return {'status': 200, 'message': 'ok'}
        return redirect('/chat/')

    return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)


