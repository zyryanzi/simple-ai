from flask import Flask, render_template, Response, request

from camera import VideoCamera

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/dataset', methods=['GET'])
def dataset():
    face_id = request.args.get("face_id")
    camera = VideoCamera()
    camera.data_set(face_id)
    msg = "%s%s" %(face_id, "采集完成")
    content = {"msg": msg}
    return render_template('index.html')


@app.route('/recognition', methods=['GET'])
def recognition():
    camera = VideoCamera()
    face_name = camera.recognition()
    msg = "认证结果为:%s" %(face_name)
    content = {"msg": msg}
    return render_template('index.html', **content)


if __name__ == '__main__':
    app.run(host='23.105.198.209', port=50501, debug=False)
