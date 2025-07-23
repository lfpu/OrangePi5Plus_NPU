import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
np.int = int  # 兼容旧代码

app = FaceAnalysis(name="buffalo_s", root="./", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def draw_on_origin(img, faces, det_size=(640, 640)):
    import cv2
    import numpy as np
    dimg = img.copy()
    h0, w0 = img.shape[:2]
    dw, dh = det_size
    for face in faces:
        # bbox按比例映射
        box = face.bbox
        box = box.astype(np.float32)
        box[0] = int(box[0] * w0 / dw)
        box[2] = int(box[2] * w0 / dw)
        box[1] = int(box[1] * h0 / dh)
        box[3] = int(box[3] * h0 / dh)
        box = box.astype(int)
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        # 关键点
        if face.kps is not None:
            kps = face.kps.astype(np.float32)
            kps[:, 0] = kps[:, 0] * w0 / dw
            kps[:, 1] = kps[:, 1] * h0 / dh
            kps = kps.astype(int)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 2, color, 20)
        # 性别年龄
        if hasattr(face, "sex") and hasattr(face, "age") and face.sex is not None and face.age is not None:
            cv2.putText(dimg, '%s,%d' % (face.sex, face.age), (box[0]-1, box[1]-4),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 1)
    return dimg

# 用法
img = cv2.imread("./src/1.jpg")
h0, w0 = img.shape[:2]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))
faces = app.get(img_resized)
rimg = draw_on_origin(img, faces, det_size=(640, 640))
cv2.imwrite("./src/t1_output.jpg", rimg)