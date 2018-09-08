import tensorflow as tf
import numpy as np
import cv2


class FaceID:
    def __init__(self):
        model = tf.keras.Sequential()
        net = tf.keras.applications.MobileNet(input_shape=(128, 128, 3), weights='imagenet', include_top=False)
        model.add(net)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        self.features_extractor = model

        self.x_holder = tf.placeholder(shape=[None, 1024], dtype=tf.float32)
        fc_1 = tf.layers.Dense(units=512, activation=tf.nn.relu)(self.x_holder)
        fc_2 = tf.layers.Dense(units=128, activation=tf.nn.sigmoid)(fc_1)

        self.face_id = fc_2

        self.sess = None

    def load_network(self, path='data\\model'):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, path)

    def get_id(self, imgs):
        imgs = imgs.reshape((-1, 128, 128, 3))
        features = self.features_extractor.predict(imgs)
        embeds = self.sess.run([self.face_id], feed_dict={self.x_holder: features})

        return embeds[0]


class FaceExtractor:
    def __init__(self, cascade_path='data\\haarcascade_frontalface.xml'):
        self.faceCascade = cv2.CascadeClassifier(cascade_path)

    def extract_single_face_from_path(self, img_path):
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            cropped_img = image[y:y + h, x:x + w]
            return cv2.resize(cropped_img, (128, 128))
        else:
            faces = self.faceCascade.detectMultiScale(gray, 1.3, 10)
            if len(faces) == 1:
                x, y, w, h = faces[0]
                cropped_img = image[y:y + h, x:x + w]
                return cv2.resize(cropped_img, (128, 128))

        return None

    def faces_from_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.faceCascade.detectMultiScale(gray, 1.3, 5)


test_nn = FaceID()
face_ex = FaceExtractor()

test_nn.load_network()

ref_face = face_ex.extract_single_face_from_path("ref.jpg")

ref_face_hash = test_nn.get_id(ref_face)[0]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = face_ex.faces_from_image(frame)

    for face in faces:
        x, y, w, h = face
        cropped_face = cv2.resize(frame[y:y + h, x:x + w], (128, 128))
        cropped_hash = test_nn.get_id(cropped_face)[0]

        cv2.rectangle(frame, (x, y), (x + w, y + h), 1, 3)

        distance_1 = np.sum(np.power(ref_face_hash - cropped_hash, 2))

        if distance_1 <= 3:
            cv2.putText(frame, 'ref ', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Nan ', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 2, cv2.LINE_AA)

    cv2.imshow('My FaceID', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        ret, frame = cap.read()
        break
