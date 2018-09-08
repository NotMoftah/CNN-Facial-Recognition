import cv2
import glob
import numpy as np

# Array of the full faces paths
save_to = 'C:\\Users\\AbdEl-Aziz\\Downloads\\MIT_Cropped_Faces\\'
all_faces = [img for img in glob.glob('C:\\Users\\AbdEl-Aziz\\Downloads\\gt_db\\s*\\*.jpg')]

faces_x = []
faces_y = []

faceCascade = cv2.CascadeClassifier('data\\haarcascade_frontalface.xml')

for i, face in enumerate(all_faces):
    image = cv2.imread(face)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    # check if the face exist
    if len(faces) == 1:
        # crop the face
        x, y, w, h = faces[0]
        cropped_img = image[y:y + h, x:x + w]

        # resize the face to 128 x 128 (MobileNet Input size)
        faces_x.append(cv2.resize(cropped_img, (128, 128)))
        faces_y.append(int(face.split('\\')[-2][1:]))

    print('Finished: ', i, ' Out of: ', len(all_faces))


faces_x, faces_y = np.array(faces_x), np.array(faces_y)

np.save(save_to + 'x_train', faces_x)
np.save(save_to + 'y_train', faces_y)
