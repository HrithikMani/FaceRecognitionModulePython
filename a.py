import face_recognition
image_of_em = face_recognition.load_image_file('img/known/emilia.jpg')
em_face_encoding = face_recognition.face_encodings(image_of_em)[0]

unknown_load = face_recognition.load_image_file('img/unknown/download (3).jpg')
un_encoding = face_recognition.face_encodings(unknown_load)[0]

results = face_recognition.compare_faces([em_face_encoding],un_encoding)
print(results)
print(un_encoding)