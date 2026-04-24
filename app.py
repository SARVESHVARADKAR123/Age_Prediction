import cv2
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Configuration
img_height, img_width = 128, 128

# Load face cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ============================================================
# Helper: Load Keras 3 weights into a Keras 2 model
# ============================================================
def load_keras3_weights(model, weights_h5_path):
    """
    Keras 3 .weights.h5 stores weights under:
        layers/<layer_name>/vars/0, vars/1, ...
    This function reads those arrays and assigns them
    to the matching layers in a locally-built Keras 2 model.
    """
    with h5py.File(weights_h5_path, 'r') as f:
        layers_group = f['layers']

        for layer in model.layers:
            name = layer.name

            # For Sequential models, the base MobileNetV2 is a nested model
            if hasattr(layer, 'layers'):
                # This is a sub-model (e.g. MobileNetV2 inside Sequential)
                # Its weights are stored under 'functional/layers/<sublayer>/vars/...'
                if 'functional' in layers_group:
                    func_group = layers_group['functional']
                    if 'layers' in func_group:
                        sub_layers_group = func_group['layers']
                        for sub_layer in layer.layers:
                            sub_name = sub_layer.name
                            if sub_name in sub_layers_group:
                                sub_g = sub_layers_group[sub_name]
                                if 'vars' in sub_g:
                                    vars_group = sub_g['vars']
                                    weight_arrays = []
                                    for i in range(len(vars_group)):
                                        weight_arrays.append(np.array(vars_group[str(i)]))
                                    if weight_arrays and len(weight_arrays) == len(sub_layer.get_weights()):
                                        sub_layer.set_weights(weight_arrays)
                continue

            # Top-level layers (dense, dropout, etc.)
            if name in layers_group:
                g = layers_group[name]
                if 'vars' in g:
                    vars_group = g['vars']
                    weight_arrays = []
                    for i in range(len(vars_group)):
                        weight_arrays.append(np.array(vars_group[str(i)]))
                    if weight_arrays and len(weight_arrays) == len(layer.get_weights()):
                        layer.set_weights(weight_arrays)

    print("Weights assigned from Keras 3 format")


# ==================== AGE MODEL ====================
def build_age_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                             input_shape=(img_height, img_width, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model

print("Loading age model...")
age_model = build_age_model()
# Build it so weights are initialized
age_model.predict(np.zeros((1, img_height, img_width, 3)))
try:
    load_keras3_weights(age_model, 'age_model/model.weights.h5')
    print("Age model loaded successfully")
except Exception as e:
    print(f"Age model load failed: {e}")
    age_model = None

# ==================== GENDER MODEL (WORKING - DO NOT TOUCH) ====================
def build_gender_model():
    base_model_gender = MobileNetV2(weights='imagenet', include_top=False,
                                     input_shape=(img_height, img_width, 3))
    for layer in base_model_gender.layers:
        layer.trainable = False
    x = base_model_gender.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model_gender.input, outputs=predictions)
    return model

print("Loading gender model...")
gender_model = build_gender_model()
try:
    gender_model.load_weights('gender_model/gender_model.h5')
    print("Gender model loaded successfully")
except Exception as e:
    print(f"Gender model load failed: {e}")
    gender_model = None

# ==================== PREDICTION ====================
def predict_attributes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print("No faces detected.")

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (img_height, img_width))
        face_array = np.expand_dims(face_resized, axis=0)
        face_preprocessed = preprocess_input(face_array.astype(np.float32))

        # Age Prediction
        age_text = "N/A"
        if age_model:
            try:
                predicted_age = age_model.predict(face_preprocessed)
                age_text = str(int(round(predicted_age[0][0])))
            except Exception as pe:
                print(f"Age prediction error: {pe}")

        # Gender Prediction
        gender_text = "N/A"
        if gender_model:
            predicted_gender_val = gender_model.predict(face_preprocessed)
            gender_text = "Male" if predicted_gender_val[0][0] < 0.5 else "Female"

        print(f"Face detected - Age: {age_text}, Gender: {gender_text}")

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        label = f'Age: {age_text}, {gender_text}'
        cv2.putText(img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

    try:
        cv2.imshow('Age & Gender Prediction', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as display_error:
        output_path = image_path.replace('.jpg', '_output.jpg')
        cv2.imwrite(output_path, img)
        print(f"Image saved to {output_path}")

if __name__ == "__main__":
    predict_attributes('photo.jpg')