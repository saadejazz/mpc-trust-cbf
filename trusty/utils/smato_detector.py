import numpy as np
import keras.backend as K


def get_cropped_images(bboxes, pil_im, return_shape = (224, 224)):
    crops = []
    padding = 0.1

    for bbox in bboxes:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        l_x = bbox[0] - padding * w
        l_y = bbox[1] - padding * h
        r_x = bbox[2] + padding * w
        r_y = bbox[3] + padding * h

        if l_x < 0: l_x = 0
        if l_y < 0: l_y = 0
        if r_x > pil_im.width: r_x = pil_im.width
        if r_y > pil_im.height: r_y = pil_im.height
            
        cropped = pil_im.crop((l_x, l_y, r_x, r_y))
        cropped = cropped.resize(return_shape)
        crops.append(cropped)
    
    return crops

def predict_smato(cropped_images, model):
    '''Returns True (has smartphone) or False based on the model
    '''
    image_array = []
    num_images = len(cropped_images)
    if num_images == 0: return []
    for image in cropped_images:
        image_array.append(np.asarray(image))
    image_array = np.array(image_array) * 1.0/255.0 
    im = image_array.reshape(num_images, 224, 224, 3)
    predictions = [a[0] for a in model.predict(im)]
    return predictions

def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall)/(precision + recall + K.epsilon())
    return f1_val
