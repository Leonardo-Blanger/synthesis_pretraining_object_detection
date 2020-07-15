import numpy as np
import pandas as pd
from scipy.io import loadmat

FILES = ['cars_train_annos.mat', 'cars_test_annos.mat']

for f in FILES:
    data = loadmat(f)['annotations'][0]

    save_data = {
        'image_file': [],
        'xmin': [],
        'ymin': [],
        'xmax': [],
        'ymax': [],
    }

    for box in data:
        x1, y1, x2, y2 = [box[i] for i in range(4)]
        img_file = box[-1][0]
        
        box = np.array([x1, y1, x2, y2]).reshape(4,)
        x1, y1, x2, y2 = box

        save_data['image_file'].append(img_file)
        save_data['xmin'].append(x1)
        save_data['ymin'].append(y1)
        save_data['xmax'].append(x2)
        save_data['ymax'].append(y2)

    save_data = pd.DataFrame(save_data)

    saved_file = f.split('.')[0] + '.csv'
    save_data.to_csv(saved_file, index=False)
