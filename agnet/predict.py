from typing import *
import os
import argparse
import yaml
from PIL import Image
import numpy as np
import cv2 as cv
import torch
import torchvision
from facenet_pytorch import MTCNN
from tqdm import tqdm

try:
    from model import AGNet
except:
    from .model import AGNet

import warnings
warnings.filterwarnings('ignore')

class Predictor:
    def __init__(
            self, 
            facenet, 
            gender_model=None, 
            age_model=None, 
            device=torch.device("cpu"),
            image_size = 456,
            face_image_size = 720,
            **kwargs
        ) -> None:
        """_summary_

        Args:
            facenet (_type_): facenet model on same device.
            gender_model (_type_, optional): _description_. Defaults to None.
            age_model (_type_, optional): _description_. Defaults to None.
            device (_type_, optional): _description_. Defaults to torch.device("cpu").
            image_size (int, optional): _description_. Defaults to 456.
        """
        self.device = device
        self.image_size = image_size
        self.face_image_size = face_image_size
        self.facenet = facenet
        self.face_present_threshold = kwargs.get("FACE_PRESENT_THRESHOLD",0.2)
        #### gender config
        if gender_model is not None:
            self.gender_model = gender_model.to(device)
            self.gender_model.eval()
        self._gender_predict_threshold = kwargs.get('GENDER_PREDICT_THRESHOLD',0.5)
        self._gender_face_margin = kwargs.get('GENDER_FACE_MARGIN',10)
        if age_model is not None:
            self.age_model = age_model.to(device)
            self.age_model.eval()
        self._age_predict_threshold = kwargs.get('AGE_PREDICT_THRESHOLD',0.5)
        self._age_face_margin = kwargs.get('AGE_FACE_MARGIN',10)
        self.age_mae = kwargs.get("age_mae",0)
        self.image2tensor = torchvision.transforms.PILToTensor()
        self.transforms1 = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
            # torchvision.transforms.Resize(self.image_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            # torchvision.transforms.Resize(self.image_size)
        ])

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            ### PIL image to tensor
            image = image.resize((self.face_image_size, self.face_image_size))
            # image = self.image2tensor(image)
            image = self.transforms1(image)
        elif isinstance(image, np.ndarray):
            ### convert numpy to tensor
            pass
        else:
            raise TypeError(f'image type {type(image)}, not in require list[np.ndarray, PIL.Image]')

        if image.shape[0] == 1:
            image = torch.concat([image, image, image], axis=0)
        image = image/255.
        return image

    def detect_face(self, image: torch.Tensor):
        boxes, logits = self.facenet.detect(image)
        return boxes, logits
    
    @torch.no_grad()
    def predict_age(self, x:torch.Tensor):
        x = self.age_model(x)
        return x

    @torch.no_grad()
    def predict_gender(self, x: torch.Tensor):
        x = self.gender_model(x)
        return torch.sigmoid(x)
    
    def extract_face(self, image, label, save_base_path, counter):
        boxes, logits = self.detect_face(image)

        boxes = boxes[logits>=self.face_present_threshold]
        logits = logits[logits>=self.face_present_threshold]
        
        # margin = margin if margin is not None else self._gender_face_margin
        margin = 30
        path = f"{save_base_path}/{label}"
        if not os.path.exists(path):
            os.makedirs(path,exist_ok=True)
        
        for face, prob in zip(boxes, logits):
            face = face+[-margin,-margin,margin,margin]
            face_image = image.crop(face)
            prob = int(prob*100)
            face_image.save(f"{path}/face_image_{counter}_{prob}.jpg")

    @torch.no_grad()
    def predict(self, image: Image.Image, margin = None):
        boxes, logits = self.detect_face(image)
        if logits is None:
            return None
        boxes = boxes[logits>self.face_present_threshold]
        logits = logits[logits>self.face_present_threshold]
        result = {
            "image": image,
            "total_face_present": len(boxes),
        }
        predict = []
        margin = margin if margin is not None else self._gender_face_margin
        for face, prob in zip(boxes, logits):
            face = face+[-margin,-margin,margin,margin]
            face_image = image.crop(face)

            input_image = self.preprocess(face_image)
            face_predict = {
                'box': face,
                'face_logits':prob
            }
            if hasattr(self, 'gender_model'):
                gender_predict = self.predict_gender(input_image.unsqueeze(0).to(self.device)).cpu()
                
                if gender_predict[0]>=self._gender_predict_threshold:
                    gender = 'female'
                    score = gender_predict[0].item()
                else:
                    gender = 'male'
                    score = 1 - gender_predict[0].item()
                face_predict['gender'] = gender
                face_predict['gender_score'] = score

            if hasattr(self, 'age_model'):
                age_predict = self.predict_age(input_image.unsqueeze(0).to(self.device)).cpu()
                age = int(age_predict[0][0])
                face_predict['age'] = age
                
            # append face related predictions
            predict.append(face_predict)
        result['predict'] = predict
        return result

    def predict_byfile(self, image_path: str):
        """_summary_

        Args:
            image_path (str): file path

        Raises:
            FileNotFoundError: _description_

        Returns:
            dict: {
                image_path:
                image: Image.Image resized image
                total_face_present:
                predict: List[
                    {
                        box: face coordinates
                        face_logits:
                        gender: Optional
                        gender_score: Optional
                        age: Optional
                    }
                ]

            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found {image_path}")
        
        image = Image.open(image_path)
        # image = image.resize((self.image_size, self.image_size))

        res = self.predict(image)
        if res is None:
            return None
        res['file_path'] = image_path
        return res
    
    def predict_byarray(self, image: np.ndarray):
        """ predict by numpy array

        Args:
            image (np.ndarray): dtype should be int and color code should be BGR
                bcz model trained upon BGR images. 

        Returns:
            dict: {
                image: Image.Image resized image
                total_face_present:
                predict: List[
                    {
                        box: face coordinates
                        face_logits:
                        gender: Optional
                        gender_score: Optional
                        age: Optional
                    }
                ]

            }
        """
        image = Image.fromarray(image)
        image = image.resize((self.image_size,self.image_size))
        return self.predict(image)
    
    def predict_and_write(self, image_path, image_write_path, color=(36,255,12), border_thickness=1, query={'gender':['male','female'],'age':(1,100)}):
        res = self.predict_byfile(image_path)
        if res is None:
            raise Exception(
                'Face not found in frame'
            )
        image = cv.cvtColor(np.array(res['image']), cv.COLOR_BGR2RGB)
        
        for face_predict in res['predict']:
            if face_predict['face_logits']<self.face_present_threshold:
                continue
            box = np.int32(face_predict['box'])
            x0, y0, x1, y1 = map(int, map(round, box))
            strt_point = (x0, y0)
            end_point = (x1, y1)
            
            ### query
            draw_rec_enable = False
            top_pad = 10
            
            if 'age' in face_predict and 'age' in query and face_predict['age'] in range(*query['age']):
                
                draw_rec_enable = True
                # write age over the box
                txt_strt_point = (x0,y0-top_pad)
                top_pad += 15
                image = cv.putText(image, f"age:{face_predict['age']: .2f} +|- {self.age_mae}", 
                                   txt_strt_point, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                                   )

            if 'gender' in face_predict:
                if 'gender' not in query or face_predict['gender'] not in query['gender']:
                    continue
                draw_rec_enable = True
                # write gender over the box
                txt_strt_point = (x0,y0-top_pad)
                top_pad += 15
                image = cv.putText(image, f"{face_predict['gender']}, prob: {face_predict['gender_score']:0.2f}", 
                                   txt_strt_point, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                                   )
            

            if draw_rec_enable:
                image = cv.rectangle(image, strt_point, end_point, color, border_thickness)
        
        cv.imwrite(image_write_path,image)
        print(f"Image saved at {image_write_path} and total number of faces found: {res['total_face_present']}")
        return image_write_path
    
    def infer(self, image, color=(36,255,12), border_thickness=1):
        res = self.predict_byarray(image)
        if res is None:
            return None
        image = cv.cvtColor(np.array(res['image']), cv.COLOR_BGR2RGB)
        
        for face_predict in res['predict']:
            if face_predict['face_logits']<self.face_present_threshold:
                continue
            box = np.int32(face_predict['box'])
            x0, y0, x1, y1 = map(int, map(round, box))
            strt_point = (x0, y0)
            end_point = (x1, y1)
            
            ### query
            draw_rec_enable = False
            top_pad = 10
            
            if 'age' in face_predict:
                draw_rec_enable = True
                # write age over the box
                txt_strt_point = (x0,y0-top_pad)
                top_pad += 15
                image = cv.putText(image, f"age:{face_predict['age']: .2f} +|- {self.age_mae}", 
                                   txt_strt_point, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                                   )

            if 'gender' in face_predict:
                draw_rec_enable = True
                # write gender over the box
                txt_strt_point = (x0,y0-top_pad)
                top_pad += 15
                image = cv.putText(image, f"{face_predict['gender']}, prob: {face_predict['gender_score']:0.2f}", 
                                   txt_strt_point, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                                   )
            

            if draw_rec_enable:
                image = cv.rectangle(image, strt_point, end_point, color, border_thickness)
        return image

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",'-c', dest="config_file", required=True)
    parser.add_argument("--model-version",'-vv', dest="model_version", default=1)
    parser.add_argument("--file-path",'-fp',dest='file_path', default=None)
    parser.add_argument("--image_base_path",'-ip', dest="image_base_path", default=None)
    parser.add_argument("--device",'-d',dest="device",default="cpu",choices=['cpu','cuda'])
    return parser.parse_args()

def load_weights(model: torch.nn.Module, ck_path):
    params = torch.load(ck_path)
    print(model.load_state_dict(params['state_dict']))
    return model

def get_predictor(args):
    config = yaml.load(open(args.config_file,'r'))
    FACE_PRESENT_THRESHOLD = config['model'].get("FACE_PRESENT_THRESHOLD",0.7)
    device = torch.device(args.device)
    mtcnn = MTCNN(image_size=config['data']['IMAGE_SIZE'], device=device)
    
    GENDER_PREDICT_THRESHOLD = 0.5
    GENDER_FACE_MARGIN = config['model']['GENDER_FACE_MARGIN']
    gender_base_model = getattr(torchvision.models, config['model']['gender_base_model'])()
    gender_dict = dict(
        _base_model=config['model']['gender_base_model'],
        output_dim=1,
        mlp_layer_name="",
        transfer_learning=False,
        estimator='gender'
    )
    gender_model = AGNet(gender_base_model,  **gender_dict)
    gender_model = load_weights(gender_model, config['model']['gender_model_path'])

    AGE_FACE_MARGIN = config['model']['AGE_FACE_MARGIN']
    
    age_base_model = getattr(torchvision.models, config['model']['age_base_model'])()
    age_dict = dict(
        _base_model=config['model']['age_base_model'],
        output_dim=1,
        mlp_layer_name="",
        transfer_learning=False,
        estimator='age'
    )
    age_model = AGNet(age_base_model,  **age_dict)
    age_model = load_weights(age_model, config['model']['age_model_path'])
    
    return Predictor(
        mtcnn, 
        gender_model=gender_model,
        age_model=age_model,
        device=device,
        image_size=config['data']['IMAGE_SIZE'],
        face_image_size=config['data']['FACE_IMAGE_SIZE'],
        FACE_PRESENT_THRESHOLD=FACE_PRESENT_THRESHOLD,
        GENDER_PREDICT_THRESHOLD=GENDER_PREDICT_THRESHOLD,
        GENDER_FACE_MARGIN=GENDER_FACE_MARGIN,
        AGE_FACE_MARGIN=AGE_FACE_MARGIN,
        age_mae=config['model'].get('age_mae',0)
    )

if __name__ == '__main__':
    args = argparser()
    predictor = get_predictor(args)

    image_path = "D:\WORK/freelance/agnet/test\images/samples"
    file_names = os.listdir(image_path)
    print(file_names)
    # image_path = image_path.replace("\\",'/')
    # image_path = "dataset/utkface/part3/21_0_3_20170119154213179.jpg"
    test_save_path = "D:\WORK/freelance/agnet/test\images\predict"
    
    ## test by file
    # res = predictor.predict_byfile(os.path.join(image_path,file_names[1]))
    # print(res)
    # exit()

    # test by numpy array
    # image = cv.imread(image_path)
    # image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    # # print(image)
    # res = predictor.predict_byarray(image)
    # print(res)

    ## test predict and save image
    for file in file_names[1:]:
        
        predictor.predict_and_write(
            os.path.join(image_path,file),
            os.path.join(test_save_path,file),
            query = {
                'gender': ['male','female'],
                'age': 100
            }
        )
        print(file, "completed")

    exit()

    # from glob import glob
    # import pandas as pd
    # paths = glob("D:\WORK/freelance/agnet/dataset/utkface/*/*.jpg")
    # save_path = "D:\WORK/freelance/agnet\dataset/age-dataset"
    # for i, p in tqdm(enumerate(paths), total=len(paths)):
    #     try:
    #         image = Image.open(p)
    #         label = int(p.split("/")[-1].split("\\")[-1].split("_")[0])
    #         predictor.extract_face(image,label,save_path,i)
    #     except:
    #         pass
    # exit()

    df = pd.DataFrame()
    df['file_paths'] = paths
    df['file_paths'] = df['file_paths'].map(lambda x: x.replace("\\",'/'))
    df['gender'] = df['file_paths'].map(lambda x: int(x.split("/")[-1].split('_')[1]))
    df['age'] = df['file_paths'].map(lambda x: int(x.split("/")[-1].split('_')[0]))
    gender_predict = []
    age_predict = []
    for file in tqdm(paths, total=len(paths)):
        try:
            res = predictor.predict_byfile(file)
            if len(res['predict'])==0:
                gender_predict.append(None)
                continue
            p = "|".join(map(lambda x: '0' if x['gender'] == 'male' else '1',res['predict']))
            gender_predict.append(p)
        except Exception as e:
            print(e)
            gender_predict.append(None)

    df['p_gender'] = gender_predict
    df.to_csv("test/eval/eval_utkface.csv",index=False)