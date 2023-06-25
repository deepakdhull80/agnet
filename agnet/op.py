import yaml
import os
import cv2 as cv
import argparse
from predict import get_predictor, Predictor

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",'-c', dest="config_file", required=True)
    parser.add_argument("--device",'-d',dest="device",default="cpu",choices=['cpu','cuda'])
    parser.add_argument("--type",'-t', dest="action_type")
    parser.add_argument("--file-path",'-f', dest="file_path")
    parser.add_argument("--write-path",'-o',dest="write_path", default="infer/output")
    parser.add_argument("--fps", dest="fps", default=10)
    return parser.parse_args()

def cvt_color(image):
    return cv.cvtColor(image, cv.COLOR_RGB2BGR)

def video_inference(predictor: Predictor, video_path: str, write_path:str, config, frame_rate: int = 32, display_video=True) -> str:
    file_name = video_path.rsplit("/",1)[-1]
    output_file = f"{write_path}/{file_name}"
    cap = cv.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv.CAP_PROP_FPS)

    # Create a VideoWriter object to write the output video
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_file, fourcc, fps, (config['data']['IMAGE_SIZE'], config['data']['IMAGE_SIZE']))
    frame_delay = int(round(1000 / frame_rate))
    print("Start")
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        frame = cvt_color(frame)
        frame = cv.flip(frame,0)
        try:
            frame = predictor.infer(frame)
        except:
            continue
        out.write(frame)

        # Display the frame (optional)
        if display_video:
            cv.imshow('Frame', frame)
        if cv.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    # Release the resources
    cap.release()
    out.release()
    cv.destroyAllWindows()
    return output_file

def image_inference(predictor: Predictor, image_path: str, write_path: str) -> str:
    file_name = image_path.rsplit("/",1)[-1]
    predictor.predict_and_write(image_path, f"{write_path}/{file_name}")

def camera_inference(predictor: Predictor, device:int = 0, frame_rate=10, **kwargs):
    cap = cv.VideoCapture(device)
    frame_delay = int(round(1000 / frame_rate))
    print("Start")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cvt_color(frame)
        try:
            frame = predictor.infer(frame)
        except:
            continue
        cv.imshow('Frame', frame)
        
        if cv.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    # Release the resources
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':

    args = argparser()
    config = yaml.load(open(args.config_file,'r'))

    predictor = get_predictor(args)
    os.makedirs(args.write_path,exist_ok=True)
    # predictor.predict_byarray(npimage)
    if args.action_type == 'video':
        pth = video_inference(
            predictor,
            args.file_path,
            args.write_path,
            config,
            frame_rate= int(args.fps),
            display_video=True
        )
        print(f'Video inference completed,: {pth}')
    
    elif args.action_type == 'image':
        pth = image_inference(
            predictor,
            args.file_path,
            args.write_path,
        )
        print(f'Image inference completed,: {pth}')
    
    elif args.action_type == "camera":
        camera_device = 0
        camera_inference(
            predictor,
            camera_device,
            frame_rate=int(args.fps),
        )
    else:
        raise ValueError(
            f"action_type should be present in [video, image, camera]"
        )