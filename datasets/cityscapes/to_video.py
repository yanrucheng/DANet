import cv2
import os, argparse

def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("--path", '-p', help="path to images folder")
    a.add_argument("--fps", '-f', type=int, default=30, help="path to images folder")
    return a.parse_args()

def frames_to_image(image_folder, fps):
    video_name = '{}.mp4'.format(os.path.split(image_folder)[1])

    #images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = os.listdir(image_folder)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))
    print('Writing to {}'.format(video_name))

    for image in sorted(images):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    #cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    args = get_args()
    frames_to_image(args.path, args.fps)

