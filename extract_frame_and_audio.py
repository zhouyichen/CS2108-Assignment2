# __author__ = "xiangwang1223@gmail.com"
from common import *
import moviepy.editor as mp
import glob
from searchers.visual_searcher import getKeyFrames

def getAudioClip(video_reading_path, audio_storing_path):
    clip = mp.VideoFileClip(video_reading_path)
    try:
        clip.audio.write_audiofile(audio_storing_path)
    except Exception:
        print Exception

def extract_for_folder(input_video_folder, output_frame_folder, output_audio_folder):
    for video_path in glob.glob(input_video_folder + "/*.mp4"):
        
        video_id = video_path[video_path.rfind("/") + 1:-4]
        # if int(video_id) != 1001088152326610944:
        #     continue
        # if int(video_id) != 1001949402207817728:
        #     continue
        print video_id
        vidcap = cv2.VideoCapture(video_path)

        frame_out_path = output_frame_folder + video_id + '-'
        getKeyFrames(vidcap=vidcap, store_frame_path=frame_out_path)
        vidcap.release()

        audio_out_path = output_audio_folder + video_id + '.wav'
        getAudioClip(video_path, audio_out_path)
        

if __name__ == '__main__':
    extract_for_folder(train_video_folder, train_frame_folder, train_audio_folder)
    extract_for_folder(test_video_folder, test_frame_folder, test_audio_folder)

    