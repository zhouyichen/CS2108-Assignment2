# __author__ = "xiangwang1223@gmail.com"
from common import *
import moviepy.editor as mp
import glob

def getKeyFrames(vidcap, store_frame_path):
    count = 0
    lastHist = None
    sumDiff = []
    frames = []

    while True:
        success, frame = vidcap.read()
        if not success:
            break
        frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        if count >0:
            diff = np.abs(hist - lastHist)
            s = np.sum(diff)
            sumDiff.append(s)
        lastHist = hist
        count += 1

    if len(frames) == 0:
        return

    m = np.mean(sumDiff)
    std = np.std(sumDiff)

    candidates = []
    candidates_value = []
    for i in range(len(sumDiff)):
        if sumDiff[i] > m + std*3:
            candidates.append(i + 1)
            candidates_value.append(sumDiff[i])

    if len(candidates) > 20:
        top10list = sorted(range(len(candidates_value)), key=lambda i: candidates_value[i])[-9:]
        res = []
        for i in top10list:
            res.append(candidates[i])
        candidates = sorted(res)

    candidates = [0] + candidates

    keyframes = []
    lastframe = -2
    for frame in candidates:
        if not frame == lastframe + 1:
            keyframes.append(frame)
        lastframe = frame

    count = 0
    for frame in keyframes:
        image = frames[frame]
        cv2.imwrite(store_frame_path+"frame%d.jpg" % count, image)
        count += 1

def getAudioClip(video_reading_path, audio_storing_path):
    clip = mp.VideoFileClip(video_reading_path)
    try:
        clip.audio.write_audiofile(audio_storing_path)
    except Exception:
        print Exception

def extract_for_folder(input_video_folder, output_frame_folder, output_audio_folder):
    for video_path in glob.glob(input_video_folder + "/*.mp4"):
        
        video_id = video_path[video_path.rfind("/") + 1:-4]
        # if int(video_id) < 1001088152326610944:
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

    