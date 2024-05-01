import cv2
import numpy as np
import threading
import queue
import firebase_admin
from firebase_admin import credentials, db
from util import get_parking_spots_bboxes, empty_or_not

# Initialize Firebase
cred = credentials.Certificate('./parkiudlap-b955710ac96b.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://parkiudlap-default-rtdb.firebaseio.com/'
})

def async_update_firebase(spot_id, status):
    def run():
        db.reference(f'parking_spots/{spot_id}').set({'occupied': status})
    threading.Thread(target=run).start()
    
mask = './mask_crop.png'
video_path = './test/parking_crop_loop.mp4'
mask = cv2.imread(mask, 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

frame_queue = queue.Queue(maxsize=10)
stop_signal = threading.Event()

def frame_reader(frame_queue, video_path, stop_signal):
    cap = cv2.VideoCapture(video_path)
    while not stop_signal.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_signal.set()
            break
        frame_queue.put(frame)
    cap.release()

def frame_processor(frame_queue, spots, stop_signal):
    previous_frame = None
    step = 30
    frame_nmr = 0

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    while not stop_signal.is_set():
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        if frame is None:
            continue

        if frame_nmr % step == 0:
            if previous_frame is not None:
                frame = process_frame(frame, previous_frame, spots)
            previous_frame = frame.copy()
        else:
            if previous_frame is not None:
                frame = previous_frame.copy()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            stop_signal.set()
            break

        frame_nmr += 1

    cv2.destroyAllWindows()

def process_frame(frame, previous_frame, spots):
    processed_frame = frame.copy()
    if previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            async_update_firebase(spot_indx, spot_status)

            color = (0, 255, 0) if spot_status else (0, 0, 255)
            cv2.rectangle(processed_frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    return processed_frame

def main():
    reader_thread = threading.Thread(target=frame_reader, args=(frame_queue, video_path, stop_signal))
    processor_thread = threading.Thread(target=frame_processor, args=(frame_queue, spots, stop_signal))

    reader_thread.start()
    processor_thread.start()

    reader_thread.join()
    processor_thread.join()

if __name__ == "__main__":
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
