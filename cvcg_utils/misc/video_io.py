def write_video(video_fn, image_seq, fps=30):
    import numpy as np
    import cv2
    from tqdm import trange

    h, w = image_seq[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_fn, fourcc, fps, (w,h))

    for frame_id in trange(len(image_seq)):
        out.write(cv2.cvtColor(image_seq[frame_id], cv2.COLOR_RGB2BGR))

def read_video_rgb(video_fn):
    import numpy as np
    import cv2
    from tqdm import trange
        
    cap = cv2.VideoCapture(video_fn)

    frame_list = []
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_id in trange(num_frames):
        flag, frame_bgr = cap.read()
        assert flag
        assert frame_bgr.shape[-1] == 3

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_list.append(frame_rgb)

    return frame_list