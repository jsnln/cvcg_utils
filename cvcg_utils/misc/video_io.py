def write_video(video_fn, image_seq, fps=30):
    import numpy as np
    import cv2
    from tqdm import trange

    h, w = image_seq[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_fn, fourcc, fps, (w,h))

    for frame_id in trange(len(image_seq)):
        out.write(image_seq[frame_id])
