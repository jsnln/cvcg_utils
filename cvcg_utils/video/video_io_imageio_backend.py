import numpy as np
import imageio
from tqdm import trange


def write_video(video_fn, image_seq, fps=30):
    """
    image_seq: either callable by index or an indexible image set
    """
    if callable(image_seq):
        def get_image(i):
            return image_seq(i)
    else:
        def get_image(i):
            return image_seq[i]
    
    image_seq = [get_image(i) for i in range(len(image_seq))]

    try:
        imageio.mimsave(video_fn, image_seq, fps=fps, quality=8)
    except Exception as e:
        print(e)
        print(f'[NOTE] If you encounter the error message: \"TypeError: TiffWriter.write() got an unexpected keyword argument \'fps\'\", please do `pip install imageio-ffmpeg`')
        
