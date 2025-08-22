# from cvcg_utils.misc.video import read_video_rgb, write_video
from cvcg_utils.video import write_video
from cvcg_utils.video import write_video_cv2
from cvcg_utils.video.video_io_imageio_backend import read_video


import numpy as np
import einops

x = np.random.rand(4, 4, 3)
y = np.random.rand(4, 4, 3)

images = []
for step in range(120):
    t = step / 119
    x_t = x * (1-t) + y * t
    x_t = (x_t * 255).astype(np.uint8)
    x_t = einops.repeat(x_t, 'h w c -> (h h2) (w w2) c', h2=128, w2=128)
    images.append(x_t)

write_video('test_video_mimsave.mp4', images)   # this is directly playable on websites
write_video_cv2('test_video_cv2.mp4', images)   # this has issues with encoding, not playable on websites

### read ###
video_reload = read_video('test_video_mimsave.mp4')
