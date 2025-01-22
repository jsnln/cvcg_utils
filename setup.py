from setuptools import setup 
  
setup( 
    name='geometry_utils', 
    version='0.0', 
    description='A sample Python package', 
    author='sst', 
    author_email='linsy21@mails.tsinghua.edu.cn', 
    packages=['geometry_utils', 'geometry_utils.image_io', 'geometry_utils.video_io', 'geometry_utils.pickle_io', 'geometry_utils.mesh_proc', 'geometry_utils.render_utils', 'geometry_utils.uv_utils'], 
    # install_requires=[ 
    #     'numpy', 
    #     'pandas', 
    # ], 
)