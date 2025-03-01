from setuptools import setup 
  
setup( 
    name='cvcg_utils', 
    version='0.0', 
    description='Easy-to-use wrappers for various CV & CG tools', 
    author='Siyou Lin', 
    author_email='linsy21@mails.tsinghua.edu.cn', 
    packages=['cvcg_utils',
              'cvcg_utils.image_io',
              'cvcg_utils.video_io',
              'cvcg_utils.pickle_io',
              'cvcg_utils.mesh_proc',
              'cvcg_utils.pcd_proc',
              'cvcg_utils.render_utils',
              'cvcg_utils.uv_utils',
              'cvcg_utils.external.nvdiffrecmc.render',
              'cvcg_utils.external.nvdiffrecmc.render.renderutils',
              'cvcg_utils.external.nvdiffrecmc.render.optixutils',
              'cvcg_utils.external.nvdiffrecmc.denoiser',
              ], 
    package_data={
        "cvcg_utils.external.nvdiffrecmc.render.renderutils": ["c_src/*"],
        "cvcg_utils.external.nvdiffrecmc.render.optixutils": ["c_src/*", "c_src/envsampling/*", "include/*", "include/internal/*"],
    }
    # install_requires=[ 
    #     'numpy', 
    #     'pandas', 
    # ], 
)