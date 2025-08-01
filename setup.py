from setuptools import setup 
  
setup( 
    name='cvcg_utils', 
    version='0.1', 
    description='Easy-to-use wrappers for various CV & CG tools', 
    author='Siyou Lin', 
    author_email='linsy21@mails.tsinghua.edu.cn', 
    packages=['cvcg_utils',
              'cvcg_utils.misc',
              'cvcg_utils.misc.train_templates',
              'cvcg_utils.mesh',
              'cvcg_utils.pcd',
              'cvcg_utils.render',
              'cvcg_utils.uv',
              'cvcg_utils.external.nvdiffrecmc.render',
              'cvcg_utils.external.nvdiffrecmc.render.renderutils',
              'cvcg_utils.external.nvdiffrecmc.render.optixutils',
              'cvcg_utils.external.nvdiffrecmc.denoiser',
              ], 
    package_data={
        "cvcg_utils.misc.train_templates": ["basic_trainer/*.sh", "basic_trainer/*.py", "basic_trainer/dataset/*.py", "basic_trainer/loggers/*.py", "basic_trainer/loss_module/*.py", "basic_trainer/model/*.py", "basic_trainer/preprocessor/*.py", "basic_trainer/setup_utils/*.py"],
        "cvcg_utils.external.nvdiffrecmc.render.renderutils": ["c_src/*"],
        "cvcg_utils.external.nvdiffrecmc.render.optixutils": ["c_src/*", "c_src/envsampling/*", "include/*", "include/internal/*"],
    }
    # install_requires=[ 
    #     'numpy', 
    #     'pandas', 
    # ],
)