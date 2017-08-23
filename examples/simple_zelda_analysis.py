import sys
import os

path = '/Users/avigan/Work/GitHub/pyZELDA/'
if path not in sys.path:
    sys.path.append(path)
import pyzelda.zelda as zelda


data_path = os.path.join(path, 'data/')

clear_pupil_file = 'SPHERE_GEN_CLEAR_PUPIL'
zelda_pupil_file = 'SPHERE_GEN_ZELDA_PUPIL'
dark_file = 'SPHERE_GEN_BACKGROUND'

wave = 1.642e-6

clear_pupil, zelda_pupil, center = zelda.read_files(data_path, clear_pupil_file, zelda_pupil_file, dark_file)

opd_map = zelda.analyze(clear_pupil, zelda_pupil, wave=wave)
