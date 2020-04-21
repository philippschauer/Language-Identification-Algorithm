# Dictionary with Parameters for the audio files and training

Config = {}

Config['root_path'] = 'train'
Config['sampling_rate'] = 16_000
Config['hop_time'] = 0.01  # in seconds, hop length
Config['fft_time'] = 0.01  # in seconds, seq length
Config['mfcc_features'] = 64
Config['sequence_length'] = 500 # 5 seconds

Config['epochs'] = 30
Config['batch_size'] = 10
Config['Dropout'] = 0.2

Config['Debug'] = False  # Only 5 files per language if True
Config['path_debug'] = 'train_debug'

mapping = {'english': 0, 'hindi': 1, 'mandarin': 2}

