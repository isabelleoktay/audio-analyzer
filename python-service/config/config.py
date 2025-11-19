CREPE_MODEL_PATH = "./models/crepe-small-1.pb"
TEMPO_MODEL_PATH = "./models/deeptemp-k16-3.pb"
CREPE_HOP_SIZE_MS = 10
CREPE_HOP_DURATION_SEC = CREPE_HOP_SIZE_MS / 1000.0

PHONATION_MODEL_PATH = "./models/phonation_vggish_cnn_tf_2_15.h5"
PHONATION_DIMENSION = 2
PHONATION_FRAME_DURATION_SEC = 0.1
PHONATION_VGGISH_URL = "https://tfhub.dev/google/vggish/1"

CLAP_MALE_PITCH_MODEL_PATH = "./models/CLAP/best_clap_pitch_male_0_5_model.pth"
CLAP_FEMALE_PITCH_MODEL_PATH = "./models/CLAP/best_clap_pitch_female_0_5_model.pth"
CLAP_MALE_TIMBRE_MODEL_PATH = "./models/CLAP/best_clap_timbre_male_0_5_model.pth"
CLAP_FEMALE_TIMBRE_MODEL_PATH = "./models/CLAP/best_clap_timbre_female_0_5_model.pth"
WHISPER_MALE_PITCH_MODEL_PATH = "./models/Whisper/best_whisper_pitch_male_0_5_model.pth"
WHISPER_FEMALE_PITCH_MODEL_PATH = "./models/Whisper/best_whisper_pitch_female_0_5_model.pth"
WHISPER_MALE_TIMBRE_MODEL_PATH = "./models/Whisper/best_whisper_timbre_male_0_5_model.pth"
WHISPER_FEMALE_TIMBRE_MODEL_PATH = "./models/Whisper/best_whisper_timbre_female_0_5_model.pth"
VTC_FRAME_DURATION_SEC = 0.5
VTC_OVERLAP = 0.25

HOP_SIZE = 25
HOP_LENGTH = 512
N_FFT = 2048
WINDOW_SIZE = 100
WINDOW_PERCENTAGE = 0.05
HOP_PERCENTAGE = 0.0125
SEGMENT_PERCENTAGE = 0.1

LOOKUP_TABLE = "frequency_to_note.csv"