class LogFormat:
    """
    Class that determines basic output (log) flair and colour format.
    Inspired by Blender source code.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


    INIT = '{}[INIT] {}'.format(OKGREEN, ENDC)
    INIT_START = '{}{}Initialization started{}'.format(INIT, HEADER, ENDC)
    INIT_DONE = '{}{}Initialization Done{}'.format(INIT, BOLD, ENDC)

    TRAINING = '{}[TRAINING] {}'.format(OKBLUE, ENDC)
    TRAINING_START = '{}{}Training started{}'.format(TRAINING, HEADER, ENDC)
    TRAINING_DONE = '{}{}Training done{}'.format(TRAINING, BOLD, ENDC)

    SAVE = '{}[SAVE]{}'.format(UNDERLINE, ENDC)
    LOAD = '{}[LOAD]{}'.format(UNDERLINE, ENDC)




