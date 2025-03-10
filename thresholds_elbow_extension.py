# thresholds_elbow_extension.py
def get_thresholds_elbow_extension():
    """
    Thresholds for the elbow extension exercise.
    'FLEXED' is the starting (bent) position,
    and 'EXTENDED' is the fully extended arm.
    """
    thresholds = {
        'ELBOW_ANGLE': {
            'FLEXED': (20, 60),       # flexed arm
            'EXTENDED': (160, 180)    # extended arm
        },
        'INACTIVE_THRESH': 15.0,
        'CNT_FRAME_THRESH': 30,
        'ELBOW_ALIGNMENT_THRESHOLD': 50  # maximum horizontal distance (in pixels) allowed between shoulder and elbow
    }
    return thresholds
