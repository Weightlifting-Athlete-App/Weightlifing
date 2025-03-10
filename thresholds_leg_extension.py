# thresholds_leg_extension.py
def get_thresholds_leg_extension():
    """
    Thresholds for the leg extension exercise.
    'BENT' represents a significantly bent knee,
    and 'EXTENDED' represents a nearly straight leg.
    """
    thresholds = {
        'KNEE_ANGLE': {
            'BENT': (70, 100),       # approximate bent knee range
            'EXTENDED': (160, 180)    # near full extension
        },
        'KNEE_ALIGNMENT_THRESHOLD': 50,
        'INACTIVE_THRESH': 15.0,
        'CNT_FRAME_THRESH': 30,
        # (Add additional parameters if needed for further corrections)
    }
    return thresholds
