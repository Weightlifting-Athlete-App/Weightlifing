# thresholds_elbow_flexion.py
def get_thresholds_elbow_flexion():
    """
    Thresholds for the elbow flexion exercise.
    'EXTENDED' corresponds to a nearly straight arm,
    and 'FLEXED' corresponds to a fully curled arm.
    """
    thresholds = {
        'ELBOW_ANGLE': {
            'EXTENDED': (140, 180),  # arm nearly straight
            'FLEXED': (20, 60)       # arm fully curled (approximate)
        },
        'INACTIVE_THRESH': 15.0,   # seconds before resetting rep count
        'CNT_FRAME_THRESH': 30,    # frame count threshold for display updates
        # Corrective measure threshold: maximum horizontal gap (in pixels)
        'ELBOW_ALIGNMENT_THRESHOLD': 50  
    }
    return thresholds
