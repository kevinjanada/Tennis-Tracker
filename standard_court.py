import numpy as np

# Court dimensions in meters
dimensions = {
    "width": 10.97,
    "height": 23.77
}

'''
Outer Court (Doubles)

    Bottom-left: (0,0)
    Bottom-right: (10.97,0)
    Top-left: (0,23.77)
    Top-right: (10.97,23.77)

Inner Court (Singles)

    Bottom-left: (1.37,0)
    Bottom-right: (9.60,0)
    Top-left: (1.37,23.77)
    Top-right: (9.60,23.77)

Net Line

    Left-midpoint: (0,11.89)
    Right-midpoint: (10.97,11.89)

Service Boxes

    Service Line (Horizontal): 6.40 meters from the net (from (0,11.89−6.40)(0,11.89−6.40) to (10.97,11.89−6.40)(10.97,11.89−6.40)).
    Center Service Line (Vertical): Divides the service boxes at the center of the court ((5.485,0)(5.485,0) to (5.485,23.77)(5.485,23.77)).

Left Service Box (from the server's perspective):

    Bottom-left: (1.37,11.89−6.40)
    Bottom-right: (5.485,11.89−6.40)
    Top-left: (1.37,11.89)
    Top-right: (5.485,11.89)

Right Service Box:

    Bottom-left: (5.485,11.89−6.40)
    Bottom-right: (9.60,11.89−6.40)
    Top-left: (5.485,11.89)
    Top-right: (9.60,11.89)

Doubles Alley

    Width: 1.37 meters on each side of the singles court.

Left Alley:

    Bottom-left: (0,0)
    Bottom-right: (1.37,0)
    Top-left: (0,23.77)
    Top-right: (1.37,23.77)

Right Alley:

    Bottom-left: (9.60,0)
    Bottom-right: (10.97,0)
    Top-left: (9.60,23.77)
    Top-right: (10.97,23.77)
'''
coordinates = {
    "doubles_outer": [
        (0, 0),         # bottom left
        (10.97, 0),     # bottom right
        (10.97, 23.77), # top right
        (0, 23.77)      # top left
    ],
    "singles_inner": [
        (1.37, 0), # bottom left
        (9.60, 0), # bottom right
        (9.60, 23.77), # top right
        (1.37, 23.77) # top left
    ],
    "net": [
        (0, 11.89), # left midpoint
        (10.97, 11.89) # right midpoint
    ],
    "service_boxes": {
        "left": [
            (1.37, 11.89 - 6.40), # bottom left
            (5.485, 11.89 - 6.40), # bottom right
            (5.485, 11.89), # top right
            (1.37, 11.89) # top left
        ],
        "right": [
            (5.485, 11.89 - 6.40), # bottom left
            (9.60, 11.89 - 6.40), # bottom right
            (9.60, 11.89), # top right
            (5.485, 11.89) # top left
        ],
    },
    "alleys": {
        "left": [
            (0, 0), # bottom left
            (1.37, 0), # bottom right
            (1.37, 23.77), # top right
            (0, 23.77) # top left
        ],
        "right": [
            (9.60, 0), # bottom left
            (10.97, 0), # bottom right
            (10.97, 23.77), # top right
            (9.60, 23.77) # top left
        ],
    }
}

# Define the real-world court lines (meters)
court_lines = [
    # Outer doubles court
    [(0, 0), (10.97, 0)],  # Bottom line
    [(0, 23.77), (10.97, 23.77)],  # Top line
    [(0, 0), (0, 23.77)],  # Left line
    [(10.97, 0), (10.97, 23.77)],  # Right line

    # Inner singles court
    [(1.37, 0), (9.60, 0)],  # Bottom line
    [(1.37, 23.77), (9.60, 23.77)],  # Top line
    [(1.37, 0), (1.37, 23.77)],  # Left line
    [(9.60, 0), (9.60, 23.77)],  # Right line

    # Net line
    [(0, 11.89), (10.97, 11.89)],

    # Service boxes
    [(1.37, 11.89 - 6.40), (9.60, 11.89 - 6.40)],  # Service line
    [(5.485, 0), (5.485, 23.77)],  # Center service line
]
