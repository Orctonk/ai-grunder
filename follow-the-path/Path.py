import json

import numpy as np

# Load the path from a file and convert it into a list of coordinates
class Path:

    def __init__(self, file_name):
    
        with open(file_name) as path_file:
            data = json.load(path_file)
    
        self.path = data
    
    def getPath(self):
        vecArray = []
        for p in self.path:
            
            vecArray.append(np.array([\
                    p['Pose']['Position']['X'], \
                    p['Pose']['Position']['Y'] \
                    ]))
        return vecArray
