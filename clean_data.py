import os
import shutil
from config import FLAGS

os.remove('freeze-graph-data/frozen_graph.pb')
# shutil.rmtree('saved-model-data')