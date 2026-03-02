import numpy as np
res = np.array([-0.00285813, -0.00367473, -0.00351771, -0.0041936, -0.0030776], dtype=np.float32)
# Max abs should be around 1.0 for healthy audio
print("Max abs:", np.max(np.abs(res)))
