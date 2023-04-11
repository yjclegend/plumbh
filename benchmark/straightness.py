import matplotlib.pyplot as plt
from imagedata.realimage.PolygonImage import find_segments

from common.utils import cal_residual, enum_path, fit_line


files = enum_path("data/testcase/zhang2")
segments = find_segments(files)
res = 0
count = 0
for seg in segments:
    norm_vec, c = fit_line(seg.sample)
    res += cal_residual(seg.sample, norm_vec, c)
    count += seg.sample.shape[0]
    plt.scatter(seg.sample[:, 0], seg.sample[:, 1])
print(res)
print(count)
print(res/count)
plt.show()
