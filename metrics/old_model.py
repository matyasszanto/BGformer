from scipy.interpolate import interp1d
import pandas as pd
import numpy as np

class OldModel():
    def __init__(self, csv):
        data = pd.read_csv(csv, index_col=False, header=0, names=["si", "p5", "p25", "p50", "p75", "p95"])[1:]
        self.ll = data.p5.values.astype(np.float64)
        self.hh = data.p95.values.astype(np.float64)
        self.x = data.si.values.astype(np.float64)
        self.li = interp1d(self.x, self.ll, kind="nearest", fill_value="extrapolate")
        self.hi = interp1d(self.x, self.hh, kind="nearest", fill_value="extrapolate")


    def predict(self, si, trans = None):
        if trans is not None:
            ll = trans.transform(self.li(si).reshape(-1, 1)).reshape(-1)
            hh = trans.transform(self.hi(si).reshape(-1, 1)).reshape(-1)
        else:
            ll = self.li(si)
            hh = self.hi(si)
        return ll, hh

    def width(self, si, trans = None):
        ll, hh = self.predict(si, trans)
        return hh - ll
