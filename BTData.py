import json
import numpy as np
from BTSession import BTSession
# from BTRestSession import BTRestSession
from datetime import datetime
from typing import List, Callable, Any, Tuple
from dataclasses import asdict, is_dataclass, replace, Field, fields
import sys
from UtilFunctions import LoadInfo, ImportOptions, Ripple


class BTData:
    def __init__(self) -> None:
        self.filename = ""
        self.allSessions: List[BTSession] = []
        # self.allRestSessions: List[BTRestSession] = []
        self.importOptions = None

    class ArrayAndDataclassEncoder(json.JSONEncoder):
        def shouldRecurse(self, type):
            try:
                C = getattr(sys.modules[__name__], type)
                if is_dataclass(C):
                    return True
            except:
                C = type
                if is_dataclass(C) or "numpy" in str(C) or "np" in str(C) or "list" in str(C).lower():
                    return True
            return False

        def default(self, o: Any) -> Any:
            if o is None:
                return {"__encoder_special_key_wdc": "None"}
            if isinstance(o, np.ndarray):
                return {"__encoder_special_key_wdc": "np.ndarray",
                        "dtype": str(o.dtype),
                        "val": o.tolist()}
            if type(o).__module__ in ["numpy", "np"]:
                return {"__encoder_special_key_wdc": "numpytype",
                        "numpyType": type(o).__name__,
                        "val": o.item()}
            if isinstance(o, datetime):
                return {"__encoder_special_key_wdc": "datetime",
                        "val": o.isoformat()}
            if is_dataclass(o):
                o = replace(o)
                fs: Tuple[Field, ...] = fields(o.__class__)
                # print(f"starting recursing for encoding type {type(o)}")
                for f in fs:
                    recurse = self.shouldRecurse(f.type)
                    # print(f.name, f.type, C, recurse, foundInModule, sep="\t")
                    if recurse:
                        fd = self.default(getattr(o, f.name))
                        setattr(o, f.name, fd)
                d = asdict(o)
                # print(f"finished recursing for encoding type {type(o)}")
                d["__encoder_special_key_wdc"] = type(o).__name__
                return d
            if isinstance(o, list):
                if len(o) == 0:
                    return []
                if self.shouldRecurse(type(o[0])):
                    return [self.default(v) for v in o]
                return o
            if isinstance(o, tuple):
                return {"__encoder_special_key_wdc": "tuple",
                        "val": self.default(list(o))}
            return super().encode(o)

    def arrayAndDataclassDecodeHook(self, o: Any):
        if isinstance(o, dict) and "__encoder_special_key_wdc" in o:
            cstr = o["__encoder_special_key_wdc"]
            if cstr == "None":
                return None
            if cstr == "np.ndarray":
                return np.asarray(o["val"]).astype(o["dtype"])
            if cstr == "datetime":
                return datetime.fromisoformat(o["val"])
            if cstr == "numpytype":
                C = getattr(sys.modules["numpy"], o["numpyType"])
                return C(o["val"])
            if cstr == "tuple":
                return tuple(o["val"])
            C = getattr(sys.modules[__name__], cstr)
            del o["__encoder_special_key_wdc"]
            for k in o:
                o[k] = self.arrayAndDataclassDecodeHook(o[k])
            if cstr == "LoadInfo" and "maximum_date" not in o:
                # Changed LoadInfo after processing all the data, so need to add this field
                o["maximum_date"] = None
            return C(**o)
        return o

    def saveToFile(self, filename: str) -> int:
        saveDict = {
            "allSessions": self.allSessions,
            # "allRestSessions": self.allRestSessions,
            "importOptions": self.importOptions
        }
        with open(filename, "w") as f:
            json.dump(saveDict, f, cls=BTData.ArrayAndDataclassEncoder)
        self.filename = filename
        return 0

    def loadFromFile(self, filename: str, prevSessionCutoff: float = 36) -> int:
        """
        prevSessionCutoff: if the time between the end of the last session and the start of the next session is 
        greater than this in hours, then prevSession is left as None. Otherwise, it is set to the previous session.
        """
        with open(filename, "r") as f:
            loadDict = json.load(f, object_hook=self.arrayAndDataclassDecodeHook)
            self.allSessions = loadDict["allSessions"]
            prevStartTime = None
            for si, s in enumerate(self.allSessions):
                try:
                    startTime = datetime.strptime(s.name, "%Y%m%d_%H%M%S")
                except ValueError:
                    startTime = datetime.strptime(s.name, "test%Y%m%d_%H%M%S")
                if si == 0:
                    s.prevSession = None
                else:
                    # sesh.name is the date and time, in year-month-day hour-minute-second format
                    if (startTime - prevStartTime).total_seconds() / 3600 < prevSessionCutoff:
                        s.prevSession = self.allSessions[si - 1]
                        self.allSessions[si - 1].nextSession = s
                        s.secondsSincePrevSession = (startTime - prevStartTime).total_seconds()
                        self.allSessions[si - 1].secondsUntilNextSession = s.secondsSincePrevSession
                    else:
                        s.prevSession = None
                prevStartTime = startTime
            # self.allRestSessions = loadDict["allRestSessions"]
            self.importOptions = loadDict["importOptions"]
        self.filename = filename
        return 0

    # can pass in optional filter function, otherwise returns all blocks
    def getSessions(self, predicate: Callable[[BTSession], bool] = lambda b: True) -> List[BTSession]:
        return list(filter(predicate, self.allSessions))

    # def getRestSessions(self, predicate: Callable[[BTSession], bool] = lambda b: True) -> List[BTSession]:
    #     return list(filter(predicate, self.allRestSessions))
