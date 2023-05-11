from multiprocessing import Pool, Manager
from typing import Callable, Iterable, Dict, List, Any, Optional
from itertools import product, repeat
from tqdm import tqdm

_DMFFunction = None


def _setDMFFunction(function: Callable) -> None:
    global _DMFFunction
    _DMFFunction = function


def _runParFunc(arg) -> None:
    global _DMFFunction
    _DMFFunction(arg)


def runEveryCombination(function: Callable, parameters: Dict[str, Iterable],
                        numParametersSequential: int = 2,
                        eachSeqIterFunc: Optional[Callable[[int], None]] = None) -> None:
    seqParams = {}
    parParams = {}
    for ki, (key, value) in enumerate(parameters.items()):
        if ki < numParametersSequential:
            seqParams[key] = value
        else:
            parParams[key] = value

    def recursiveProduct(d: Dict[str, Iterable]) -> List[Dict[str, Any]]:
        if len(d) == 0:
            return []
        key = list(d.keys())[0]
        values = d[key]
        newD = d.copy()
        del newD[key]
        ret = []
        for value in values:
            if len(newD) > 0:
                for combo in recursiveProduct(newD):
                    ret.append({key: value, **combo})
            else:
                ret.append({key: value})
        return ret

    seqCombos = recursiveProduct(seqParams)
    parCombos = recursiveProduct(parParams)

    for seqI, seqParam in tqdm(enumerate(seqCombos), total=len(seqCombos)):
        if len(parParams) > 0:
            args = []
            for parParam in parCombos:
                args.append({**seqParam, **parParam})

            # args = zip(args, repeat(sharedDict, len(args)))
            # args = [seqParam + parParam for parParam in product(*parParams)]
            # print(f"{ args = }")
            with Pool(processes=7, initializer=_setDMFFunction, initargs=(function, )) as pool:
                pool.map(_runParFunc, args)
            # sharedDictCallback(sharedDict)
        else:
            function(*seqParam)

        if eachSeqIterFunc is not None:
            eachSeqIterFunc(seqI)


if __name__ == "__main__":
    def f(a, b, c, d):
        # print(a, b, c, d)
        for i in range(30000000):
            pass

    def f2(a, b, c, d, e):
        # print(a, b, c, d, e)
        for i in range(30000000):
            pass

    print("f, 2")
    runEveryCombination(f, [[1, 2], [3, 4], [5, 6], [7, 8]], 2)
    print("f2, 2")
    runEveryCombination(f2, [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], 2)
    print("f2, 3")
    runEveryCombination(f2, [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], 3)
    print("f, 2")
    runEveryCombination(f, [[1, 2], [3, 4], [5, 6], [7, 8]], 2)
