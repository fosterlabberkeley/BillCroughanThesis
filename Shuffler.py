from __future__ import annotations
from enum import IntEnum, auto
from typing import List, Dict, Tuple, Optional, Callable, Iterable
import numpy as np
import pandas as pd
import os
import pickle
import time
from datetime import datetime
from tqdm import tqdm
import warnings
from UtilFunctions import getPreferredCategoryOrder
from itertools import product
from scipy.stats import pearsonr, linregress
import matplotlib.pyplot as plt
import math


def notNanPvalFilter(plotName: str, shuffleResult: ShuffleResult, pval: float, measureName: str) -> bool:
    return not np.isnan(pval)


class ShuffSpec:
    """
    Specifies one shuffle to run
    if type is WITHIN, GLOBAL, or INTERACTION, value is a single value
    if type is ACROSS or RECURSIVE_ALL, value is None
    """
    class ShuffType(IntEnum):
        UNSPECIFIED = auto()
        GLOBAL = auto()
        WITHIN = auto()
        ACROSS = auto()
        INTERACTION = auto()
        RECURSIVE_ALL = auto()  # includes interaction, across, and within

        def __repr__(self) -> str:
            return self.name

        def __str__(self) -> str:
            return self.name[0:3]

    def __init__(self, shuffType=ShuffType.UNSPECIFIED, categoryName="", value=None):
        self.shuffType = shuffType
        self.categoryName = categoryName
        self.value = value

    def copy(self) -> ShuffSpec:
        ret = ShuffSpec()
        ret.shuffType = self.shuffType
        ret.categoryName = self.categoryName
        ret.value = self.value
        return ret

    def __str__(self) -> str:
        return "{} {} ({})".format(self.shuffType, self.categoryName, self.value)

    def __repr__(self) -> str:
        return "{} {} ({})".format(self.shuffType, self.categoryName, self.value)

    def __lt__(self, other) -> bool:
        if self.shuffType < other.shuffType:
            return True
        elif self.shuffType > other.shuffType:
            return False

        if self.categoryName < other.categoryName:
            return True
        elif self.categoryName > other.categoryName:
            return False

        if self.value is None or other.value is None:
            return False
        return self.value < other.value

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ShuffSpec):
            return False
        return self.shuffType == __o.shuffType and self.categoryName == __o.categoryName and \
            (self.value is None) == (__o.value is None) and (
                self.value is None or self.value == __o.value)

    def __ne__(self, __o: object) -> bool:
        return not self == __o


class ShuffleResult:
    def __init__(self, specs: List[ShuffSpec] = [], diff=np.nan, shuffleDiffs=None, dataNames: List[str] = []) -> None:
        self.specs = specs
        self.diff = diff
        self.shuffleDiffs = shuffleDiffs
        self.dataNames = dataNames

    def copy(self) -> ShuffleResult:
        ret = ShuffleResult()
        ret.diff = self.diff
        if self.shuffleDiffs is not None:
            ret.shuffleDiffs = np.copy(self.shuffleDiffs)
        ret.specs = [s.copy() for s in self.specs]
        ret.dataNames = self.dataNames.copy()
        return ret

    def getPVals(self) -> np.ndarray:
        if np.all(np.isnan(self.shuffleDiffs)):
            return np.full(self.diff.shape, np.nan)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"invalid value encountered in divide")
            pvals1 = np.count_nonzero(self.diff.T < self.shuffleDiffs,
                                      axis=0) / np.count_nonzero(~np.isnan(self.shuffleDiffs), axis=0)
            pvals2 = np.count_nonzero(self.diff.T <= self.shuffleDiffs,
                                      axis=0) / np.count_nonzero(~np.isnan(self.shuffleDiffs), axis=0)
        return (pvals1 + pvals2) / 2

    def getFullInfoString(self, linePfx="") -> str:
        sdmin = np.min(self.shuffleDiffs, axis=0)
        sdmax = np.max(self.shuffleDiffs, axis=0)
        pvals1 = np.count_nonzero(self.diff.T < self.shuffleDiffs,
                                  axis=0) / self.shuffleDiffs.shape[0]
        pvals2 = np.count_nonzero(self.diff.T <= self.shuffleDiffs,
                                  axis=0) / self.shuffleDiffs.shape[0]
        ret = [f"{self.specs}:"] + [linePfx + f"{self.dataNames[i]}: {float(self.diff[i])} ({sdmin[i]}, {sdmax[i]}) "
                                    f"p1 = {pvals1[i]}\tp2 = {pvals2[i]}" for i in range(len(self.diff))]
        return "\n".join(ret)

    def __str__(self) -> str:
        if isinstance(self.diff, np.ndarray):
            diffStr = ", ".join([f"{d:.3f}" for d in self.diff.flat])
        else:
            diffStr = f"{self.diff:.3f}"
        if self.shuffleDiffs is None:
            return f"{self.specs}: {diffStr}"
        else:
            return f"{self.specs}: {diffStr} ({np.min(self.shuffleDiffs):.3f}, {np.max(self.shuffleDiffs):.3f})"

    def __repr__(self) -> str:
        if isinstance(self.diff, np.ndarray):
            diffStr = ", ".join([f"{d:.3f}" for d in self.diff.flat])
        else:
            diffStr = f"{self.diff:.3f}"
        if self.shuffleDiffs is None:
            return f"{self.specs}: {diffStr}"
        else:
            return f"{self.specs}: {diffStr} ({np.min(self.shuffleDiffs):.3f}, {np.max(self.shuffleDiffs):.3f})"


class Shuffler:
    def __init__(self, rng: np.random.Generator = None, numShuffles=100) -> None:
        self.rng = rng
        self.numShuffles = numShuffles
        self.customShuffleFunctions: Dict[str, Callable[[
            pd.DataFrame, str, np.random.Generator], pd.Series]] = {}

    def setCustomShuffleFunction(self, category: str, func: Callable[[pd.DataFrame, str, np.random.Generator], pd.Series]):
        self.customShuffleFunctions[category] = func

    def doShuffles(self, df: pd.DataFrame, specs: List[List[ShuffSpec]],
                   dataNames: List[str], numShuffles: List[int] = None) -> List[List[ShuffleResult]]:
        if numShuffles is not None:
            assert len(numShuffles) == len(specs)

        columnsToShuffle = set()
        for spec in specs:
            for s in spec:
                columnsToShuffle.add(s.categoryName)

        # print(df)
        valSet: Dict[str, set] = {}
        for col in columnsToShuffle:
            vs = set(df[col])
            if len(vs) <= 1:
                raise Exception(
                    "Only one value {} for category {}".format(vs, col))
            valSet[col] = vs
        # print("created valset:", valSet)

        ret: List[List[ShuffleResult]] = []
        for si, spec in enumerate(specs):
            # print(f'{spec}                 ', end='\r')
            if numShuffles is not None:
                self.numShuffles = numShuffles[si]
            res = self._doShuffleSpec(df, spec, valSet, dataNames)
            ret.append(res)

        return ret

    def getListOfStatsFiles(self, infoFileName: str) -> List[Tuple[str, str]]:
        with open(infoFileName, "r") as fid:
            lines = fid.readlines()
            statsFiles = []
            for line in lines:
                if line.startswith("statsFile:"):
                    statsFiles.append(
                        (line.split("__!__")[1].strip(), line.split("__!__")[2].strip()))
        return statsFiles

    def makePlotsForShuffleResults(self, sr: List[List[ShuffleResult]], df: pd.DataFrame, fname: str):
        flatRes = [r for rr in sr for r in rr]
        ncols = math.ceil(math.sqrt(len(flatRes)))
        nrows = math.ceil(len(flatRes) / ncols) * 2
        statFig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        statFig.set_figheight(1.5 * nrows)
        statFig.set_figwidth(2 * ncols)
        for ri, r in enumerate(flatRes):
            if len(r.getPVals()) > 1:
                # destroy the figure and return
                plt.close(statFig)
                return
            pval = r.getPVals().item()
            shufDiffs = r.shuffleDiffs
            dataDiff = r.diff
            txtAx = axs[(ri // ncols) * 2, ri % ncols]
            shufAx = axs[(ri // ncols) * 2 + 1, ri % ncols]
            # shufAx = axs[1, ri]
            # txtAx = axs[0, ri]

            if not np.isnan(shufDiffs).all() and not np.isnan(dataDiff).all():
                # axis should have text at the top with the following on each row:
                #   name of shuffle with the category and type of shuffle
                #   pval
                #   direction of difference with names of the categories
                # Then below that text put the histogram of the shuffled differences
                # with a red vertical line at the data difference
                topText = ""
                for s in r.specs:
                    if s.shuffType == ShuffSpec.ShuffType.GLOBAL:
                        topText += f"Main: {s.value} ({s.categoryName})\n"
                        shuffledCategories = df[s.categoryName].unique()
                        cat1 = s.value
                        otherCats = shuffledCategories[shuffledCategories != cat1]
                        if len(otherCats) == 1:
                            cat2 = otherCats.item()
                        else:
                            cat2 = "other"
                    elif s.shuffType == ShuffSpec.ShuffType.ACROSS:
                        topText += f"Across: {s.categoryName}\n"
                    elif s.shuffType == ShuffSpec.ShuffType.WITHIN:
                        topText += f"Within: {s.value} ({s.categoryName})\n"
                    elif s.shuffType == ShuffSpec.ShuffType.INTERACTION:
                        topText += f"Interaction: {s.value} ({s.categoryName})\n"

                direction = pval > 0.5
                if direction:
                    pval = 1 - pval
                if pval > 0:
                    topText += f"p={round(pval, 3)}\n{cat1} {'<' if direction else '>'} {cat2}"
                else:
                    topText += f"p<{round(1/len(shufDiffs), 3)}\n{cat1} {'<' if direction else '>'} {cat2}"
                # make this text fill up all of txtAx
                txtAx.axis("off")
                txtAx.text(0.5, 0, topText, horizontalalignment="center",
                           verticalalignment="bottom", transform=txtAx.transAxes)

                shufAx.hist(shufDiffs, bins=20, color="black")
                shufAx.axvline(dataDiff, color="red")
                shufAx.set_xlabel(f"{cat1} - {cat2}")
                # turn off y axis
                shufAx.get_yaxis().set_visible(False)

        statFig.savefig(fname + ".png", bbox_inches="tight",
                                dpi=100, transparent=False)
        plt.close(statFig)

    def runImmediateShufflesAcrossPersistentCategories(self, infoFileNames: Optional[List[str]], numShuffles=100,
                                                       significantThreshold: Optional[float] = 0.05,
                                                       resultsFilter: Callable[[
                                                           str, ShuffleResult, float, str], bool] = lambda *_: True,
                                                       savedStatsFiles: Optional[List[Tuple[str, str]]] = None,
                                                       outputFileName: Optional[str] = None,
                                                       makePlots: bool = False
                                                       ) -> str:
        self.numShuffles = numShuffles

        if infoFileNames is None and savedStatsFiles is None:
            raise Exception(
                "Either infoFileNames or savedStatsFiles must be specified")

        if savedStatsFiles is None:
            savedStatsFiles = []
            for infoFileName in infoFileNames:
                savedStatsFiles += self.getListOfStatsFiles(infoFileName)
        filesForEachPlot = {}
        for plotName, statsFile in savedStatsFiles:
            if plotName not in filesForEachPlot:
                filesForEachPlot[plotName] = []
            filesForEachPlot[plotName].append(statsFile)

        if outputFileName is None:
            outputFileName = os.path.join(os.path.dirname(
                infoFileNames[0]), datetime.now().strftime("%Y%m%d_%H%M%S_shufflesAcrossPersistentCategories.h5"))
        if makePlots:
            plotOutFileNameBase = os.path.join(os.path.dirname(
                infoFileNames[0]), "shuffleImages")
            if not os.path.exists(plotOutFileNameBase):
                os.makedirs(plotOutFileNameBase)

        todel = []
        for plotName in filesForEachPlot:
            # Get the immediate shuffles, and if there's multiple versions, run each separately
            immGroups = {}
            for fname in filesForEachPlot[plotName]:
                pickleFile = fname[:-3] + "_immediateShuffles.pkl"
                if not os.path.exists(pickleFile):
                    print("No immediate shuffles for {}".format(pickleFile))
                    continue
                with open(pickleFile, "rb") as fid:
                    key = str(pickle.load(fid))

                if key not in immGroups:
                    immGroups[key] = [fname]
                else:
                    immGroups[key].append(fname)

            if len(immGroups) > 1:
                for ki, key in enumerate(immGroups):
                    filesForEachPlot[f"{plotName}_immgroup{ki}"] = immGroups[key]
                todel.append(plotName)
        for d in todel:
            del filesForEachPlot[d]

        shuffResults: List[Tuple[str, List[List[ShuffleResult]], str]] = []
        for plotName in tqdm(filesForEachPlot, desc="Running shuffles", total=len(filesForEachPlot)):
            # print("Running shuffles for {}".format(plotName))
            # Get the immediate shuffles. Should be the same for all files
            fname = filesForEachPlot[plotName][0]
            pickleFile = fname[:-3] + "_immediateShuffles.pkl"
            if not os.path.exists(pickleFile):
                # print("No immediate shuffles for {}".format(pickleFile))
                continue
            with open(pickleFile, "rb") as fid:
                immediateShuffles = pickle.load(fid)

            df = pd.concat([pd.read_hdf(f, key="stats")
                           for f in filesForEachPlot[plotName]])
            yvalNames = pd.read_hdf(
                filesForEachPlot[plotName][0], key="yvalNames")
            # categoryNames = pd.read_hdf(filesForEachPlot[plotName][0], key="categoryNames")
            persistentCategoryNames = pd.read_hdf(
                filesForEachPlot[plotName][0], key="persistentCategoryNames")
            # infoNames = pd.read_hdf(filesForEachPlot[plotName][0], key="infoNames")
            # persistentInfoNames = pd.read_hdf(filesForEachPlot[plotName][0], key="persistentInfoNames")
            measureName = os.path.dirname(
                filesForEachPlot[plotName][0]).split(os.path.sep)[-2]

            nu = df.nunique(axis=0)
            catsToShuffle = list(persistentCategoryNames)
            # print("Categories to shuffle: {}".format(catsToShuffle))
            # print("Unique vals per category:\n{}".format(nu))
            todel = set()
            for cat in catsToShuffle:
                if nu[cat] <= 1:
                    todel.add(cat)
                    print("Category {} has only one val. Not including in shuffle for plot {}".format(
                        cat, plotName))
            for td in todel:
                catsToShuffle.remove(td)

            specs = []
            for ish in immediateShuffles:
                specs += self.getAllShuffleSpecsWithLeaf(
                    df, leaf=ish[0], columnsToShuffle=catsToShuffle)
            ss = [len(s) for s in specs]
            specs = [x for _, x in sorted(zip(ss, specs))]
            # print("All specs:")
            # print("\n".join(["\t" + str(s) for s in specs]))
            sr = self._doShuffles(df, specs, yvalNames)
            shuffResults.append((plotName, sr, measureName))

            if makePlots:
                if "rat" in catsToShuffle:
                    # get unique rats
                    rats = sorted(list(df["rat"].unique()))
                    outdir = os.path.join(plotOutFileNameBase, "_".join(rats))
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    fname = os.path.join(outdir, plotName)
                else:
                    fname = os.path.join(plotOutFileNameBase, plotName)
                self.makePlotsForShuffleResults(sr, df, fname)

        if significantThreshold is None:
            significantThreshold = 1.0

        significantResults: List[Tuple[str,
                                       List[List[ShuffleResult]], float, str]] = []
        for plotName, sr, measureName in shuffResults:
            for s in sr:
                for s2 in s:
                    for pi, pval in enumerate(s2.getPVals()):
                        if pval < significantThreshold or pval > 1 - significantThreshold:
                            if resultsFilter(plotName, s2, pval, measureName):
                                plotSuffix = plotName[len(measureName):]
                                significantResults.append(
                                    (plotName, str(s2), pval if pval < 0.5 else 1 - pval, pval < 0.5,
                                        measureName, plotSuffix, pi))

        sdf = pd.DataFrame(significantResults,
                           columns=["plot", "shuffle", "pval", "direction", "measure", "plotSuffix", "pvalIndex"])
        sdf.sort_values(by="pval", inplace=True)
        sdf.to_hdf(outputFileName, key="significantShuffles")
        print(sdf)

        return outputFileName

    def getAllShuffleSpecsWithLeaf(self, df: pd.DataFrame, leaf: List[ShuffSpec],
                                   columnsToShuffle: Optional[List[str]] = None) -> List[List[ShuffSpec]]:
        if columnsToShuffle is None:
            raise Exception("Not implemented. leaf is a list...")

        ret = [leaf]
        for col in columnsToShuffle:
            otherCols = [c for c in columnsToShuffle if c != col]
            rec = self.getAllShuffleSpecsWithLeaf(
                df, leaf, columnsToShuffle=otherCols)
            for r in rec:
                ret.append([ShuffSpec(shuffType=ShuffSpec.ShuffType.RECURSIVE_ALL,
                                      categoryName=col, value=None)] + r)
        return ret

    def runCorrelations(self, df: pd.DataFrame, xvar: str, yvar: str, catNames: List[str], returnFitLine=False):
        catVals = [getPreferredCategoryOrder(df[cn].values) for cn in catNames]
        catCombs = list(product(*catVals))
        ret = []

        for cc in catCombs:
            dfc = df
            for cn, cv in zip(catNames, cc):
                dfc = dfc[dfc[cn] == cv]

            x = dfc[xvar].values
            y = dfc[yvar].values
            # print(f"Doing correlation for {catStr}x={xvar} y={yvar} with {len(x)} values")
            # filter out inf and nan values
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            if len(x) < 2:
                if returnFitLine:
                    ret.append((cc, np.nan, np.nan, np.nan, np.nan))
                else:
                    ret.append((cc, np.nan, np.nan))
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", r"An input array is constant; the correlation coefficient is not defined.")
                if returnFitLine:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        x, y)
                    ret.append((cc, r_value, p_value, slope, intercept))
                else:
                    r, p = pearsonr(x, y)
                    ret.append((cc, r, p))

        return ret

    def runAllShuffles(self,
                       infoFileNames: Optional[List[str]],
                       numShuffles: int,
                       significantThreshold: Optional[float] = 0.15,
                       shuffleResultsFilter: Callable[[
                           str, ShuffleResult, float, str], bool] = notNanPvalFilter,
                       savedStatsFiles: Optional[List[Tuple[str, str]]] = None,
                       outputFileName: Optional[str] = None,
                       justGlobal=False,
                       skipCorrelations=False,
                       skipShuffles=False, makePlots=True) -> str:
        self.numShuffles = numShuffles

        if infoFileNames is None and savedStatsFiles is None:
            raise Exception(
                "Either infoFileNames or savedStatsFiles must be specified")

        if savedStatsFiles is None:
            savedStatsFiles = []
            for infoFileName in infoFileNames:
                savedStatsFiles += self.getListOfStatsFiles(infoFileName)
        filesForEachPlot = {}
        for plotName, statsFile in savedStatsFiles:
            if plotName not in filesForEachPlot:
                filesForEachPlot[plotName] = []
            filesForEachPlot[plotName].append(statsFile)

        if makePlots:
            plotOutFileNameBase = os.path.join(os.path.dirname(
                infoFileNames[0]), "shuffleImages")
            if not os.path.exists(plotOutFileNameBase):
                os.makedirs(plotOutFileNameBase)

        shuffResults: List[Tuple[str, List[List[ShuffleResult]], str]] = []
        correlationResults: List[Tuple[str,
                                       List[Tuple[Iterable[str], float, float]], str]] = []
        for plotName in tqdm(filesForEachPlot, desc="Running shuffles and correlations", total=len(filesForEachPlot), smoothing=0):
            df = pd.concat([pd.read_hdf(f, key="stats")
                           for f in filesForEachPlot[plotName]])
            yvalNames = pd.read_hdf(
                filesForEachPlot[plotName][0], key="yvalNames")
            xvalNames = pd.read_hdf(
                filesForEachPlot[plotName][0], key="xvalNames")
            categoryNames = pd.read_hdf(
                filesForEachPlot[plotName][0], key="categoryNames")
            persistentCategoryNames = pd.read_hdf(
                filesForEachPlot[plotName][0], key="persistentCategoryNames")
            # infoNames = pd.read_hdf(filesForEachPlot[plotName][0], key="infoNames")
            # persistentInfoNames = pd.read_hdf(filesForEachPlot[plotName][0], key="persistentInfoNames")
            measureName = os.path.dirname(
                filesForEachPlot[plotName][0]).split(os.path.sep)[-2]

            # Shuffle categories
            catsToShuffle = list(categoryNames) + list(persistentCategoryNames)
            todel = set()
            for cat in catsToShuffle:
                vals = set(df[cat])
                if len(vals) <= 1:
                    todel.add(cat)
                    print("Category {} has only one val ({}). Not including in shuffle for plot {}".format(
                        cat, vals, plotName))
            for td in todel:
                catsToShuffle.remove(td)

            if not skipShuffles:
                try:
                    specs = self.getAllShuffleSpecs(
                        df, columnsToShuffle=catsToShuffle, justGlobal=justGlobal)
                except Exception as e:
                    print("Error getting shuffle specs for plot {} of measure {}".format(
                        plotName, measureName))
                    print(e)
                    raise e
                ss = [len(s) for s in specs]
                specs = [x for _, x in sorted(zip(ss, specs))]
                # print("\n".join([str(s) for s in specs]))
                sr = self._doShuffles(df, specs, yvalNames)
                shuffResults.append((plotName, sr, measureName))

                if makePlots:
                    if "rat" in catsToShuffle:
                        # get unique rats
                        rats = sorted(list(df["rat"].unique()))
                        outdir = os.path.join(
                            plotOutFileNameBase, "_".join(rats))
                        if not os.path.exists(outdir):
                            os.makedirs(outdir)
                        fname = os.path.join(outdir, plotName)
                    else:
                        fname = os.path.join(plotOutFileNameBase, plotName)
                    self.makePlotsForShuffleResults(sr, df, fname)

            if not skipCorrelations:
                # Look for xval correlations
                xynamepairs = list(product(xvalNames, yvalNames))
                cr = []
                for xname, yname in xynamepairs:
                    cr.append(self.runCorrelations(
                        df, xname, yname, catsToShuffle))

                correlationResults.append((plotName, cr, measureName))

        if significantThreshold is None:
            significantThreshold = 1.0

        if outputFileName is None:
            outputFileName = os.path.join(os.path.dirname(
                infoFileNames[0]), datetime.now().strftime("%Y%m%d_%H%M%S_significantShuffles.h5"))

        if not skipShuffles:
            significantResults: List[Tuple[str, str,
                                           float, str, str, bool, bool]] = []
            for plotName, sr, measureName in tqdm(shuffResults, desc="Filtering shuffles", total=len(shuffResults)):
                for s in sr:
                    for s2 in s:
                        for pi, pval in enumerate(s2.getPVals()):
                            if pval < significantThreshold or pval > 1 - significantThreshold:
                                if shuffleResultsFilter(plotName, s2, pval, measureName):
                                    plotSuffix = plotName[len(measureName):]
                                    significantResults.append(
                                        (plotName, str(s2), pval if pval < 0.5 else 1 - pval, pval < 0.5,
                                         measureName, plotSuffix, pi))

            sdf = pd.DataFrame(significantResults,
                               columns=["plot", "shuffle", "pval", "direction", "measure", "plotSuffix", "pvalIndex"])
            sdf.sort_values(by="pval", inplace=True)
            sdf.to_hdf(outputFileName, key="significantShuffles")
            print(sdf)
            print(f"{ len(shuffResults)=  }")

        if not skipCorrelations:
            significantCorrelations = []
            for plotName, cr, measureName in tqdm(correlationResults, desc="Filtering correlations", total=len(correlationResults)):
                for catCombRes in cr:
                    for catNames, corr, pval in catCombRes:
                        if pval < significantThreshold or pval > 1 - significantThreshold:
                            if shuffleResultsFilter(plotName, None, pval, measureName):
                                plotSuffix = plotName[len(measureName)-3:]
                                significantCorrelations.append(
                                    # (plotName, "_".join(catNames), corr, pval if pval < 0.5 else 1 - pval, pval < 0.5,
                                    (plotName, "_".join(catNames), corr, pval, corr > 0,
                                        measureName, plotSuffix))

            sdf = pd.DataFrame(significantCorrelations,
                               columns=["plot", "categories", "correlation", "pval", "direction", "measure", "plotSuffix"])
            sdf.sort_values(by="pval", inplace=True)
            sdf.to_hdf(outputFileName, key="significantCorrelations")
            infoDf = pd.DataFrame({"corrFuncVersion": [0.2]})
            infoDf.to_hdf(outputFileName, key="corrInfo")

        return outputFileName

    def _doShuffles(self, df: pd.DataFrame, specs: List[List[ShuffSpec]],
                    dataNames: List[str], numShuffles: List[int] = None) -> List[List[ShuffleResult]]:
        if numShuffles is not None:
            assert len(numShuffles) == len(specs)

        columnsToShuffle = set()
        for spec in specs:
            for s in spec:
                columnsToShuffle.add(s.categoryName)

        # print(df)
        valSet: Dict[str, set] = {}
        for col in columnsToShuffle:
            vs = set(df[col])
            if len(vs) <= 1:
                raise Exception(
                    "Only one value {} for category {}".format(vs, col))
            valSet[col] = vs
        # print("created valset:", valSet)

        ret: List[List[ShuffleResult]] = []
        for si, spec in enumerate(specs):
            # print(f'{spec}                 ', end='\r')
            if numShuffles is not None:
                self.numShuffles = numShuffles[si]
            res = self._doShuffleSpec(df, spec, valSet, dataNames)
            ret.append(res)

        return ret

    def getAllShuffleSpecs(self, df, columnsToShuffle=None, justGlobal=False):
        if columnsToShuffle is None:
            columnsToShuffle = list(df.columns)

        valSet = {}
        for col in columnsToShuffle:
            try:
                valSet[col] = sorted(list(set(df[col])))
            except TypeError as te:
                print("Error getting values for column", col)
                print("Values:", df[col].to_numpy())
                print(df.head())
                raise te

        ret = []
        for col in columnsToShuffle:
            for val in valSet[col]:
                ret.append(
                    [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName=col, value=val)])

            if not justGlobal:
                otherCols = [c for c in columnsToShuffle if c != col]
                rec = self.getAllShuffleSpecs(df, otherCols)
                for r in rec:
                    ret.append([ShuffSpec(shuffType=ShuffSpec.ShuffType.RECURSIVE_ALL,
                                          categoryName=col, value=None)] + r)

        return ret

    def _doOneWayShuffle(self, df: pd.DataFrame, spec: ShuffSpec, dataNames: List[str]) -> ShuffleResult:
        assert isinstance(spec, ShuffSpec) and \
            spec.shuffType == ShuffSpec.ShuffType.GLOBAL

        if spec.categoryName in self.customShuffleFunctions:
            shufFunc = self.customShuffleFunctions[spec.categoryName]
        else:
            def shufFunc(dataframe: pd.DataFrame, colName: str, rng: np.random.Generator) -> pd.Series:
                return dataframe[colName].sample(frac=1, random_state=rng).reset_index(drop=True)

        ret = ShuffleResult([spec], dataNames=dataNames)

        withinIdx = df[spec.categoryName] == spec.value
        d = np.array(df[withinIdx][dataNames].mean() -
                     df[~withinIdx][dataNames].mean())
        ret.diff = np.reshape(d, (-1, 1))

        ret.shuffleDiffs = np.empty((self.numShuffles, len(dataNames)))
        sdf = df.copy().reset_index(drop=True)
        for si in range(self.numShuffles):
            sdf[spec.categoryName] = shufFunc(sdf, spec.categoryName, self.rng)
            withinIdx = sdf[spec.categoryName] == spec.value
            ret.shuffleDiffs[si, :] = sdf[withinIdx][dataNames].mean() - \
                sdf[~withinIdx][dataNames].mean()

        return ret

    def _doShuffleSpec(self, df: pd.DataFrame, spec: List[ShuffSpec], valSet: Dict[str, set], dataNames: List[str]) -> List[ShuffleResult]:
        assert isinstance(spec, list)
        for si, s in enumerate(spec):
            assert isinstance(s, ShuffSpec)
            assert s.shuffType != ShuffSpec.ShuffType.UNSPECIFIED
            assert s.shuffType != ShuffSpec.ShuffType.GLOBAL or si == len(
                spec) - 1

        s = spec[0]
        if s.shuffType == ShuffSpec.ShuffType.GLOBAL:
            return [self._doOneWayShuffle(df, s, dataNames)]

        if s.shuffType == ShuffSpec.ShuffType.WITHIN:
            withinIdx = df[s.categoryName] == s.value
            recres = self._doShuffleSpec(
                df.loc[withinIdx], spec[1:], valSet, dataNames)
            for r in recres:
                r.specs = [s] + r.specs.copy()
            return recres

        if s.shuffType == ShuffSpec.ShuffType.INTERACTION:
            withinIdx = df[s.categoryName] == s.value
            withoutIdx = ~ withinIdx
            rin = self._doShuffleSpec(
                df.loc[withinIdx], spec[1:], valSet, dataNames)
            rout = self._doShuffleSpec(
                df.loc[withoutIdx], spec[1:], valSet, dataNames)
            ret = []
            for reti, (i, o) in enumerate(zip(rin, rout)):
                r = i.copy()
                r.specs = [s] + i.specs.copy()
                r.diff = i.diff - o.diff
                r.shuffleDiffs = i.shuffleDiffs - o.shuffleDiffs
                ret.append(r)
            return ret

        if s.shuffType == ShuffSpec.ShuffType.ACROSS:
            vals = valSet[s.categoryName]
            withinRes: List[List[ShuffleResult]] = []
            numInGroup: List[int] = []
            for val in vals:
                withinIdx = df[s.categoryName] == val
                numInGroup.append(withinIdx.sum())
                withinRes.append(self._doShuffleSpec(
                    df.loc[withinIdx], spec[1:], valSet, dataNames))

            groupWeight = np.array(numInGroup)
            groupWeight = groupWeight / groupWeight.sum()
            numEffects = len(withinRes[0])
            ret = []
            for ei in range(numEffects):
                r = withinRes[0][ei].copy()
                r.specs = [s] + r.specs.copy()
                r.diff = np.zeros((len(dataNames), 1))
                r.shuffleDiffs = np.zeros((self.numShuffles, len(dataNames)))
                for vi in range(len(vals)):
                    if not np.isnan(withinRes[vi][ei].diff).any():
                        r.diff += withinRes[vi][ei].diff * groupWeight[vi]
                        r.shuffleDiffs += withinRes[vi][ei].shuffleDiffs * \
                            groupWeight[vi]
                ret.append(r)
            return ret

        if s.shuffType == ShuffSpec.ShuffType.RECURSIVE_ALL:
            # all the recursive calls needed
            vals = valSet[s.categoryName]
            withinRes = []
            withoutRes = []
            numInGroup = []
            for val in vals:
                withinIdx = df[s.categoryName] == val
                withinRes.append(self._doShuffleSpec(
                    df.loc[withinIdx], spec[1:], valSet, dataNames))
                withoutIdx = ~ withinIdx
                withoutRes.append(self._doShuffleSpec(
                    df.loc[withoutIdx], spec[1:], valSet, dataNames))
                numInGroup.append(withinIdx.sum())

            groupWeight = np.array(numInGroup)
            groupWeight = groupWeight / groupWeight.sum()

            ret = []
            # within effects
            for vi, val in enumerate(vals):
                wr = withinRes[vi]
                for ei, effect in enumerate(wr):
                    r = effect.copy()
                    ss = s.copy()
                    ss.shuffType = ShuffSpec.ShuffType.WITHIN
                    ss.value = val
                    r.specs = [ss] + r.specs.copy()
                    ret.append(r)

            # across effect
            numEffects = len(withinRes[0])
            for ei in range(numEffects):
                r = withinRes[0][ei].copy()
                ss = s.copy()
                ss.shuffType = ShuffSpec.ShuffType.ACROSS
                r.specs = [ss] + r.specs.copy()
                r.diff = np.zeros((len(dataNames), 1))
                r.shuffleDiffs = np.zeros((self.numShuffles, len(dataNames)))
                for vi in range(len(vals)):
                    if not np.isnan(withinRes[vi][ei].diff).any():
                        r.diff += withinRes[vi][ei].diff * groupWeight[vi]
                        r.shuffleDiffs += withinRes[vi][ei].shuffleDiffs * \
                            groupWeight[vi]
                ret.append(r)

            # interaction effects
            for vi, val in enumerate(vals):
                wr = withinRes[vi]
                wor = withoutRes[vi]
                for ei, (rin, rout) in enumerate(zip(wr, wor)):
                    ss = s.copy()
                    ss.shuffType = ShuffSpec.ShuffType.INTERACTION
                    ss.value = val
                    r = ShuffleResult([ss] + rin.specs.copy())
                    r.dataNames = dataNames
                    r.diff = rin.diff - rout.diff
                    r.shuffleDiffs = rin.shuffleDiffs - rout.shuffleDiffs
                    ret.append(r)

            return ret
