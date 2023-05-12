from __future__ import annotations
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Callable, List, Tuple, Optional, Iterable, Dict, TypeVar, Any, Set
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike
from multiprocessing import Pool
import warnings
from dataclasses import dataclass
from functools import partial
import os
from itertools import product

from UtilFunctions import offWall, getWellPosCoordinates, getRotatedWells
from PlotUtil import violinPlot, PlotManager, setupBehaviorTracePlot, blankPlot, \
    plotIndividualAndAverage
from Shuffler import ShuffSpec
from consts import allWellNames, TRODES_SAMPLING_RATE
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP


# Some stuff for allowing me to run lambda functions in parallel
_poolWorkerFunc = None


def poolWorkerFuncWrapper(args):
    return _poolWorkerFunc(args)


def poolWorkerInit(func):
    global _poolWorkerFunc
    _poolWorkerFunc = func


class TimeMeasure():
    """
    A measure that has a value for some series of time periods. Is supposed to be a generalization
    of TrialMeasure, although comparing home and away in this class might not be easy to do,
    so for now will focus on comparing different conditions.
    """

    @dataclass
    class TimePeriod:
        inProbe: bool
        start_posIdx: int
        end_posIdx: int
        xval: float
        category: Optional[str]
        data: Any

    @staticmethod
    def sweepingTimePerodsFunction(windowSize: float, stepSize: float, inProbe: bool) -> Callable[[BTSession], List[TimePeriod]]:
        """
        Returns a function that takes a session and returns a list of time periods
        :param windowSize: the size of the window in seconds
        :param stepSize: the step size in seconds
        :param inProbe: if True, the time periods span the duration of the probe, otherwise they span the duration of the task
        """
        def timePeriodsFunc(sesh: BTSession) -> List[Tuple[int, int, float, None]]:
            tseconds = sesh.probePos_secs if inProbe else sesh.btPos_secs
            t0_secs = np.arange(0, tseconds[-1] - windowSize, stepSize)
            t1_secs = t0_secs + windowSize

            t0 = np.searchsorted(tseconds, t0_secs)
            t1 = np.searchsorted(tseconds, t1_secs)
            xvals = t1_secs

            return [TimeMeasure.TimePeriod(inProbe, t0, t1, x, None, None) for t0, t1, x in zip(t0, t1, xvals)]
        return timePeriodsFunc

    @staticmethod
    def _trialTimePeriodsFunction(trialInterval, sesh: BTSession) -> List[TimeMeasure.TimePeriod]:
        trialPosIdxs = sesh.getAllTrialPosIdxs()
        trialPosIdxs = trialPosIdxs[trialInterval[0]:trialInterval[1], :]
        tidxOffset = trialInterval[0] if trialInterval[0] is not None else 0
        return [TimeMeasure.TimePeriod(False, t0, t1, ii + tidxOffset, "home" if ii % 2 == 0 else "away", ii + tidxOffset) for ii, (t0, t1) in enumerate(trialPosIdxs)]
        # cutoff = np.random.randint(4, len(trialPosIdxs) - 4)
        # return [TimeMeasure.TimePeriod(False, t0, t1, ii, f"{'home' if ii % 2 == 0 else 'away'}{0 if ii < cutoff else 1}", ii) for ii, (t0, t1) in enumerate(trialPosIdxs)]

    @staticmethod
    def trialTimePeriodsFunction(trialInterval=(None, None)) -> List[TimePeriod]:
        """
        Returns a function that takes a session and returns a list of time periods corresponding to the trials
        The data category is home or away and the data field is the trial index (including away trials)
        :param sesh: the session to get the time periods from
        """
        return partial(TimeMeasure._trialTimePeriodsFunction, trialInterval)

    @staticmethod
    def _earlyVsLateTrialsFunction(earlyRange, lateRange, sesh: BTSession) -> List[TimeMeasure.TimePeriod]:
        trialPosIdxs = sesh.getAllTrialPosIdxs()
        ret = []
        for ii, (t0, t1) in enumerate(trialPosIdxs):
            if ii % 2 == 1:
                # away trial, skip
                continue
            trialIndex = ii // 2
            if trialIndex >= earlyRange[0] and trialIndex < earlyRange[1]:
                ret.append(TimeMeasure.TimePeriod(
                    False, t0, t1, trialIndex, "early", ii))
            elif trialIndex >= lateRange[0] and trialIndex < lateRange[1]:
                ret.append(TimeMeasure.TimePeriod(
                    False, t0, t1, trialIndex, "late", ii))
            else:
                pass
        return ret

    @staticmethod
    def earlyVsLateTrials(earlyRange=(1, 4), lateRange=(5, 8)) -> List[TimePeriod]:
        """
        Returns a function that takes a session and returns a list of time periods corresponding to the early and late home trials
        data field is the trial index (including away trials)
        :param sesh: the session to get the time periods from
        """
        return partial(TimeMeasure._earlyVsLateTrialsFunction, earlyRange, lateRange)

    T = TypeVar('T')

    def __init__(self, name: str,
                 measureFunc: Callable[[BTSession, int, int, bool, T], float],
                 sessionList: List[BTSession],
                 timePeriodsGenerator: Callable[[BTSession], List[TimePeriod]] |
                 Tuple[float, float, bool] = (30, 5, False),
                 parallelize: bool = False,
                 runImmediately: bool = True):
        """
        :param name: the name of the measure
        :param measureFunc: a function that takes a session, a start posIdx, an end posIdx, a bool indicating whether posIdxs are in probe,
            and a value of type T and returns a float
        :param sessionList: the list of sessions to measure
        :param timePeriodsGenerator: a function that takes a session and returns a list of time periods to measure.
            The data field of each time period can be anything, and is passed to the measureFunc.
            For example, if TimeMeasure.trialTimePeriodsFunction is used, then the data value is the trial index.
            Alternatively, a tuple of (windowSize, stepSize, inProbe) can be passed, in which case those arguments are
            passed to TimeMeasure.sweepingTimePerodsFunction to generate the time periods.
        """
        if parallelize:
            raise NotImplementedError("Parallelization not implemented yet")

        self.name = name
        self.measureFunc = measureFunc
        self.sessionList = sessionList
        self.timePeriodsGenerator = timePeriodsGenerator
        self.parallelize = parallelize

        if runImmediately:
            self.runMeasureFunc()
        else:
            self.valid = False

    def runMeasureFunc(self):
        self.valid = True

        if isinstance(self.timePeriodsGenerator, tuple):
            self.timePeriodsGenerator = TimeMeasure.sweepingTimePerodsFunction(
                *self.timePeriodsGenerator)

        allTimePeriods = [self.timePeriodsGenerator(
            sesh) for sesh in self.sessionList]
        maxNumTimePeriods = max([len(tps) for tps in allTimePeriods])

        measure = []
        self.measure2d = np.empty((len(self.sessionList), maxNumTimePeriods))
        self.measure2d[:, :] = np.nan
        sessionIdx = []
        sessionCondition = []
        posIdxRange = []
        xvals = []
        self.xvals2d = np.empty((len(self.sessionList), maxNumTimePeriods))
        self.xvals2d[:, :] = np.nan
        timePeriodCategory = []
        self.category2d = np.empty(
            (len(self.sessionList), maxNumTimePeriods), dtype=object)
        self.category2d[:, :] = "UNASSIGNED"

        for si, sesh in enumerate(self.sessionList):
            timePeriods = allTimePeriods[si]
            for ti, tp in enumerate(timePeriods):
                val = self.measureFunc(
                    sesh, tp.start_posIdx, tp.end_posIdx, tp.inProbe, tp.data)
                measure.append(val)
                self.measure2d[si, ti] = val
                sessionIdx.append(si)
                sessionCondition.append(
                    "SWR" if sesh.isRippleInterruption else "Ctrl")
                posIdxRange.append(
                    (tp.inProbe, tp.start_posIdx, tp.end_posIdx))
                xvals.append(tp.xval)
                self.xvals2d[si, ti] = tp.xval
                timePeriodCategory.append(tp.category)
                self.category2d[si, ti] = tp.category

        self.valMin = np.nanmin(measure)
        self.valMax = np.nanmax(measure)
        self.measureDf = pd.DataFrame({"sessionIdx": sessionIdx,
                                       "val": measure,
                                       "condition": sessionCondition,
                                       "posIdxRange": posIdxRange,
                                       "timePeriodCategory": timePeriodCategory,
                                       "xval": xvals})

        if all([v is None for v in timePeriodCategory]):
            self.hasCategories = False
        else:
            self.hasCategories = True
            self.uniqueCategories: Set[str] = set(timePeriodCategory)
            # self.numCategories = len(self.uniqueCategories)
            self.withinCategoryAvgs = self.measureDf.groupby(["sessionIdx", "timePeriodCategory"]
                                                             ).agg(withinAvg=("val", "mean"))
            self.withinCategoryAvgs["withoutAvg"] = np.nan
            for cat in self.uniqueCategories:
                self.measureDf[f"CAT_IS_{cat}"] = self.measureDf["timePeriodCategory"] == cat
                withoutCategoryAvgs = self.measureDf.groupby(["sessionIdx", f"CAT_IS_{cat}"]).agg(
                    withoutAvg=("val", "mean"))
                withoutCategoryAvgs.rename(
                    index={False: cat}, level=1, inplace=True)
                self.withinCategoryAvgs.loc[pd.IndexSlice[:, cat], "withoutAvg"] = withoutCategoryAvgs.xs(
                    cat, level=1)["withoutAvg"].to_numpy()
            self.withinCategoryAvgs["withinSessionDiff"] = self.withinCategoryAvgs["withinAvg"] - \
                self.withinCategoryAvgs["withoutAvg"]
            sessionConditionDf = self.measureDf.groupby(
                "sessionIdx")[["condition"]].nth(0)
            self.withinCategoryAvgs = self.withinCategoryAvgs.join(
                sessionConditionDf, on="sessionIdx")
            diffs = self.withinCategoryAvgs["withinSessionDiff"].to_numpy()
            self.diffMin = np.nanmin(diffs)
            self.diffMax = np.nanmax(diffs)

    def makeFigures(self,
                    plotManager: PlotManager,
                    plotFlags: str | List[str] = [
                        "noteveryperiod", "noteverysession"],
                    subFolder: bool = True,
                    runStats: bool = True,
                    excludeFromCombo: bool = False,
                    numShuffles: int = 100):

        figName = self.name.replace(" ", "_")
        statsId = self.name + "_"
        figPrefix = "" if subFolder else figName + "_"
        if subFolder:
            plotManager.pushOutputSubDir("TiM_" + figName)

        allPossibleFlags = ["measure", "measureByCat", "measureWithinCat", "diff",
                            "everyperiod", "everysession", "averages"]

        if isinstance(plotFlags, str):
            if plotFlags == "all":
                plotFlags = allPossibleFlags
            else:
                plotFlags = [plotFlags]
        else:
            # so list passed in isn't modified
            plotFlags = [v for v in plotFlags]

        if len(plotFlags) > 0 and plotFlags[0].startswith("not"):
            if not all([v.startswith("not") for v in plotFlags]):
                raise ValueError(
                    "if one plot flag starts with 'not', all must")
            plotFlags = [v[3:] for v in plotFlags]
            plotFlags = list(set(allPossibleFlags) - set(plotFlags))

        if "measure" in plotFlags:
            plotFlags.remove("measure")

            dotColors = ["orange" if c ==
                         "SWR" else "cyan" for c in self.measureDf["condition"]]
            with plotManager.newFig(figName, excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName) as pc:
                violinPlot(pc.ax, self.measureDf["val"], categories=self.measureDf["condition"],
                           dotColors=dotColors, axesNames=["Condition", self.name])

                if runStats:
                    pc.yvals[figName] = self.measureDf["val"].to_numpy()
                    pc.categories["condition"] = self.measureDf["condition"].to_numpy(
                    )
                    pc.immediateShuffles.append((
                        [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

        if "measureByCat" in plotFlags:
            plotFlags.remove("measureByCat")

            if self.hasCategories:
                dotColors = ["orange" if c ==
                             "SWR" else "cyan" for c in self.measureDf["condition"]]
                with plotManager.newFig(figName + "_byCat", excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName + "_byCat") as pc:
                    violinPlot(pc.ax, self.measureDf["val"], categories2=self.measureDf["timePeriodCategory"],
                               categories=self.measureDf["condition"], dotColors=dotColors,
                               axesNames=["Condition", self.name, "Category"])

                    if runStats:
                        pc.yvals[figName] = self.measureDf["val"].to_numpy()
                        pc.categories["condition"] = self.measureDf["condition"].to_numpy(
                        )
                        pc.categories["timePeriodCategory"] = self.measureDf["timePeriodCategory"].to_numpy(
                        )

        if "measureWithinCat" in plotFlags:
            plotFlags.remove("measureWithinCat")

            if self.hasCategories:
                uc = sorted(self.uniqueCategories)

                for cat in uc:
                    catMeasureDf = self.measureDf[self.measureDf["timePeriodCategory"] == cat]
                    dotColors = ["orange" if c ==
                                 "SWR" else "cyan" for c in catMeasureDf["condition"]]
                    catReplace = cat.replace(" ", "_")
                    with plotManager.newFig(f"{figName}_withinCat_{catReplace}", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figName}_withinCat_{catReplace}") as pc:
                        violinPlot(pc.ax, catMeasureDf["val"],
                                   categories=catMeasureDf["condition"], dotColors=dotColors,
                                   axesNames=["Condition", self.name])

                        if runStats:
                            pc.yvals[figName] = catMeasureDf["val"].to_numpy()
                            pc.categories["condition"] = catMeasureDf["condition"].to_numpy(
                            )
                            pc.immediateShuffles.append((
                                [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

        if "diff" in plotFlags:
            plotFlags.remove("diff")

            if self.hasCategories:
                for cat in self.uniqueCategories:
                    catReplace = cat.replace(" ", "_")
                    with plotManager.newFig(f"{figName}_cat_{catReplace}_diff", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figName}_cat_{catReplace}_diff") as pc:
                        catVals = self.withinCategoryAvgs.xs(cat, level=1)
                        dotColors = ["orange" if c ==
                                     "SWR" else "cyan" for c in catVals["condition"]]
                        violinPlot(pc.ax, catVals["withinSessionDiff"], categories=catVals["condition"],
                                   dotColors=dotColors, axesNames=["Contidion", self.name + " within-session difference"])

                        if runStats:
                            pc.yvals[figName] = catVals["withinSessionDiff"].to_numpy()
                            pc.categories["condition"] = catVals["condition"].to_numpy(
                            )

                            pc.immediateShuffles.append((
                                [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

        if "averages" in plotFlags:
            plotFlags.remove("averages")
            swrIdx = np.array(
                [sesh.isRippleInterruption for sesh in self.sessionList])
            ctrlIdx = np.array([not v for v in swrIdx])

            # If all the rows of xvals2d are equal, just make one array of xvals
            nonNanXRow = np.nanmax(self.xvals2d, axis=0)
            if np.all(np.isclose(self.xvals2d, nonNanXRow) | np.isnan(self.xvals2d)):
                xvals = nonNanXRow
                swrXVals = xvals
                ctrlXVals = xvals
                xsAreSame = True
            else:
                xvals = self.xvals2d
                swrXVals = self.xvals2d[swrIdx, :]
                ctrlXVals = self.xvals2d[ctrlIdx, :]
                xsAreSame = False

            with plotManager.newFig(figName + "_avgs_all", excludeFromCombo=excludeFromCombo) as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d, xvals, avgColor="grey")

            with plotManager.newFig(figName + "_avgs_byCond", excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName + "_avgs_byCond") as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[swrIdx, :], swrXVals, avgColor="orange", label="SWR",
                    spread="sem")
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[ctrlIdx, :], ctrlXVals, avgColor="cyan", label="Ctrl",
                    spread="sem")
                pc.ax.legend()

                if runStats and xsAreSame:
                    for xi, x in enumerate(xvals):
                        pc.yvals[f"{figName}_avgs_byCond_{x}"] = self.measure2d[:, xi]
                        # print(xi, x, self.measure2d[:, xi].shape)
                    pc.categories["condition"] = np.array(
                        ["SWR" if v else "Ctrl" for v in swrIdx])
                    pc.immediateShuffles.append((
                        [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

            if self.hasCategories:
                cmap = mpl.colormaps["Dark2"]
                uc = sorted(self.uniqueCategories)
                with plotManager.newFig(figName + "_avgs_byCategory", excludeFromCombo=excludeFromCombo) as pc:
                    for cat in uc:
                        m = self.measure2d.copy()
                        m[self.category2d != cat] = np.nan
                        c = cmap(uc.index(cat))
                        plotIndividualAndAverage(
                            pc.ax, m, xvals, avgColor=c, individualColor=c, label=cat, spread="sem",
                            interpolateOverNans=True)
                    pc.ax.legend()

                for cat in uc:
                    with plotManager.newFig(f"{figName}_avgs_cat_{cat}_vsnon", excludeFromCombo=excludeFromCombo) as pc:
                        m = self.measure2d.copy()
                        m[self.category2d != cat] = np.nan
                        c = cmap(uc.index(cat))
                        plotIndividualAndAverage(pc.ax, m, xvals,
                                                 avgColor=c, individualColor=c, label=cat, spread="sem",
                                                 interpolateOverNans=True)
                        m = self.measure2d.copy()
                        m[self.category2d == cat] = np.nan
                        c = cmap(len(uc))
                        plotIndividualAndAverage(pc.ax, m, xvals,
                                                 avgColor=c, individualColor=c, label=f"not {cat}", spread="sem",
                                                 interpolateOverNans=True)
                        pc.ax.legend()

                for cat in uc:
                    with plotManager.newFig(f"{figName}_avgs_cat_{cat}", excludeFromCombo=excludeFromCombo) as pc:
                        m = self.measure2d.copy()
                        m[self.category2d != cat] = np.nan
                        c = cmap(uc.index(cat))
                        plotIndividualAndAverage(pc.ax, m, xvals,
                                                 avgColor=c, individualColor=c, label=cat, spread="sem",
                                                 interpolateOverNans=True)
                        pc.ax.legend()

                for cat in uc:
                    with plotManager.newFig(f"{figName}_avgs_cat_{cat}_byCond", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figName}_avgs_cat_{cat}_byCond") as pc:
                        m = self.measure2d.copy()
                        m[self.category2d != cat] = np.nan
                        swrm = m[swrIdx, :]
                        swrx = swrXVals
                        plotIndividualAndAverage(
                            pc.ax, swrm, swrx, avgColor="orange", label="SWR", spread="sem",
                            interpolateOverNans=True)
                        ctrlm = m[ctrlIdx, :]
                        ctrlx = ctrlXVals
                        plotIndividualAndAverage(
                            pc.ax, ctrlm, ctrlx, avgColor="cyan", label="Ctrl", spread="sem",
                            interpolateOverNans=True)
                        pc.ax.legend()

                        if runStats and xsAreSame:
                            for xi, x in enumerate(xvals):
                                pc.yvals[f"{figName}_avgs_cat_{cat}_byCond_{x}"] = m[:, xi]
                            pc.categories["condition"] = np.array(
                                ["SWR" if v else "Ctrl" for v in swrIdx])
                            pc.immediateShuffles.append((
                                [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

        if "everyperiod" in plotFlags:
            plotFlags.remove("everyperiod")

            cmap = mpl.colormaps["coolwarm"]
            for si, sesh in enumerate(self.sessionList):
                thisSession = self.measureDf[self.measureDf["sessionIdx"]
                                             == si].sort_values("posIdxRange").reset_index(drop=True)
                vals = thisSession["val"].to_numpy()
                normvals = (vals - self.valMin) / (self.valMax - self.valMin)

                if subFolder:
                    thisFigName = f"{figPrefix}allTimePeriods_{sesh.name}"
                else:
                    thisFigName = figName + "_allTimePeriods"
                    plotManager.pushOutputSubDir(sesh.name)

                ncols = np.ceil(np.sqrt(len(thisSession.index))).astype(int)
                nrows = np.ceil(len(thisSession.index) / ncols).astype(int)
                with plotManager.newFig(thisFigName, subPlots=(nrows, ncols), excludeFromCombo=excludeFromCombo) as pc:
                    for i in range(len(thisSession.index), nrows * ncols):
                        blankPlot(pc.axs.flat[i])

                    for index, row in thisSession.iterrows():
                        ax = pc.axs.flat[index]
                        assert isinstance(ax, Axes)
                        tp = row["posIdxRange"]

                        c = "orange" if row["condition"] == "SWR" else "cyan"
                        setupBehaviorTracePlot(ax, sesh, outlineColors=c)
                        inProbe = tp[0]
                        t0 = tp[1]
                        t1 = tp[2]
                        xs = sesh.probePosXs[t0: t1] if inProbe else sesh.btPosXs[t0: t1]
                        ys = sesh.probePosYs[t0: t1] if inProbe else sesh.btPosYs[t0: t1]
                        ax.plot(xs, ys, c="black")
                        ax.set_facecolor(cmap(normvals[index]))
                        if self.hasCategories:
                            ax.set_title(
                                f"{vals[index]} ({row['timePeriodCategory']})")
                        else:
                            ax.set_title(str(vals[index]))

                    pc.figure.colorbar(mappable=ScalarMappable(
                        norm=Normalize(self.valMin, self.valMax), cmap=cmap), ax=ax)

                if not subFolder:
                    plotManager.popOutputSubDir()

        if "everysession" in plotFlags:
            plotFlags.remove("everysession")
            cmap = mpl.colormaps["coolwarm"]
            wellSize = mpl.rcParams['lines.markersize']**2 / 4

            nCols = max([len(self.measureDf[self.measureDf["sessionIdx"] == si])
                         for si in range(len(self.sessionList))])

            with plotManager.newFig(figName + "_allTimePeriods_allSessions",
                                    subPlots=(len(self.sessionList), nCols), figScale=0.3, excludeFromCombo=excludeFromCombo) as pc:
                for si, sesh in enumerate(self.sessionList):
                    print(
                        f"every session plot: ({si} / {len(self.sessionList)})")
                    thisSession = self.measureDf[self.measureDf["sessionIdx"]
                                                 == si].sort_values("posIdxRange").reset_index(drop=True)
                    vals = thisSession["val"].to_numpy()
                    normvals = (vals - self.valMin) / \
                        (self.valMax - self.valMin)

                    for i in range(len(thisSession.index), nCols):
                        blankPlot(pc.axs[si, i])

                    for index, row in thisSession.iterrows():
                        ax = pc.axs[si, index]
                        assert isinstance(ax, Axes)
                        tp = row["posIdxRange"]

                        c = "orange" if sesh.isRippleInterruption else "cyan"
                        setupBehaviorTracePlot(
                            ax, sesh, outlineColors=c, wellSize=wellSize)
                        inProbe = tp[0]
                        t0 = tp[1]
                        t1 = tp[2]
                        xs = sesh.probePosXs[t0: t1] if inProbe else sesh.btPosXs[t0: t1]
                        ys = sesh.probePosYs[t0: t1] if inProbe else sesh.btPosYs[t0: t1]
                        ax.plot(xs, ys, c="black")
                        ax.set_facecolor(cmap(normvals[index]))
                        if self.hasCategories:
                            ax.set_title(
                                f"{vals[index]} ({row['timePeriodCategory']})")
                        else:
                            ax.set_title(str(vals[index]))

                    pc.figure.colorbar(mappable=ScalarMappable(
                        norm=Normalize(self.valMin, self.valMax), cmap=cmap), ax=pc.axs[si, -1])

                    pc.axs[si, 0].set_title(sesh.name)

        with open(os.path.join(plotManager.fullOutputDir, "processed.txt"), "w") as f:
            f.write(plotManager.infoFileFullName)

        if subFolder:
            plotManager.popOutputSubDir()

        if len(plotFlags) > 0:
            print(f"Warning: unused plot flags: {plotFlags}")


class TrialMeasure():
    """
    A measure that has a value for every trial during a task, such as trial duration.

    Callbacks:
    measureFunc(session, trialStart_posIdx, trialEnd_posIdx, trial type ("home" | "away")) -> measure value

    optional trial filter
    trialFilter(session, trial type ("home" | "away"), trial index, trialStart_posIdx, trialEnd_posIdx, well number)
        -> True if trial should be included, False if if should be skipped
    trial index is independently tracked for home and away. So the first home and first away trials
        will both have index 0
    These functions are not necessarily called in the order in which they occur during the task
    """

    def __init__(self, name: str,
                 measureFunc: Callable[[BTSession, int, int, str], float],
                 sessionList: List[BTSession],
                 trialFilter: None | Callable[[
                     BTSession, str, int, int, int, int], bool] = None,
                 runImmediately=True):
        self.name = name
        self.measureFunc = measureFunc
        self.sessionList = sessionList
        self.trialFilter = trialFilter

        if runImmediately:
            self.runMeasureFunc()
        else:
            self.valid = False

    def runMeasureFunc(self):
        self.valid = True
        measure = []
        self.measure2d = np.empty((len(self.sessionList), 25))
        self.measure2d[:, :] = np.nan
        trialType = []
        conditionColumn = []
        dotColors = []
        sessionIdxColumn = []
        wasExcluded = []
        trial_posIdx = []

        for si, sesh in enumerate(self.sessionList):
            t1 = np.array(sesh.homeRewardEnter_posIdx)
            t0 = np.array(np.hstack(([0], sesh.awayRewardExit_posIdx)))
            if not sesh.endedOnHome:
                t0 = t0[0:-1]
            assert len(t1) == len(t0)

            for ii, (i0, i1) in enumerate(zip(t0, t1)):
                trialType.append("home")
                conditionColumn.append(
                    "SWR" if sesh.isRippleInterruption else "Ctrl")
                # dotColors.append(si)
                dotColors.append(
                    "orange" if sesh.isRippleInterruption else "cyan")
                sessionIdxColumn.append(si)
                trial_posIdx.append((i0, i1))

                if self.trialFilter is not None and not self.trialFilter(sesh, "home", ii, i0, i1, sesh.homeWell):
                    wasExcluded.append(1)
                    measure.append(np.nan)
                else:
                    val = self.measureFunc(sesh, i0, i1, "home")
                    measure.append(val)
                    self.measure2d[si, ii*2] = val
                    wasExcluded.append(0)

            # away trials
            t1 = np.array(sesh.awayRewardEnter_posIdx)
            t0 = np.array(sesh.homeRewardExit_posIdx)
            if sesh.endedOnHome:
                t0 = t0[0:-1]
            assert len(t1) == len(t0)

            for ii, (i0, i1) in enumerate(zip(t0, t1)):
                trialType.append("away")
                conditionColumn.append(
                    "SWR" if sesh.isRippleInterruption else "Ctrl")
                # dotColors.append(si)
                dotColors.append(
                    "orange" if sesh.isRippleInterruption else "cyan")
                sessionIdxColumn.append(si)
                trial_posIdx.append((i0, i1))

                if self.trialFilter is not None and not self.trialFilter(sesh, "away", ii, i0, i1, sesh.visitedAwayWells[ii]):
                    wasExcluded.append(1)
                    measure.append(np.nan)
                else:
                    val = self.measureFunc(sesh, i0, i1, "away")
                    measure.append(val)
                    self.measure2d[si, ii*2+1] = val
                    wasExcluded.append(0)

        self.valMin = np.nanmin(measure)
        self.valMax = np.nanmax(measure)
        self.trialDf = pd.DataFrame({"sessionIdx": sessionIdxColumn,
                                     "val": measure,
                                     "trialType": trialType,
                                     "condition": conditionColumn,
                                     "wasExcluded": wasExcluded,
                                     "trial_posIdx": trial_posIdx,
                                     "dotColor": dotColors})
        self.sessionDf = self.trialDf.groupby(
            "sessionIdx")[["condition", "dotColor", "sessionIdx"]].nth(0)

        # Groups home and away trials within a session, takes nanmean of each
        # Then takes difference of home and away values within each sesion
        # Finally, lines them up with sessionDf info
        self.withinSessionDiffs = self.trialDf.groupby(["sessionIdx", "trialType"]).agg(
            withinSessionDiff=("val", "mean")).diff().xs("home", level="trialType").merge(self.sessionDf, on="sessionIdx")
        diffs = self.withinSessionDiffs["withinSessionDiff"].to_numpy()
        self.diffMin = np.nanmin(diffs)
        self.diffMax = np.nanmax(diffs)

    def makeFigures(self,
                    plotManager: PlotManager,
                    plotFlags: str | List[str] = "all",
                    subFolder: bool = True,
                    excludeFromCombo: bool = False,
                    runStats: bool = True,
                    numShuffles: int = 100):

        figName = self.name.replace(" ", "_")
        statsId = self.name + "_"
        if subFolder:
            plotManager.pushOutputSubDir("TrM_" + figName)

        allPossibleFlags = ["measure", "diff",
                            "everytrial", "everysession", "averages"]

        if isinstance(plotFlags, str):
            if plotFlags == "all":
                plotFlags = allPossibleFlags
            else:
                plotFlags = [plotFlags]
        else:
            # so list passed in isn't modified
            plotFlags = [v for v in plotFlags]

        if len(plotFlags) > 0 and plotFlags[0].startswith("not"):
            if not all([v.startswith("not") for v in plotFlags]):
                raise ValueError(
                    "if one plot flag starts with 'not', all must")
            plotFlags = [v[3:] for v in plotFlags]
            plotFlags = list(set(allPossibleFlags) - set(plotFlags))

        useTitle = True
        if np.all(np.isnan(self.trialDf["val"])):
            useTitle = False
            cantRun = ["measure", "diff", "averages"]
            for cr in cantRun:
                if cr in plotFlags:
                    plotFlags.remove(cr)

        if "measure" in plotFlags:
            plotFlags.remove("measure")
            with plotManager.newFig(figName, excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName) as pc:
                violinPlot(pc.ax, self.trialDf["val"], categories2=self.trialDf["trialType"],
                           categories=self.trialDf["condition"], dotColors=self.trialDf["dotColor"],
                           axesNames=["Condition", self.name, "Trial type"])
                #    category2Order=["home", "away"])

                if runStats:
                    pc.yvals[figName] = self.trialDf["val"].to_numpy()
                    pc.categories["trialType"] = self.trialDf["trialType"].to_numpy()
                    pc.categories["condition"] = self.trialDf["condition"].to_numpy()

        if "diff" in plotFlags:
            plotFlags.remove("diff")
            with plotManager.newFig(figName + "_diff", excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName + "_diff") as pc:
                violinPlot(pc.ax, self.withinSessionDiffs["withinSessionDiff"],
                           categories=self.withinSessionDiffs["condition"],
                           dotColors=self.withinSessionDiffs["dotColor"],
                           axesNames=["Contidion", self.name + " within-session difference"])

                if runStats:
                    pc.yvals[figName] = self.withinSessionDiffs["withinSessionDiff"].to_numpy(
                    )
                    pc.categories["condition"] = self.withinSessionDiffs["condition"].to_numpy(
                    )
                    pc.immediateShuffles.append((
                        [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

        if "averages" in plotFlags:
            plotFlags.remove("averages")
            xvalsAll = np.arange(self.measure2d.shape[1]) + 1
            xvalsHalf = np.arange(math.ceil(self.measure2d.shape[1] / 2)) + 1
            swrIdx = np.array(
                [sesh.isRippleInterruption for sesh in self.sessionList])
            ctrlIdx = np.array([not v for v in swrIdx])

            with plotManager.newFig(figName + "_byTrialAvgs_all", excludeFromCombo=excludeFromCombo) as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d, xvalsAll, avgColor="grey")
                pc.ax.set_xlim(1, len(xvalsAll))
                pc.ax.set_xticks(np.arange(0, len(xvalsAll), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_all_byCond", excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName + "_byTrialAvgs_all_byCond") as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[swrIdx, :], xvalsAll, avgColor="orange", label="SWR",
                    spread="sem", skipIndividuals=True)
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[ctrlIdx, :], xvalsAll, avgColor="cyan", label="Ctrl",
                    spread="sem", skipIndividuals=True)
                pc.ax.set_xlim(1, len(xvalsAll))
                pc.ax.legend()
                pc.ax.set_xticks(np.arange(0, len(xvalsAll), 2) + 1)

                if runStats:
                    for xi, x in enumerate(xvalsAll):
                        pc.yvals[f"{figName}_byTrialAvgs_all_byCond_{x}"] = self.measure2d[:, xi]
                        # print(xi, x, self.measure2d[:, xi].shape)
                    pc.categories["condition"] = np.array(
                        ["SWR" if v else "Ctrl" for v in swrIdx])
                    pc.immediateShuffles.append((
                        [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))
                    pc.immediateShufflePvalThreshold = 0.15

            with plotManager.newFig(figName + "_byTrialAvgs_all_byTrialType", excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName + "_byTrialAvgs_all_byTrialType") as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[:, ::2], xvalsHalf, avgColor="red", label="home",
                    spread="sem", skipIndividuals=True)
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[:, 1::2], xvalsHalf[:-1], avgColor="blue", label="away",
                    spread="sem", skipIndividuals=True)
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.legend()
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

                if runStats:
                    # print("home vs away")
                    for xi, x in enumerate(xvalsHalf[:-1]):
                        yv = np.hstack(
                            (self.measure2d[:, xi * 2], self.measure2d[:, xi * 2 + 1]))
                        # print(xi, x, yv.shape)
                        pc.yvals[f"{figName}_byTrialAvgs_all_byTrialType_{x}"] = yv
                    cats = np.array(["home"] * self.measure2d.shape[0] +
                                    ["away"] * self.measure2d.shape[0])
                    pc.categories["trialType"] = cats
                    # print(f"{ cats.shape=  }")

                    pc.immediateShuffles.append((
                        [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="trialType", value="home")], numShuffles))
                    pc.immediateShufflePvalThreshold = 0.15

            with plotManager.newFig(figName + "_byTrialAvgs_home", excludeFromCombo=excludeFromCombo) as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[:, ::2], xvalsHalf, avgColor="grey", skipIndividuals=True)
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_away", excludeFromCombo=excludeFromCombo) as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[:, 1::2], xvalsHalf[:-1], avgColor="grey", skipIndividuals=True)
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_home_byCond", excludeFromCombo=excludeFromCombo) as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[swrIdx, ::2], xvalsHalf, avgColor="orange", label="SWR",
                    spread="sem", skipIndividuals=True)
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[ctrlIdx, ::2], xvalsHalf, avgColor="cyan", label="Ctrl",
                    spread="sem", skipIndividuals=True)
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.legend()
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_away_byCond", excludeFromCombo=excludeFromCombo) as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[swrIdx, 1::2], xvalsHalf[:-1], avgColor="orange", label="SWR",
                    spread="sem", skipIndividuals=True)
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[ctrlIdx, 1::2], xvalsHalf[:-1], avgColor="cyan", label="Ctrl",
                    spread="sem", skipIndividuals=True)
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.legend()
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_ctrl_byTrialType", excludeFromCombo=excludeFromCombo) as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[ctrlIdx, ::2], xvalsHalf, avgColor="red", label="home",
                    spread="sem", skipIndividuals=True)
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[ctrlIdx, 1::2], xvalsHalf[:-1], avgColor="blue", label="away",
                    spread="sem", skipIndividuals=True)
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.legend()
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_SWR_byTrialType", excludeFromCombo=excludeFromCombo) as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[swrIdx, ::2], xvalsHalf, avgColor="red", label="home",
                    spread="sem", skipIndividuals=True)
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[swrIdx, 1::2], xvalsHalf[:-1], avgColor="blue", label="away",
                    spread="sem", skipIndividuals=True)
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.legend()
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

        if "everytrial" in plotFlags:
            plotFlags.remove("everytrial")
            cmap = mpl.colormaps["coolwarm"]
            for si, sesh in enumerate(self.sessionList):
                thisSession = self.trialDf[self.trialDf["sessionIdx"]
                                           == si].sort_values("trial_posIdx")
                tpis = np.array([list(v) for v in thisSession["trial_posIdx"]])
                vals = thisSession["val"].to_numpy()
                normvals = (vals - self.valMin) / (self.valMax - self.valMin)
                plotManager.pushOutputSubDir(sesh.name)

                ncols = len(sesh.visitedAwayWells) + 1
                with plotManager.newFig(figName + "_allTrials", subPlots=(2, ncols), excludeFromCombo=excludeFromCombo) as pc:
                    for ti in range(2*ncols):
                        # ai0 = ti % 2
                        # ai1 = ti // 2
                        # ax = pc.axs[ai0, ai1]
                        ax = pc.axs.flatten()[ti]

                        if ti >= tpis.shape[0]:
                            blankPlot(ax)
                            continue

                        assert isinstance(ax, Axes)
                        c = "orange" if self.withinSessionDiffs.loc[si,
                                                                    "condition"] == "SWR" else "cyan"
                        setupBehaviorTracePlot(ax, sesh, outlineColors=c)
                        t0 = tpis[ti, 0]
                        t1 = tpis[ti, 1]
                        ax.plot(sesh.btPosXs[t0:t1],
                                sesh.btPosYs[t0:t1], c="black")
                        ax.set_facecolor(cmap(normvals[ti]))
                        if useTitle:
                            ax.set_title(str(vals[ti]))

                    pc.figure.colorbar(mappable=ScalarMappable(
                        norm=Normalize(self.valMin, self.valMax), cmap=cmap), ax=ax)

                    # ax = pc.axs[1, -1]
                    ax = pc.axs.flatten()[-1]
                    assert isinstance(ax, Axes)
                    v = self.withinSessionDiffs.loc[si, "withinSessionDiff"]
                    ax.set_facecolor(
                        cmap((v - self.diffMin) / (self.diffMax - self.diffMin)))
                    c = "orange" if self.withinSessionDiffs.loc[si,
                                                                "condition"] == "SWR" else "cyan"
                    setupBehaviorTracePlot(
                        ax, sesh, outlineColors=c, showWells="")
                    ax.set_title(f"avg diff = {v}")
                    pc.figure.colorbar(mappable=ScalarMappable(
                        norm=Normalize(self.diffMin, self.diffMax), cmap=cmap), ax=ax)

                plotManager.popOutputSubDir()

        if "everysession" in plotFlags:
            plotFlags.remove("everysession")
            cmap = mpl.colormaps["coolwarm"]
            ncols = 14
            wellSize = mpl.rcParams['lines.markersize']**2 / 4
            with plotManager.newFig(figName + "_allTrials_allSessions",
                                    subPlots=(2*len(self.sessionList), ncols), figScale=0.3, excludeFromCombo=excludeFromCombo) as pc:
                for si, sesh in enumerate(self.sessionList):
                    print(si)
                    thisSession = self.trialDf[self.trialDf["sessionIdx"]
                                               == si].sort_values("trial_posIdx")
                    tpis = np.array([list(v)
                                    for v in thisSession["trial_posIdx"]])
                    vals = thisSession["val"].to_numpy()
                    normvals = (vals - self.valMin) / \
                        (self.valMax - self.valMin)

                    blankPlot(pc.axs[2*si, 0])
                    blankPlot(pc.axs[2*si+1, 0])

                    for ti in range(2 * ncols - 2):
                        ax = pc.axs.flatten()[ti]
                        if ti >= tpis.shape[0]:
                            blankPlot(ax)
                            continue

                        assert isinstance(ax, Axes)
                        c = "orange" if self.withinSessionDiffs.loc[si,
                                                                    "condition"] == "SWR" else "cyan"
                        setupBehaviorTracePlot(
                            ax, sesh, outlineColors=c, wellSize=wellSize)
                        t0 = tpis[ti, 0]
                        t1 = tpis[ti, 1]
                        ax.plot(sesh.btPosXs[t0:t1],
                                sesh.btPosYs[t0:t1], c="black")
                        ax.set_facecolor(cmap(normvals[ti]))
                        ax.set_title(str(vals[ti]))

                    pc.figure.colorbar(mappable=ScalarMappable(
                        norm=Normalize(self.valMin, self.valMax), cmap=cmap), ax=pc.axs[2*si+1, -1])

                    ax = pc.axs[2*si+1, 0]
                    assert isinstance(ax, Axes)
                    v = self.withinSessionDiffs.loc[si, "withinSessionDiff"]
                    ax.set_facecolor(
                        cmap((v - self.diffMin) / (self.diffMax - self.diffMin)))
                    c = "orange" if self.withinSessionDiffs.loc[si,
                                                                "condition"] == "SWR" else "cyan"
                    setupBehaviorTracePlot(
                        ax, sesh, outlineColors=c, showWells="")
                    ax.set_title(f"avg diff = {v}")
                    pc.figure.colorbar(mappable=ScalarMappable(
                        norm=Normalize(self.diffMin, self.diffMax), cmap=cmap), ax=ax)

                    pc.axs[2*si, 0].set_title(sesh.name)

        with open(os.path.join(plotManager.fullOutputDir, "processed.txt"), "w") as f:
            f.write(plotManager.infoFileFullName)

        if subFolder:
            plotManager.popOutputSubDir()

        if len(plotFlags) > 0:
            print(f"Warning: unused plot flags: {plotFlags}")


class SessionMeasure:
    def __init__(self, name: str,
                 measureFunc: Callable[[BTSession], float],
                 sessionList: List[BTSession],
                 parallelize: bool = True,
                 runImmediately=True) -> None:
        """
        A measurement that summarizes an entire session with a float.
        Comparison is done between control and SWR sessions
        :param name: name of the measure
        :param measureFunc: function that takes a BTSession and returns a float
        :param sessionList: list of sessions to measure
        :param parallelize: whether to parallelize the measure computation across sessions
        """
        self.name = name
        self.measureFunc = measureFunc
        self.sessionList = sessionList
        self.parallelize = parallelize

        if runImmediately:
            self.runMeasureFunc()
        else:
            self.valid = False

    def runMeasureFunc(self):
        self.valid = True
        self.dotColors = [
            "orange" if s.isRippleInterruption else "cyan" for s in self.sessionList]
        self.conditionBySession = np.array(
            ["SWR" if s.isRippleInterruption else "Ctrl" for s in self.sessionList])

        if self.parallelize:
            with Pool(None, initializer=poolWorkerInit, initargs=(self.measureFunc, )) as p:
                self.sessionVals: np.ndarray = p.map(
                    poolWorkerFuncWrapper, self.sessionList)
        else:
            self.sessionVals = np.array(
                [self.measureFunc(sesh) for sesh in self.sessionList])

    def makeFigures(self,
                    plotManager: PlotManager,
                    plotFlags: str | List[str] = "all",
                    everySessionBehaviorPeriod: Optional[BP | Callable[[
                        BTSession], BP]] = None,
                    everySessionBackground: Optional[Callable[[
                        BTSession], ArrayLike]] = None,
                    runStats: bool = True, subFolder: bool = True,
                    excludeFromCombo=False,
                    numShuffles: int = 100) -> None:
        """
        Make figures for this measure
        :param plotManager: plot manager to use
        :param plotFlags: "all" or list of plot flags to make. Possible flags are:
            "all": make all figures
            "violin": make violin plot of session measure divided by condition
            "everySession": make everysession figure
        :param everySessionBehaviorPeriod: behavior period to use for everysession path trace
        :param everySessionBackground: background to use for everysession plot
        :param runStats: whether to run stats
        :param subFolder: whether to put the figures in a subfolder
        :param excludeFromCombo: whether to exclude this measure from the combo figure
        """
        figName = self.name.replace(" ", "_")
        statsId = self.name + "_"
        if subFolder:
            plotManager.pushOutputSubDir("SM_" + figName)

        allPossibleFlags = ["violin", "everysession",
                            "bySessionIdx", "bySessionIdxByCond"]

        if isinstance(plotFlags, str):
            if plotFlags == "all":
                plotFlags = allPossibleFlags
            else:
                plotFlags = [plotFlags]
        else:
            # so list passed in isn't modified
            plotFlags = [v for v in plotFlags]

        if len(plotFlags) > 0 and plotFlags[0].startswith("not"):
            if not all([v.startswith("not") for v in plotFlags]):
                raise ValueError(
                    "if one plot flag starts with 'not', all must")
            plotFlags = [v[3:] for v in plotFlags]
            plotFlags = list(set(allPossibleFlags) - set(plotFlags))

        if "violin" in plotFlags:
            plotFlags.remove("violin")
            with plotManager.newFig(figName, excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName) as pc:
                violinPlot(pc.ax, self.sessionVals, categories=self.conditionBySession,
                           dotColors=self.dotColors, axesNames=["Condition", self.name])

                if runStats:
                    pc.yvals[figName] = self.sessionVals
                    pc.categories["condition"] = self.conditionBySession
                    pc.immediateShuffles.append((
                        [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

        if "everysession" in plotFlags:
            plotFlags.remove("everysession")

            if everySessionBackground is not None:
                valImgs = [everySessionBackground(
                    sesh) for sesh in self.sessionList]
                valMin = np.nanmin(valImgs)
                valMax = np.nanmax(valImgs)
            else:
                normVals = (self.sessionVals - np.nanmin(self.sessionVals)) / \
                    (np.nanmax(self.sessionVals) - np.nanmin(self.sessionVals))

            wellSize = mpl.rcParams['lines.markersize']**2 / 4
            ncols = int(np.ceil(np.sqrt(len(self.sessionList))))
            with plotManager.newFig(figName + "_every_session", subPlots=(ncols, ncols),
                                    figScale=0.6, excludeFromCombo=excludeFromCombo) as pc:
                for si, sesh in enumerate(self.sessionList):
                    ax = pc.axs[si // ncols, si % ncols]
                    assert isinstance(ax, Axes)

                    if everySessionBehaviorPeriod is not None:
                        if callable(everySessionBehaviorPeriod):
                            bp = everySessionBehaviorPeriod(sesh)
                        else:
                            bp = everySessionBehaviorPeriod

                        _, xs, ys = sesh.posDuringBehaviorPeriod(bp)
                        ax.plot(xs, ys, c="#deac7f", lw=0.5)
                        if len(xs) > 0:
                            ax.scatter(xs[-1], ys[-1], marker="*")

                    setupBehaviorTracePlot(
                        ax, sesh, wellSize=wellSize, showWells="HA")

                    for v in ax.spines.values():
                        v.set_zorder(0)
                    ax.set_title(f"{self.sessionVals[si]:.3f} {sesh.name}", fontdict={
                                 'fontsize': 6})

                    if everySessionBackground is not None:
                        im = ax.imshow(valImgs[si].T, cmap=mpl.colormaps["coolwarm"],
                                       vmin=valMin, vmax=valMax, interpolation="nearest",
                                       extent=(-0.5, 6.5, -0.5, 6.5), origin="lower")
                    else:
                        ax.set_facecolor(
                            mpl.colormaps["coolwarm"](normVals[si]))

                for si in range(len(self.sessionList), ncols * ncols):
                    ax = pc.axs[si // ncols, si % ncols]
                    blankPlot(ax)

                if everySessionBackground is not None:
                    plt.colorbar(im, ax=ax)

        if "bySessionIdx" in plotFlags:
            plotFlags.remove("bySessionIdx")

            with plotManager.newFig(figName + "_overTime", excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName + "_overTime") as pc:
                xvals = np.arange(len(self.sessionList))
                pc.ax.scatter(range(len(self.sessionList)),
                              self.sessionVals, color="black")
                pc.ax.set_ylabel(self.name)
                pc.ax.set_xlabel("Session Index")

                if runStats:
                    pc.yvals[figName] = self.sessionVals
                    pc.xvals["index"] = xvals
                    pc.immediateCorrelations.append(("index", figName))

        if "bySessionIdxByCond" in plotFlags:
            plotFlags.remove("bySessionIdxByCond")

            with plotManager.newFig(figName + "_overTime_byCond", excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName + "_overTime_byCond") as pc:
                xvals = np.arange(len(self.sessionList))
                pc.ax.scatter(range(len(self.sessionList)),
                              self.sessionVals, color=self.dotColors)
                pc.ax.set_ylabel(self.name)
                pc.ax.set_xlabel("Session Index")

                if runStats:
                    pc.yvals[figName] = self.sessionVals
                    pc.xvals["index"] = xvals
                    pc.categories["condition"] = self.conditionBySession
                    pc.immediateCorrelations.append(("index", figName))

        if len(plotFlags) > 0:
            print(f"Warning: unused plot flags: {plotFlags}")

        with open(os.path.join(plotManager.fullOutputDir, "processed.txt"), "w") as f:
            f.write(plotManager.infoFileFullName)

        if subFolder:
            plotManager.popOutputSubDir()

    def makeCorrelationFigures(self,
                               plotManager: PlotManager,
                               sm: SessionMeasure,
                               plotFlags: str | List[str] = "all",
                               runStats: bool = True,
                               subFolder: bool = True,
                               excludeFromCombo=False):
        figName = self.name.replace(
            " ", "_") + "_X_" + sm.name.replace(" ", "_")
        statsId = self.name + "_"
        figPrefix = "" if subFolder else figName + "_"
        dataName = self.name.replace(" ", "_")

        if subFolder:
            plotManager.pushOutputSubDir("SM_" + figName)

        allPossibleFlags = ["measure", "measureByCondition"]

        if isinstance(plotFlags, str):
            if plotFlags == "all":
                plotFlags = allPossibleFlags
            else:
                plotFlags = [plotFlags]
        else:
            # so list passed in isn't modified
            plotFlags = [v for v in plotFlags]

        if len(plotFlags) > 0 and plotFlags[0].startswith("not"):
            if not all([v.startswith("not") for v in plotFlags]):
                raise ValueError(
                    "if one plot flag starts with 'not', all must")
            plotFlags = [v[3:] for v in plotFlags]
            plotFlags = list(set(allPossibleFlags) - set(plotFlags))

        smVals = np.array(sm.sessionVals)
        selfVals = np.array(self.sessionVals)

        if "measure" in plotFlags:
            plotFlags.remove("measure")
            with plotManager.newFig(f"{figPrefix}measure", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}measure") as pc:
                pc.ax.scatter(smVals, selfVals)
                pc.ax.set_xlabel(sm.name)
                pc.ax.set_ylabel(self.name)

                if runStats:
                    pc.yvals[dataName] = selfVals
                    pc.xvals[sm.name] = smVals
                    pc.immediateCorrelations.append((sm.name, self.name))

        if "measureByCondition" in plotFlags:
            plotFlags.remove("measureByCondition")
            swrIdx = sm.conditionBySession == "SWR"
            ctrlIdx = sm.conditionBySession == "Ctrl"
            with plotManager.newFig(f"{figPrefix}measureByCondition", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}measureByCondition") as pc:
                pc.ax.scatter(
                    smVals[swrIdx], selfVals[swrIdx], label="SWR", color="orange")
                pc.ax.scatter(
                    smVals[ctrlIdx], selfVals[ctrlIdx], label="Ctrl", color="cyan")
                pc.ax.set_xlabel(sm.name)
                pc.ax.set_ylabel(self.name)
                pc.ax.legend()

                if runStats:
                    pc.yvals[dataName] = selfVals
                    pc.xvals[sm.name] = smVals
                    pc.categories["condition"] = sm.conditionBySession
                    pc.immediateCorrelations.append((sm.name, self.name))

        if len(plotFlags) > 0:
            print(f"Warning: unused plot flags: {plotFlags}")

        with open(os.path.join(plotManager.fullOutputDir, "corr_processed.txt"), "w") as f:
            f.write(plotManager.infoFileFullName)

        if subFolder:
            plotManager.popOutputSubDir()


class LocationMeasure():
    @staticmethod
    def measureAtLocation(vals: np.ndarray, pos: Tuple[float, float], smoothDist: Optional[float] = 0.5) -> ArrayLike:
        x = pos[0]
        y = pos[1]
        if x < -0.5 or x > 6.5 or y < -0.5 or y > 6.5:
            return np.nan

        pxc = np.digitize(x, np.linspace(-0.5, 6.5, vals.shape[0]))
        pyc = np.digitize(y, np.linspace(-0.5, 6.5, vals.shape[1]))
        if smoothDist is None or smoothDist == 0:
            return vals[pxc, pyc]

        pxc0 = np.digitize(
            x - smoothDist, np.linspace(-0.5, 6.5, vals.shape[0]))
        pyc0 = np.digitize(
            y - smoothDist, np.linspace(-0.5, 6.5, vals.shape[1]))
        pxc1 = np.digitize(
            x + smoothDist, np.linspace(-0.5, 6.5, vals.shape[0]))
        pyc1 = np.digitize(
            y + smoothDist, np.linspace(-0.5, 6.5, vals.shape[1]))

        pxlWidth = 7 / vals.shape[0]

        numer = 0
        denom = 0
        for x in range(pxc0, pxc1):
            for y in range(pyc0, pyc1):
                if np.isnan(vals[x, y]):
                    continue
                fac = smoothDist - \
                    np.linalg.norm([pxc - x, pyc - y]) * pxlWidth
                if fac < 0:
                    continue
                numer += vals[x, y] * fac
                denom += fac

        if denom == 0:
            return np.nan

        return numer / denom

    @staticmethod
    def measureAtWell(vals: ArrayLike, well: int, smoothDist: Optional[float] = 0.5) -> ArrayLike:
        return LocationMeasure.measureAtLocation(vals, getWellPosCoordinates(well), smoothDist)

    @staticmethod
    def measureValueAtHome(sesh: BTSession, vals: ArrayLike, smoothDist: Optional[float] = 0.5) -> ArrayLike:
        return LocationMeasure.measureAtWell(vals, sesh.homeWell, smoothDist=smoothDist)

    @staticmethod
    def measureValueAtAwayWells(sesh: BTSession, vals: ArrayLike, allVals: ArrayLike, seshIdx: int,
                                offWallOnly=True, smoothDist: Optional[float] = 0.5) -> ArrayLike:
        awVals = []
        for aw in sesh.visitedAwayWells:
            if offWallOnly and not offWall(aw):
                continue
            awVals.append(LocationMeasure.measureAtWell(
                vals, aw, smoothDist=smoothDist))
        return np.nanmean(awVals)

    @staticmethod
    def measureValueAtSymmetricWells(sesh: BTSession, vals: ArrayLike, allVals: ArrayLike, seshIdx: int,
                                     offWallOnly=True, smoothDist: Optional[float] = 0.5) -> ArrayLike:
        swVals = [LocationMeasure.measureAtWell(vals, rw, smoothDist=smoothDist)
                  for rw in getRotatedWells(sesh.homeWell)]
        return np.nanmean(swVals)

    @staticmethod
    def measureValueAtHomeOtherSeshs(sesh: BTSession, vals: ArrayLike, allVals: ArrayLike,  seshIdx: int,
                                     offWallOnly=True, smoothDist: Optional[float] = 0.5) -> ArrayLike:
        osVals = []
        for i in range(allVals.shape[0]):
            if i == seshIdx:
                continue
            osVals.append(LocationMeasure.measureAtWell(
                allVals[i], sesh.homeWell, smoothDist=smoothDist))
        return np.nanmean(osVals)

    @staticmethod
    def defaultCtrlLocations(sesh: BTSession, seshIdx: int, otherSessions: List[BTSession]) -> List[Tuple[List[Tuple[int, float, float]], str, Tuple[str, str, str]]]:
        ret = []
        aret = []
        for aw in sesh.visitedAwayWells:
            if offWall(aw):
                aret.append((seshIdx, *getWellPosCoordinates(aw)))
        ret.append((aret, "away", ("home", "away", "well type")))

        sret = []
        for sw in getRotatedWells(sesh.homeWell):
            sret.append((seshIdx, *getWellPosCoordinates(sw)))
        ret.append((sret, "symmetric", ("home", "symmetric", "well type")))

        osret = []
        for i in range(len(otherSessions)):
            if i == seshIdx:
                continue
            if otherSessions[i].homeWell == sesh.homeWell:
                continue
            osret.append((i, *getWellPosCoordinates(sesh.homeWell)))
        ret.append((osret, "othersessions", ("same", "other", "session type")))

        laterSessionsRet = []
        for i in range(seshIdx + 1, len(otherSessions)):
            if otherSessions[i].homeWell == sesh.homeWell:
                continue
            laterSessionsRet.append((i, *getWellPosCoordinates(sesh.homeWell)))
        ret.append((laterSessionsRet, "latersessions",
                   ("same", "later", "session type")))

        nextSessionRet = []
        if seshIdx + 1 < len(otherSessions):
            nextSessionRet.append(
                (seshIdx + 1, *getWellPosCoordinates(sesh.homeWell)))
        ret.append((nextSessionRet, "nextsession",
                   ("same", "next", "session type")))

        prevSessionRet = []
        if seshIdx > 0:
            prevSessionRet.append(
                (seshIdx - 1, *getWellPosCoordinates(sesh.homeWell)))
        ret.append((prevSessionRet, "prevsession",
                   ("same", "prev", "session type")))

        nextSessionNoAwayRet = []
        if seshIdx + 1 < len(otherSessions):
            if sesh.homeWell not in otherSessions[seshIdx + 1].visitedAwayWells:
                nextSessionNoAwayRet.append(
                    (seshIdx + 1, *getWellPosCoordinates(sesh.homeWell)))
        ret.append((nextSessionNoAwayRet, "nextsessionnoaway",
                   ("same", "next", "session type")))

        return ret

    @staticmethod
    def prevSessionCtrlLocations(sesh: BTSession, seshIdx: int, otherSessions: List[BTSession]) -> List[Tuple[List[Tuple[int, float, float]], str, Tuple[str, str, str]]]:
        ret = []
        prevSessionRet = []
        if seshIdx > 0:
            prevSessionRet.append(
                (seshIdx - 1, *getWellPosCoordinates(sesh.homeWell)))
        ret.append((prevSessionRet, "prevsession",
                   ("same", "prev", "session type")))
        return ret

    @staticmethod
    def prevSessionHomeLocation(sesh: BTSession) -> Tuple[float, float]:
        if sesh.prevSession is None:
            return getWellPosCoordinates(sesh.homeWell)
        return getWellPosCoordinates(sesh.prevSession.homeWell)

    def __init__(self, name: str, measureFunc: Callable[[BTSession], ArrayLike],
                 sessionList: List[BTSession],
                 sessionValLocation: Callable[[BTSession], Tuple[float,
                                                                 float]] = lambda s: getWellPosCoordinates(s.homeWell),
                 sessionCtrlLocations: Callable[[BTSession, int, List[BTSession]],
                                                Iterable[Tuple[List[Tuple[int, float, float]], str, Tuple[str, str, str]]]] = defaultCtrlLocations,
                 smoothDist: Optional[float] = 0.5,
                 distancePlotResolution: Tuple[int, int, int] = (15, 8, 15),
                 parallelize=True,
                 runImmediately=True) -> None:
        """
        :param name: name of the measure
        :param measureFunc: function that takes a session and returns a 2D array of values
                The array of values is indexed by x coordinate first. The returned array should be a square
                array with the same number of pixels in each dimension.
        :param sessionList: list of sessions to process
        :param sessionValLocation: function that takes a session and returns the location within the measureFunction return
                array where the session value should be measured. The location is given as a tuple of (x, y) coordinates.
        :param sessionCtrlLocations: function that takes a session, its index, and the list of all sessions, and returns
                a list of triplets. Each triplets is a list of locations, a name for the control, and three strings that are
                the names for the measure group, control group, and axis label. The locations are given as a list of tuples
                of (session index, x, y) coordinates. The name is a string that will be used to label the control values.
        :param smoothDist: distance over which to smooth the measure values, in feet
        :param distancePlotResolution: resolution of the distance plot, specified as a tuple of
                (number of distances to sample, number of samples at closest distance, number of samples at furthest distance)
        :param parallelize: whether to parallelize the processing of the sessions
        """
        self.name = name
        self.measureFunc = measureFunc
        self.sessionList = sessionList
        self.sessionValLocation = sessionValLocation
        self.sessionCtrlLocations = sessionCtrlLocations
        self.smoothDist = smoothDist
        self.distancePlotResolution = distancePlotResolution
        self.parallelize = parallelize

        if runImmediately:
            self.runMeasureFunc()
        else:
            self.valid = False

    def runMeasureFunc(self):
        self.valid = True
        self.sessionValsBySession = [None] * len(self.sessionList)
        self.dotColorsBySession = [None] * len(self.sessionList)
        self.conditionBySession = [None] * len(self.sessionList)
        self.sessionValLocations = [None] * len(self.sessionList)

        # self.controlVals[ctrlName][sessionIdx][ctrlIdx] = control value
        self.controlVals: Dict[str, List[List[float]]] = {}
        self.controlValMeans: Dict[str, List[float]] = {}
        self.controlValLabels: Dict[str, Tuple[str, str, str]] = {}
        self.dotColorsByCtrlVal: Dict[str, List[str]] = {}
        conditionByCtrlVal: Dict[str, List[str]] = {}

        if self.parallelize:
            with Pool(None, initializer=poolWorkerInit, initargs=(self.measureFunc, )) as p:
                self.measureValsBySession: List[np.ndarray] = p.map(
                    poolWorkerFuncWrapper, self.sessionList)
        else:
            self.measureValsBySession = [
                self.measureFunc(sesh) for sesh in self.sessionList]

        if np.isnan(self.measureValsBySession).all():
            # print(f"WARNING: all values for {self.name} are NaN. Not plotting.")
            self.valid = False
            return

        for si, sesh in enumerate(self.sessionList):
            self.measureValsBySession[si] = np.array(
                self.measureValsBySession[si])
            v = self.measureValsBySession[si]
            if v.shape[0] != v.shape[1]:
                raise ValueError("measureFunc must return a square array")
            self.sessionValLocations[si] = self.sessionValLocation(sesh)
            self.sessionValsBySession[si] = LocationMeasure.measureAtLocation(
                v, self.sessionValLocations[si], smoothDist=self.smoothDist)

            if sesh.isRippleInterruption:
                self.conditionBySession[si] = "SWR"
                self.dotColorsBySession[si] = "orange"
            else:
                self.conditionBySession[si] = "Ctrl"
                self.dotColorsBySession[si] = "cyan"

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"All-NaN (slice|axis) encountered")
            self.sessionValMin = np.nanmin(self.sessionValsBySession)
            self.sessionValMax = np.nanmax(self.sessionValsBySession)

        self.controlSpecsBySession = [self.sessionCtrlLocations(
            sesh, si, self.sessionList) for si, sesh in enumerate(self.sessionList)]

        for si, sesh in enumerate(self.sessionList):
            for ctrlLocs, ctrlName, ctrlLabels in self.controlSpecsBySession[si]:
                if ctrlName not in self.controlVals:
                    self.controlVals[ctrlName] = []
                    self.controlValMeans[ctrlName] = []
                    self.controlValLabels[ctrlName] = ctrlLabels
                    self.dotColorsByCtrlVal[ctrlName] = []
                    conditionByCtrlVal[ctrlName] = []
                else:
                    if self.controlValLabels[ctrlName] != ctrlLabels:
                        raise ValueError(
                            "control labels must be the same for each control")
                ctrlVals = []
                for ctrlLoc in ctrlLocs:
                    ctrlVals.append(LocationMeasure.measureAtLocation(
                        self.measureValsBySession[ctrlLoc[0]], (ctrlLoc[1], ctrlLoc[2]), smoothDist=self.smoothDist))

                self.controlVals[ctrlName].append(ctrlVals)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", r"Mean of empty slice")
                    cvalMean = np.nanmean(ctrlVals)
                self.controlValMeans[ctrlName].append(cvalMean)
                c = "orange" if sesh.isRippleInterruption else "cyan"
                self.dotColorsByCtrlVal[ctrlName] += [c] * len(ctrlVals)
                c = "SWR" if sesh.isRippleInterruption else "Ctrl"
                conditionByCtrlVal[ctrlName] += [c] * len(ctrlVals)

                if cvalMean > self.sessionValMax:
                    self.sessionValMax = cvalMean
                if cvalMean < self.sessionValMin:
                    self.sessionValMin = cvalMean

        self.distancePlotXVals = np.linspace(
            0, 7*np.sqrt(2), self.distancePlotResolution[0])
        self.numSamples = np.round(np.linspace(
            self.distancePlotResolution[1], self.distancePlotResolution[2], self.distancePlotResolution[0])).astype(int)
        self.measureValsByDistance = np.empty(
            (len(self.sessionList), len(self.distancePlotXVals)))
        self.measureValsByDistance[:] = np.nan
        self.controlMeasureValsByDistance: Dict[str, np.ndarray] = {}

        for si, sesh in enumerate(self.sessionList):
            for di, d in enumerate(self.distancePlotXVals):
                if d == 0:
                    self.measureValsByDistance[si,
                                               di] = self.sessionValsBySession[si]
                    continue

                nsamps = self.numSamples[di]
                vals = []
                for i in range(nsamps):
                    theta = i * 2 * np.pi / nsamps
                    x = d * np.cos(theta) + self.sessionValLocations[si][0]
                    y = d * np.sin(theta) + self.sessionValLocations[si][1]
                    vals.append(LocationMeasure.measureAtLocation(
                        self.measureValsBySession[si], (x, y), smoothDist=self.smoothDist))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", r"Mean of empty slice")
                    self.measureValsByDistance[si, di] = np.nanmean(vals)

        for ctrlName in self.controlVals:
            self.controlMeasureValsByDistance[ctrlName] = np.empty(
                (len(self.dotColorsByCtrlVal[ctrlName]), len(self.distancePlotXVals)))
            self.controlMeasureValsByDistance[ctrlName][:] = np.nan
            cidx = 0
            for si, sesh in enumerate(self.sessionList):
                for ctrlLocs, cname, ctrlLabels in self.controlSpecsBySession[si]:
                    if cname != ctrlName:
                        continue

                    for li, loc in enumerate(ctrlLocs):
                        for di, d in enumerate(self.distancePlotXVals):
                            if d == 0:
                                self.controlMeasureValsByDistance[ctrlName][cidx, di] = \
                                    self.controlVals[ctrlName][si][li]
                                continue

                            nsamps = self.numSamples[di]
                            vals = []
                            for i in range(nsamps):
                                theta = i * 2 * np.pi / nsamps
                                x = d * np.cos(theta) + loc[1]
                                y = d * np.sin(theta) + loc[2]
                                vals.append(LocationMeasure.measureAtLocation(
                                    self.measureValsBySession[loc[0]], (x, y), smoothDist=self.smoothDist))
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore", r"Mean of empty slice")
                                self.controlMeasureValsByDistance[ctrlName][cidx, di] = np.nanmean(
                                    vals)
                        cidx += 1

        self.measureValsBySession = np.array(self.measureValsBySession)
        self.sessionValsBySession = np.array(
            self.sessionValsBySession).astype(np.float64)
        self.conditionBySession = np.array(self.conditionBySession)
        self.conditionByCtrlVal: Dict[str, np.ndarray] = {}
        for ctrlName in conditionByCtrlVal:
            self.conditionByCtrlVal[ctrlName] = np.array(
                conditionByCtrlVal[ctrlName])

        self.valMin = np.nanmin(self.measureValsBySession)
        self.valMax = np.nanmax(self.measureValsBySession)

        if np.isnan(self.sessionValsBySession).all():
            # print(f"WARNING: all values for {self.name} are NaN. Not plotting.")
            self.valid = False
            return

    def makeFigures(self,
                    plotManager: PlotManager,
                    plotFlags: str | List[str] = "all",
                    everySessionBehaviorPeriod: Optional[BP | Callable[[
                        BTSession], BP]] = None,
                    runStats: bool = True,
                    subFolder: bool = True,
                    excludeFromCombo=False,
                    verbose=False,
                    numShuffles: int = 100) -> None:
        """
        :param plotManager: PlotManager to use to make figures
        :param plotFlags: "all" or list of strings indicating which figures to make. Possible values are:
            "measureByCondition": plot the measure values for each session, separated by condition
            "measureVsCtrl": plot the measure values for each session, with the control values for each session
            "measureVsCtrlByCondition": plot the measure values for each session, with the control values for each session, separated by condition
            "ctrlByCondition": plot just the measure values at control locations, separated by condition
            "diff": plot the difference between the measure values for each session and the mean control values within that session
            "measureByDistance": plot average measure values for each session as a function of distance from the session location
            "measureByDistanceByCondition": plot average measure values for each session as a function of distance from the session location, separated by condition
            "measureVsCtrlByDistance": plot average measure values for each session as a function of distance from the session location,
                                    compared to measure values as a function of distance from the control location
            "measureVsCtrlByDistanceByCondition": plot average measure values for each session as a function of distance from the session location, separated by condition,
                                                compared to measure values as a function of distance from the control location, separated by condition
            "everysession": plot the measure values for each session overlaid with behavior and well locations
            "everysessionoverlayatlocation": plot the measure values for each session overlaid on top of each other and summed to line up the session locations
            "everysessionoverlayatctrl": plot the measure values for each session overlaid on top of each other and summed to line up the control locations. Note each
                                        map is weighted to contribute the same (its values are divided by the number of control locations)
            "everysessionoverlayatlocationbycondition": plot the measure values for each session overlaid on top of each other and summed to line up the session locations.
                                        Two plots are made, one for each condition
            "everysessionoverlaydirect": plot the measure values for each session overlaid on top of each other and summed. Maps are not lined up by their session locations
            Optionally, any of the above can be preceded by "not" to exclude it from the list of figures to make
                If any flag starts with "not", all flags must start with "not", and any flag not excluded will be included
        :param everySessionBehaviorPeriod: if not None, behavior period to use for everySession plot
        :param runStats: if True, run stats on the data and add p value to output plot
        :param subFolder: if True, put figures in a subfolder named after the measure
        :param excludeFromCombo: if True, don't include this measure in the combo plot
        """

        if not self.valid:
            if verbose:
                print(f"WARNING: {self.name} is not valid. Not plotting.")
            return

        figName = self.name.replace(" ", "_")
        statsId = self.name + "_"
        # figPrefix = "" if subFolder else figName + "_"
        figPrefix = "" if subFolder else figName + "_"
        if subFolder:
            plotManager.pushOutputSubDir("LM_" + figName)

        allPossibleFlags = ["measureByCondition", "measureVsCtrl",
                            "measureVsCtrlByCondition", "diff",
                            "measureByDistance", "measureByDistanceByCondition",
                            "measureVsCtrlByDistance", "measureVsCtrlByDistanceByCondition",
                            "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl",
                            "everysessionoverlayatlocationbycondition",
                            "everysessionoverlaydirect"]

        if isinstance(plotFlags, str):
            if plotFlags == "all":
                plotFlags = allPossibleFlags
            else:
                plotFlags = [plotFlags]
        else:
            # so list passed in isn't modified
            plotFlags = [v for v in plotFlags]

        if len(plotFlags) > 0 and plotFlags[0].startswith("not"):
            if not all([v.startswith("not") for v in plotFlags]):
                raise ValueError(
                    "if one plot flag starts with 'not', all must")
            plotFlags = [v[3:] for v in plotFlags]
            plotFlags = list(set(allPossibleFlags) - set(plotFlags))

        if "measureByCondition" in plotFlags:
            plotFlags.remove("measureByCondition")
            if verbose:
                print(f"Plotting {self.name} by condition")

            with plotManager.newFig(f"{figPrefix}measureByCondition", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}measureByCondition") as pc:
                violinPlot(pc.ax, self.sessionValsBySession, self.conditionBySession,
                           dotColors=self.dotColorsBySession, axesNames=["Condition", self.name])
                #    categoryOrder=["SWR", "Ctrl"])
                pc.ax.set_title(self.name, fontdict={'fontsize': 6})

                if runStats:
                    pc.yvals[figName] = self.sessionValsBySession
                    pc.categories["condition"] = self.conditionBySession
                    pc.immediateShuffles.append((
                        [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

        if "measureVsCtrl" in plotFlags:
            plotFlags.remove("measureVsCtrl")
            if verbose:
                print(f"Plotting {self.name} vs control")

            for ctrlName in self.controlValLabels:
                vals = np.concatenate(
                    (self.sessionValsBySession, *self.controlVals[ctrlName]))
                dotColors = np.concatenate(
                    (self.dotColorsBySession, self.dotColorsByCtrlVal[ctrlName]))
                valCtrlCats = np.concatenate((np.full_like(self.sessionValsBySession, self.controlValLabels[ctrlName][0], dtype=object),
                                              np.full_like(self.dotColorsByCtrlVal[ctrlName], self.controlValLabels[ctrlName][1], dtype=object)))

                with plotManager.newFig(f"{figPrefix}ctrl_" + ctrlName, excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}ctrl_" + ctrlName) as pc:
                    violinPlot(pc.ax, vals, categories=valCtrlCats,
                               dotColors=dotColors, axesNames=[
                                   self.controlValLabels[ctrlName][2], self.name],
                               #    categoryOrder=self.controlValLabels[ctrlName][0:2],
                               dotColorLabels={"orange": "SWR", "cyan": "Ctrl"})
                    pc.ax.set_title(self.name + " vs " +
                                    ctrlName, fontdict={'fontsize': 6})

                    if runStats:
                        catName = self.controlValLabels[ctrlName][2].replace(
                            " ", "_")
                        pc.yvals[figName] = vals
                        pc.categories[catName] = valCtrlCats
                        pc.immediateShuffles.append((
                            [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName=catName, value=self.controlValLabels[ctrlName][0])], numShuffles))

        if "measureVsCtrlByCondition" in plotFlags:
            plotFlags.remove("measureVsCtrlByCondition")
            if verbose:
                print(f"Plotting {self.name} vs control by condition")

            for ctrlName in self.controlValLabels:
                vals = np.concatenate(
                    (self.sessionValsBySession, *self.controlVals[ctrlName]))
                conditionCats = np.concatenate(
                    (self.conditionBySession, self.conditionByCtrlVal[ctrlName]))
                dotColors = np.concatenate(
                    (self.dotColorsBySession, self.dotColorsByCtrlVal[ctrlName]))
                valCtrlCats = np.concatenate((np.full_like(self.sessionValsBySession, self.controlValLabels[ctrlName][0], dtype=object),
                                              np.full_like(self.conditionByCtrlVal[ctrlName], self.controlValLabels[ctrlName][1], dtype=object)))

                with plotManager.newFig(f"{figPrefix}ctrl_" + ctrlName + "_cond", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}ctrl_" + ctrlName + "_cond") as pc:
                    violinPlot(pc.ax, vals, conditionCats, categories2=valCtrlCats,
                               dotColors=dotColors, axesNames=[
                                   "Condition", self.name, self.controlValLabels[ctrlName][2]])
                    pc.ax.set_title(self.name + " vs " +
                                    ctrlName, fontdict={'fontsize': 6})

                    if runStats:
                        pc.yvals[figName] = vals
                        cat2Name = self.controlValLabels[ctrlName][2].replace(
                            " ", "_")
                        pc.categories["condition"] = conditionCats
                        pc.categories[cat2Name] = valCtrlCats
                        pc.immediateShuffles.append((
                            [ShuffSpec(shuffType=ShuffSpec.ShuffType.RECURSIVE_ALL, categoryName=cat2Name, value=None),
                             ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

        if "ctrlByCondition" in plotFlags:
            plotFlags.remove("ctrlByCondition")
            if verbose:
                print(f"Plotting {self.name} control by condition")

            for ctrlName in self.controlValLabels:
                vals = np.concatenate(self.controlVals[ctrlName])
                conditionCats = self.conditionByCtrlVal[ctrlName]
                dotColors = self.dotColorsByCtrlVal[ctrlName]

                with plotManager.newFig(f"{figPrefix}ctrl_{ctrlName}_solo_cond", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}ctrl_{ctrlName}_solo_cond") as pc:
                    violinPlot(pc.ax, vals, conditionCats, dotColors=dotColors,
                               axesNames=["Condition", self.name])
                    pc.ax.set_title(self.name + " at " + ctrlName +
                                    " by condition", fontdict={'fontsize': 6})

                    if runStats:
                        pc.yvals[figName] = vals
                        pc.categories["condition"] = conditionCats
                        pc.immediateShuffles.append((
                            [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

        if "diff" in plotFlags:
            plotFlags.remove("diff")
            if verbose:
                print(f"Plotting {self.name} difference")

            for ctrlName in self.controlValLabels:
                vals = self.sessionValsBySession - \
                    self.controlValMeans[ctrlName]
                if all(np.isnan(vals)):
                    continue
                with plotManager.newFig(f"{figPrefix}ctrl_" + ctrlName + "_diff", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}ctrl_" + ctrlName + "_diff") as pc:
                    violinPlot(pc.ax, vals, self.conditionBySession,
                               dotColors=self.dotColorsBySession, axesNames=[
                                   "Condition", self.name])
                    pc.ax.set_title(self.name + " vs " + ctrlName +
                                    " difference", fontdict={'fontsize': 6})

                    if runStats:
                        pc.yvals[figName] = vals
                        pc.categories["condition"] = self.conditionBySession
                        pc.immediateShuffles.append((
                            [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

        if "measureByDistance" in plotFlags:
            plotFlags.remove("measureByDistance")
            if verbose:
                print(f"Plotting {self.name} by distance")

            with plotManager.newFig(f"{figPrefix}by_distance", excludeFromCombo=excludeFromCombo) as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measureValsByDistance, self.distancePlotXVals)
                pc.ax.set_title(self.name, fontdict={'fontsize': 6})
                pc.ax.set_xlabel("Distance from well (ft)")

        if "measureByDistanceByCondition" in plotFlags:
            plotFlags.remove("measureByDistanceByCondition")
            if verbose:
                print(f"Plotting {self.name} by distance by condition")

            swrIdx = self.conditionBySession == "SWR"
            ctrlIdx = ~swrIdx

            with plotManager.newFig(f"{figPrefix}by_distance_by_condition", excludeFromCombo=excludeFromCombo) as pc:
                plotIndividualAndAverage(pc.ax, self.measureValsByDistance[swrIdx], self.distancePlotXVals,
                                         color="orange", label="SWR",
                                         spread="sem")
                plotIndividualAndAverage(pc.ax, self.measureValsByDistance[ctrlIdx], self.distancePlotXVals,
                                         color="cyan", label="Ctrl",
                                         spread="sem")
                pc.ax.set_title(self.name, fontdict={'fontsize': 6})
                pc.ax.set_xlabel("Distance from well (ft)")
                pc.ax.legend()

        if "measureVsCtrlByDistance" in plotFlags:
            plotFlags.remove("measureVsCtrlByDistance")
            if verbose:
                print(f"Plotting {self.name} vs control by distance")

            for ctrlName in self.controlVals:
                with plotManager.newFig(f"{figPrefix}vs_ctrl_" + ctrlName + "_by_distance", excludeFromCombo=excludeFromCombo) as pc:
                    plotIndividualAndAverage(pc.ax, self.measureValsByDistance, self.distancePlotXVals,
                                             color="red", label="vals",
                                             spread="sem")
                    plotIndividualAndAverage(pc.ax, self.controlMeasureValsByDistance[ctrlName], self.distancePlotXVals,
                                             color="blue", label="ctrl",
                                             spread="sem")
                    pc.ax.set_title(self.name + " vs " +
                                    ctrlName, fontdict={'fontsize': 6})
                    pc.ax.set_xlabel("Distance from well (ft)")
                    pc.ax.legend()

        if "measureVsCtrlByDistanceByCondition" in plotFlags:
            plotFlags.remove("measureVsCtrlByDistanceByCondition")
            if verbose:
                print(
                    f"Plotting {self.name} vs control by distance by condition")

            swrIdx = self.conditionBySession == "SWR"
            ctrlIdx = ~swrIdx

            for ctrlName in self.controlVals:
                swrIdxByCtrlVal = self.conditionByCtrlVal[ctrlName] == "SWR"
                ctrlIdxByCtrlVal = ~swrIdxByCtrlVal
                with plotManager.newFig(f"{figPrefix}vs_ctrl_" + ctrlName + "_by_distance_by_condition", excludeFromCombo=excludeFromCombo) as pc:
                    plotIndividualAndAverage(pc.ax, self.measureValsByDistance[swrIdx], self.distancePlotXVals,
                                             color="orange", label="SWR",
                                             spread="sem")
                    plotIndividualAndAverage(pc.ax, self.measureValsByDistance[ctrlIdx], self.distancePlotXVals,
                                             color="cyan", label="Ctrl",
                                             spread="sem")
                    plotIndividualAndAverage(pc.ax, self.controlMeasureValsByDistance[ctrlName][swrIdxByCtrlVal, :], self.distancePlotXVals,
                                             color="red", label="SWR ctrl",
                                             spread="sem")
                    plotIndividualAndAverage(pc.ax, self.controlMeasureValsByDistance[ctrlName][ctrlIdxByCtrlVal, :], self.distancePlotXVals,
                                             color="blue", label="Ctrl ctrl",
                                             spread="sem")
                    pc.ax.set_title(self.name + " vs " +
                                    ctrlName, fontdict={'fontsize': 6})
                    pc.ax.set_xlabel("Distance from well (ft)")
                    pc.ax.legend()

        if "everysession" in plotFlags:
            plotFlags.remove("everysession")
            if verbose:
                print(f"Plotting {self.name} every session")

            wellSize = mpl.rcParams['lines.markersize']**2 / 4
            ncols = int(np.ceil(np.sqrt(len(self.sessionList))))
            nrows = int(np.ceil(len(self.sessionList) / ncols))
            with plotManager.newFig(f"{figPrefix}every_session", subPlots=(nrows, ncols),
                                    figScale=0.6, excludeFromCombo=excludeFromCombo) as pc:
                for si, sesh in enumerate(self.sessionList):
                    ax = pc.axs.flat[si]
                    assert isinstance(ax, Axes)

                    if everySessionBehaviorPeriod is not None:
                        if callable(everySessionBehaviorPeriod):
                            bp = everySessionBehaviorPeriod(sesh)
                        else:
                            bp = everySessionBehaviorPeriod

                        _, xs, ys = sesh.posDuringBehaviorPeriod(bp)
                        ax.plot(xs, ys, c="#0000007f", lw=1, zorder=1.5)
                        if len(xs) > 0:
                            ax.scatter(xs[-1], ys[-1], marker="*")

                    setupBehaviorTracePlot(
                        ax, sesh, wellSize=wellSize, showWells="HA")

                    for v in ax.spines.values():
                        v.set_zorder(0)
                    ax.set_title(f"{sesh.name}", fontdict={'fontsize': 6})

                    im = ax.imshow(self.measureValsBySession[si].T, cmap=mpl.colormaps["coolwarm"],
                                   vmin=self.valMin, vmax=self.valMax,
                                   interpolation="nearest", extent=(-0.5, 6.5, -0.5, 6.5),
                                   origin="lower")

                for si in range(len(self.sessionList), nrows * ncols):
                    ax = pc.axs.flat[si]
                    blankPlot(ax)

                plt.colorbar(im, ax=ax)

        if any(["overlay" in pf for pf in plotFlags]):
            # synchronize the color limits across all overlay plots
            # So create them all before plotting
            if "everysessionoverlayatlocation" in plotFlags:
                wellSize = mpl.rcParams['lines.markersize']**2 / 4
                resolution = self.measureValsBySession[0].shape[0]
                overlayImg = np.empty(
                    (2 * resolution - 1, 2 * resolution - 1, len(self.sessionList)))
                overlayImg[:] = np.nan

                for si, sesh in enumerate(self.sessionList):
                    centerPoint = self.sessionValLocations[si]

                    px = np.digitize(
                        centerPoint[0], np.linspace(-0.5, 6.5, resolution))
                    py = np.digitize(
                        centerPoint[1], np.linspace(-0.5, 6.5, resolution))
                    ox = resolution - px - 1
                    oy = resolution - py - 1

                    overlayImg[ox:ox + resolution, oy:oy + resolution,
                               si] = self.measureValsBySession[si]

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", r"Mean of empty slice")
                    combinedImgAtLocation = np.nanmean(overlayImg, axis=2)

            if "everysessionoverlayatlocationbycondition" in plotFlags:
                wellSize = mpl.rcParams['lines.markersize']**2 / 4
                resolution = self.measureValsBySession[0].shape[0]
                overlayImg = np.empty(
                    (2 * resolution - 1, 2 * resolution - 1, len(self.sessionList)))
                overlayImg[:] = np.nan

                for si, sesh in enumerate(self.sessionList):
                    centerPoint = self.sessionValLocations[si]

                    px = np.digitize(
                        centerPoint[0], np.linspace(-0.5, 6.5, resolution))
                    py = np.digitize(
                        centerPoint[1], np.linspace(-0.5, 6.5, resolution))
                    ox = resolution - px - 1
                    oy = resolution - py - 1

                    overlayImg[ox:ox + resolution, oy:oy + resolution,
                               si] = self.measureValsBySession[si]

                swrIdx = self.conditionBySession == "SWR"
                ctrlIdx = ~swrIdx
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", r"Mean of empty slice")
                    combinedImgSWR = np.nanmean(
                        overlayImg[:, :, swrIdx], axis=2)
                    combinedImgCtrl = np.nanmean(
                        overlayImg[:, :, ctrlIdx], axis=2)

            if "everysessionoverlayatctrl" in plotFlags:
                wellSize = mpl.rcParams['lines.markersize']**2 / 4
                resolution = self.measureValsBySession[0].shape[0]
                combinedImgByCtrlName = {}

                for ctrlName in self.controlVals:
                    overlayImg = np.empty((2 * resolution - 1, 2 * resolution - 1,
                                           len(self.dotColorsByCtrlVal[ctrlName])))
                    overlayImg[:] = np.nan
                    overlayWeights = np.empty_like(overlayImg)
                    overlayWeights[:] = np.nan
                    cidx = 0

                    for si, sesh in enumerate(self.sessionList):
                        for ctrlLocs, cname, ctrlLabels in self.controlSpecsBySession[si]:
                            if cname != ctrlName:
                                continue
                            for centerPoint in ctrlLocs:
                                px = np.digitize(
                                    centerPoint[1], np.linspace(-0.5, 6.5, resolution))
                                py = np.digitize(
                                    centerPoint[2], np.linspace(-0.5, 6.5, resolution))
                                ox = resolution - px - 1
                                oy = resolution - py - 1

                                overlayImg[ox:ox + resolution, oy:oy + resolution,
                                           cidx] = self.measureValsBySession[centerPoint[0]]
                                overlayWeights[ox:ox + resolution, oy:oy +
                                               resolution, cidx] = 1 / len(ctrlLocs)

                                cidx += 1

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", r"invalid value encountered in divide")
                        combinedImgByCtrlName[ctrlName] = np.nansum(overlayImg * overlayWeights, axis=2) / \
                            np.nansum(overlayWeights, axis=2)

            if "everysessionoverlaydirect" in plotFlags:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", r"Mean of empty slice")
                    combinedImgDirect = np.nanmean(
                        self.measureValsBySession, axis=0)

            # Now get the minmum and maximum values for the color scale
            cmin = np.nanmin([np.nanmin(a) for a in [combinedImgAtLocation, combinedImgSWR, combinedImgCtrl,
                             combinedImgDirect] + list(combinedImgByCtrlName.values())])
            cmax = np.nanmax([np.nanmax(a) for a in [combinedImgAtLocation, combinedImgSWR, combinedImgCtrl,
                             combinedImgDirect] + list(combinedImgByCtrlName.values())])

            if "everysessionoverlayatlocation" in plotFlags:
                plotFlags.remove("everysessionoverlayatlocation")
                if verbose:
                    print("Plotting overlay at location")

                with plotManager.newFig(f"{figPrefix}every_session_overlay_at_location", excludeFromCombo=excludeFromCombo) as pc:
                    im = pc.ax.imshow(combinedImgAtLocation.T, cmap=mpl.colormaps["coolwarm"],
                                      interpolation="nearest", extent=(-7, 7, -7, 7),
                                      vmin=cmin, vmax=cmax,
                                      origin="lower")

                    pc.ax.set_xlabel("Distance from well (ft)")
                    pc.ax.set_ylabel("Distance from well (ft)")
                    pc.ax.set_title("Average of all sessions")

                    plt.colorbar(im, ax=pc.ax)

            if "everysessionoverlayatctrl" in plotFlags:
                plotFlags.remove("everysessionoverlayatctrl")
                if verbose:
                    print("Plotting overlay at control")

                for ctrlName in self.controlVals:
                    with plotManager.newFig(f"{figPrefix}every_session_overlay_at_ctrl_" + ctrlName, excludeFromCombo=excludeFromCombo) as pc:
                        im = pc.ax.imshow(combinedImgByCtrlName[ctrlName].T, cmap=mpl.colormaps["coolwarm"],
                                          interpolation="nearest", extent=(-7, 7, -7, 7),
                                          vmin=cmin, vmax=cmax,
                                          origin="lower")

                        pc.ax.set_xlabel("Distance from well (ft)")
                        pc.ax.set_ylabel("Distance from well (ft)")
                        pc.ax.set_title(ctrlName)

                        plt.colorbar(im, ax=pc.ax)

            if "everysessionoverlayatlocationbycondition" in plotFlags:
                plotFlags.remove("everysessionoverlayatlocationbycondition")
                if verbose:
                    print("Plotting overlay at location by condition")

                with plotManager.newFig(f"{figPrefix}every_session_overlay_at_location_swr", excludeFromCombo=excludeFromCombo) as pc:
                    im = pc.ax.imshow(combinedImgSWR.T, cmap=mpl.colormaps["coolwarm"],
                                      interpolation="nearest", extent=(-7, 7, -7, 7),
                                      vmin=cmin, vmax=cmax,
                                      origin="lower")

                    pc.ax.set_xlabel("Distance from well (ft)")
                    pc.ax.set_ylabel("Distance from well (ft)")
                    pc.ax.set_title("SWR")

                    plt.colorbar(im, ax=pc.ax)

                with plotManager.newFig(f"{figPrefix}every_session_overlay_at_location_ctrl", excludeFromCombo=excludeFromCombo) as pc:
                    im = pc.ax.imshow(combinedImgCtrl.T, cmap=mpl.colormaps["coolwarm"],
                                      interpolation="nearest", extent=(-7, 7, -7, 7),
                                      vmin=cmin, vmax=cmax,
                                      origin="lower")

                    pc.ax.set_xlabel("Distance from well (ft)")
                    pc.ax.set_ylabel("Distance from well (ft)")
                    pc.ax.set_title("Control")

                    plt.colorbar(im, ax=pc.ax)

            if "everysessionoverlaydirect" in plotFlags:
                plotFlags.remove("everysessionoverlaydirect")
                if verbose:
                    print("Plotting overlay at location")

                with plotManager.newFig(f"{figPrefix}every_session_overlay_direct", excludeFromCombo=excludeFromCombo) as pc:
                    im = pc.ax.imshow(combinedImgDirect.T, cmap=mpl.colormaps["coolwarm"],
                                      interpolation="nearest", extent=(-0.5, 6.5, -0.5, 6.5),
                                      vmin=cmin, vmax=cmax,
                                      origin="lower")

                    pc.ax.set_xlabel("xpos (ft)")
                    pc.ax.set_ylabel("ypos (ft)")
                    pc.ax.set_title("Average of all sessions")

                    plt.colorbar(im, ax=pc.ax)

        if len(plotFlags) > 0:
            print(f"Warning: unused plot flags: {plotFlags}")

        with open(os.path.join(plotManager.fullOutputDir, "processed.txt"), "w") as f:
            f.write(plotManager.infoFileFullName)
        if verbose:
            print(f"wrote processed.txt to {plotManager.fullOutputDir}")

        if subFolder:
            plotManager.popOutputSubDir()

    def makeCorrelationFigures(self,
                               plotManager: PlotManager,
                               sm: SessionMeasure,
                               plotFlags: str | List[str] = "all",
                               runStats: bool = True,
                               subFolder: bool = True,
                               excludeFromCombo=False) -> None:
        """
        Makes figures similar to the ones in makeFigures but considering the values given by sm
        """
        if not self.valid:
            return

        figName = self.name.replace(
            " ", "_") + "_X_" + sm.name.replace(" ", "_")
        statsId = self.name + "_"
        figPrefix = "" if (subFolder and False) else figName + "_"
        dataName = self.name.replace(" ", "_")

        if subFolder:
            plotManager.pushOutputSubDir("LM_" + figName)

        allPossibleFlags = ["measure", "measureByCondition", "measureVsCtrl",
                            "measureVsCtrlByCondition", "diff", "diffByCondition"]

        if isinstance(plotFlags, str):
            if plotFlags == "all":
                plotFlags = allPossibleFlags
            else:
                plotFlags = [plotFlags]
        else:
            # so list passed in isn't modified
            plotFlags = [v for v in plotFlags]

        if len(plotFlags) > 0 and plotFlags[0].startswith("not"):
            if not all([v.startswith("not") for v in plotFlags]):
                raise ValueError(
                    "if one plot flag starts with 'not', all must")
            plotFlags = [v[3:] for v in plotFlags]
            plotFlags = list(set(allPossibleFlags) - set(plotFlags))

        smVals = np.array(sm.sessionVals)

        if "measure" in plotFlags:
            plotFlags.remove("measure")
            with plotManager.newFig(f"{figPrefix}measure", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}measure") as pc:
                pc.ax.scatter(smVals, self.sessionValsBySession)
                pc.ax.set_xlabel(sm.name)
                pc.ax.set_ylabel(self.name)
                pc.ax.set_title(f"{self.name} vs {sm.name}")

                if runStats:
                    pc.yvals[dataName] = self.sessionValsBySession
                    pc.xvals[sm.name] = smVals
                    pc.immediateCorrelations.append((sm.name, dataName))

        if "measureByCondition" in plotFlags:
            plotFlags.remove("measureByCondition")
            swrIdx = sm.conditionBySession == "SWR"
            ctrlIdx = sm.conditionBySession == "Ctrl"
            with plotManager.newFig(f"{figPrefix}measureByCondition", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}measureByCondition") as pc:
                pc.ax.scatter(
                    smVals[swrIdx], self.sessionValsBySession[swrIdx], label="SWR", color="orange")
                pc.ax.scatter(
                    smVals[ctrlIdx], self.sessionValsBySession[ctrlIdx], label="Ctrl", color="cyan")
                pc.ax.set_xlabel(sm.name)
                pc.ax.set_ylabel(self.name)
                pc.ax.set_title(f"{self.name} vs {sm.name}")
                pc.ax.legend()

                if runStats:
                    pc.yvals[dataName] = self.sessionValsBySession
                    pc.xvals[sm.name] = smVals
                    pc.categories["condition"] = sm.conditionBySession
                    pc.immediateCorrelations.append((sm.name, dataName))

        if "measureVsCtrl" in plotFlags:
            plotFlags.remove("measureVsCtrl")

            for ctrlName in self.controlValLabels:
                vals = np.concatenate(
                    (self.sessionValsBySession, *self.controlVals[ctrlName]))
                valCtrlCats = np.concatenate((np.full_like(self.sessionValsBySession, self.controlValLabels[ctrlName][0], dtype=object),
                                              np.full_like(self.dotColorsByCtrlVal[ctrlName], self.controlValLabels[ctrlName][1], dtype=object)))
                xvals = list(smVals)
                for si in range(len(self.controlVals[ctrlName])):
                    xvals += [smVals[si]] * len(self.controlVals[ctrlName][si])
                xvals = np.array(xvals)

                valIdx = valCtrlCats == self.controlValLabels[ctrlName][0]
                ctrlIdx = valCtrlCats == self.controlValLabels[ctrlName][1]

                with plotManager.newFig(f"{figPrefix}ctrl_" + ctrlName, excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}ctrl_" + ctrlName) as pc:
                    pc.ax.scatter(xvals[valIdx], vals[valIdx],
                                  label=self.controlValLabels[ctrlName][0], color="red", zorder=2)
                    pc.ax.scatter(xvals[ctrlIdx], vals[ctrlIdx],
                                  label=self.controlValLabels[ctrlName][1], color="blue", zorder=1)
                    pc.ax.set_title(self.name + " vs " +
                                    ctrlName, fontdict={'fontsize': 6})
                    pc.ax.set_xlabel(sm.name)
                    pc.ax.set_ylabel(self.name)
                    pc.ax.legend(loc="lower center")

                    if runStats:
                        catName = self.controlValLabels[ctrlName][2].replace(
                            " ", "_")
                        pc.yvals[dataName] = vals
                        pc.categories[catName] = valCtrlCats
                        pc.xvals[sm.name] = xvals
                        pc.immediateCorrelations.append((sm.name, dataName))

        if "measureVsCtrlByCondition" in plotFlags:
            plotFlags.remove("measureVsCtrlByCondition")

            for ctrlName in self.controlValLabels:
                vals = np.concatenate(
                    (self.sessionValsBySession, *self.controlVals[ctrlName]))
                valCtrlCats = np.concatenate((np.full_like(self.sessionValsBySession, self.controlValLabels[ctrlName][0], dtype=object),
                                              np.full_like(self.dotColorsByCtrlVal[ctrlName], self.controlValLabels[ctrlName][1], dtype=object)))
                conditionCats = np.concatenate(
                    (self.conditionBySession, self.conditionByCtrlVal[ctrlName]))
                xvals = list(smVals)
                for si in range(len(self.controlVals[ctrlName])):
                    xvals += [smVals[si]] * len(self.controlVals[ctrlName][si])
                xvals = np.array(xvals)

                valIdx = valCtrlCats == self.controlValLabels[ctrlName][0]
                swrIdx = conditionCats == "SWR"
                swrValIdx = swrIdx & valIdx
                swrCtrlIdx = swrIdx & ~valIdx
                ctrlValIdx = ~swrIdx & valIdx
                ctrlCtrlIdx = ~swrIdx & ~valIdx

                with plotManager.newFig(f"{figPrefix}ctrl_" + ctrlName + "_byCond", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}ctrl_" + ctrlName + "_byCond") as pc:
                    pc.ax.scatter(xvals[swrValIdx], vals[swrValIdx],
                                  label=self.controlValLabels[ctrlName][0] + " SWR", color="orange", marker="o", zorder=2)
                    pc.ax.scatter(xvals[ctrlValIdx], vals[ctrlValIdx],
                                  label=self.controlValLabels[ctrlName][0] + " Ctrl", color="cyan", marker="o", zorder=2)
                    pc.ax.scatter(xvals[swrCtrlIdx], vals[swrCtrlIdx],
                                  label=self.controlValLabels[ctrlName][1] + " SWR", color="orange", marker="x", zorder=1)
                    pc.ax.scatter(xvals[ctrlCtrlIdx], vals[ctrlCtrlIdx],
                                  label=self.controlValLabels[ctrlName][1] + " Ctrl", color="cyan", marker="x", zorder=1)
                    pc.ax.set_title(self.name + " " + ctrlName,
                                    fontdict={'fontsize': 6})
                    pc.ax.set_xlabel(sm.name)
                    pc.ax.set_ylabel(self.name)
                    pc.ax.legend(loc="lower center")

                    if runStats:
                        catName = self.controlValLabels[ctrlName][2].replace(
                            " ", "_")
                        pc.yvals[dataName] = vals
                        pc.categories[catName] = valCtrlCats
                        pc.categories["condition"] = conditionCats
                        pc.xvals[sm.name] = xvals
                        pc.immediateCorrelations.append((sm.name, dataName))

        if "diff" in plotFlags:
            plotFlags.remove("diff")
            for ctrlName in self.controlValLabels:
                vals = self.sessionValsBySession - \
                    self.controlValMeans[ctrlName]
                with plotManager.newFig(f"{figPrefix}ctrl_" + ctrlName + "_diff", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}ctrl_" + ctrlName + "_diff") as pc:
                    if all(np.isnan(vals)):
                        continue
                    pc.ax.scatter(smVals, vals, color="black")
                    pc.ax.set_xlabel(sm.name)
                    pc.ax.set_ylabel(self.name)
                    pc.ax.set_title(self.name + " difference" + " " +
                                    ctrlName, fontdict={'fontsize': 6})

                    if runStats:
                        pc.yvals[dataName] = vals
                        pc.xvals[sm.name] = smVals
                        pc.immediateCorrelations.append((sm.name, dataName))

        if "diffByCondition" in plotFlags:
            plotFlags.remove("diffByCondition")
            for ctrlName in self.controlValLabels:
                vals = self.sessionValsBySession - \
                    self.controlValMeans[ctrlName]
                swrIdx = sm.conditionBySession == "SWR"
                ctrlIdx = sm.conditionBySession == "Ctrl"

                with plotManager.newFig(f"{figPrefix}ctrl_" + ctrlName + "_diff_byCond", excludeFromCombo=excludeFromCombo, uniqueID=statsId + f"{figPrefix}ctrl_" + ctrlName + "_diff_byCond") as pc:
                    if all(np.isnan(vals)):
                        continue
                    pc.ax.scatter(smVals[swrIdx], vals[swrIdx], color="orange")
                    pc.ax.scatter(smVals[ctrlIdx], vals[ctrlIdx], color="cyan")
                    pc.ax.set_xlabel(sm.name)
                    pc.ax.set_ylabel(self.name)
                    pc.ax.set_title(self.name + " difference by condition" + " " + ctrlName,
                                    fontdict={'fontsize': 6})

                    if runStats:
                        pc.yvals[dataName] = vals
                        pc.xvals[sm.name] = smVals
                        pc.categories["condition"] = sm.conditionBySession
                        pc.immediateCorrelations.append((sm.name, dataName))

        if len(plotFlags) > 0:
            print(f"Warning: unused plot flags: {plotFlags}")

        # Finally, in the output folder, create a blank file to indicate that this session has been processed
        with open(os.path.join(plotManager.fullOutputDir, "corr_processed.txt"), "w") as f:
            f.write(plotManager.infoFileFullName)

        if subFolder:
            plotManager.popOutputSubDir()

    def makeLocationCorrelationFigures(self,
                                       plotManager: PlotManager,
                                       lm: LocationMeasure,
                                       subFolder: bool = True,
                                       excludeFromCombo: bool = False,
                                       runStats: bool = True):
        if not self.valid:
            return
        if not lm.valid:
            return

        figName = self.name.replace(
            " ", "_") + "_X_" + lm.name.replace(" ", "_")
        statsId = self.name + "_"
        figPrefix = "" if subFolder else figName + "_"
        dataName = self.name.replace(" ", "_")
        otherDataName = lm.name.replace(" ", "_")

        if subFolder:
            plotManager.pushOutputSubDir("LM_" + figName)

        pxlVals = np.arange(-0.5, 6.5, 0.5) + 0.25
        nPxl = len(pxlVals) * len(pxlVals)
        thisMeasureVals = np.empty((len(self.sessionList), nPxl))
        otherMeasureVals = np.empty((len(self.sessionList), nPxl))
        for si in range(len(self.sessionList)):
            pxlPairs = np.array(list(product(pxlVals, pxlVals)))
            for pi, pxlPairs in enumerate(pxlPairs):
                thisMeasureVals[si, pi] = LocationMeasure.measureAtLocation(
                    self.measureValsBySession[si], pxlPairs, smoothDist=self.smoothDist)
                otherMeasureVals[si, pi] = LocationMeasure.measureAtLocation(
                    lm.measureValsBySession[si], pxlPairs, smoothDist=lm.smoothDist)

        with plotManager.newFig(figName, excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName) as pc:
            pc.ax.scatter(thisMeasureVals.flatten(),
                          otherMeasureVals.flatten(), color="black", s=0.1)
            pc.ax.set_xlabel(self.name)
            pc.ax.set_ylabel(lm.name)
            pc.ax.set_title(self.name + " X " + lm.name,
                            fontdict={'fontsize': 6})

            if runStats:
                pc.yvals[otherDataName] = otherMeasureVals.flatten()
                pc.xvals[dataName] = thisMeasureVals.flatten()
                pc.immediateCorrelations.append((dataName, otherDataName))

        with plotManager.newFig(figName + "_byCond", excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName + "_byCond") as pc:
            swrIdx = self.conditionBySession == "SWR"
            ctrlIdx = self.conditionBySession == "Ctrl"
            pc.ax.scatter(thisMeasureVals[swrIdx, :].flatten(),
                          otherMeasureVals[swrIdx, :].flatten(), color="orange")
            pc.ax.scatter(thisMeasureVals[ctrlIdx, :].flatten(),
                          otherMeasureVals[ctrlIdx, :].flatten(), color="cyan")

            pc.ax.set_xlabel(self.name)
            pc.ax.set_ylabel(lm.name)
            pc.ax.set_title(self.name + " X " + lm.name +
                            " by condition", fontdict={'fontsize': 6})

            if runStats:
                # tile the categories so it's the same size as the data
                cats = np.tile(self.conditionBySession, (nPxl, 1)).T.flatten()
                pc.yvals[otherDataName] = otherMeasureVals.flatten()
                pc.xvals[dataName] = thisMeasureVals.flatten()
                pc.categories["condition"] = cats
                pc.categoryColors = {"SWR": "orange", "Ctrl": "cyan"}
                pc.immediateCorrelations.append((dataName, otherDataName))

        if subFolder:
            plotManager.popOutputSubDir()

    def makeDualCategoryFigures(self,
                                plotManager: PlotManager,
                                subFolder: bool = True,
                                excludeFromCombo: bool = False,
                                runStats: bool = True,
                                numShuffles: int = 100):
        if not self.valid:
            return

        figName = self.name.replace(" ", "_")
        statsId = self.name + "_"
        figPrefix = "" if subFolder else figName + "_"
        dataName = self.name.replace(" ", "_")

        if subFolder:
            plotManager.pushOutputSubDir("LM_" + figName)

        # Make an array of session categories that is a concatenation of the session categories with the previous session's category
        dualSessionList = []
        dualSessionVals = []
        dualCatsSame = []
        dualCatsPrev = []
        dualCatsCombo = []
        for si in range(1, len(self.sessionList)):
            sesh = self.sessionList[si]
            if sesh.prevSession is None:
                continue
            dualSessionList.append(sesh)
            dualSessionVals.append(self.sessionValsBySession[si])
            dualCatsSame.append(self.conditionBySession[si])
            dualCatsPrev.append(self.conditionBySession[si - 1])
            dualCatsCombo.append(
                self.conditionBySession[si] + "_" + self.conditionBySession[si - 1])

        dualSessionVals = np.array(dualSessionVals)
        dualCatsSame = np.array(dualCatsSame)
        dualCatsPrev = np.array(dualCatsPrev)
        dualCatsCombo = np.array(dualCatsCombo)

        with plotManager.newFig(figName + "_byDualCond", excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName + "_byDualCond") as pc:
            violinPlot(pc.ax, dualSessionVals, categories=dualCatsSame, categories2=dualCatsPrev,
                       dotColors=["orange" if c ==
                                  "SWR" else "cyan" for c in dualCatsPrev],
                       dotColorLabels={"orange": "SWR", "cyan": "Ctrl"})
            pc.ax.set_xlabel("Previous condition")

            if runStats:
                pc.yvals[dataName] = dualSessionVals
                pc.categories["condition"] = dualCatsSame
                pc.categories["prevCondition"] = dualCatsPrev
                pc.categoryColors = {"SWR": "orange", "Ctrl": "cyan"}
                pc.immediateShuffles.append((
                    [ShuffSpec(shuffType=ShuffSpec.ShuffType.RECURSIVE_ALL, categoryName="prevCondition", value=None),
                        ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

        with plotManager.newFig(figName + "_byThisCond", excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName + "_byThisCond") as pc:
            violinPlot(pc.ax, dualSessionVals, categories=dualCatsSame,
                       dotColors=["orange" if c ==
                                  "SWR" else "cyan" for c in dualCatsSame],
                       dotColorLabels={"orange": "SWR", "cyan": "Ctrl"})
            pc.ax.set_xlabel("This session condition")

            if runStats:
                pc.yvals[dataName] = dualSessionVals
                pc.categories["condition"] = dualCatsSame
                pc.categoryColors = {"SWR": "orange", "Ctrl": "cyan"}
                pc.immediateShuffles.append((
                    [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")], numShuffles))

        with plotManager.newFig(figName + "_byPrevCond", excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName + "_byPrevCond") as pc:
            violinPlot(pc.ax, dualSessionVals, categories=dualCatsPrev,
                       dotColors=["orange" if c ==
                                  "SWR" else "cyan" for c in dualCatsSame],
                       dotColorLabels={"orange": "SWR", "cyan": "Ctrl"})
            pc.ax.set_xlabel("Previous condition")

            if runStats:
                pc.yvals[dataName] = dualSessionVals
                pc.categories["prevCondition"] = dualCatsPrev
                pc.categoryColors = {"SWR": "orange", "Ctrl": "cyan"}
                pc.immediateShuffles.append((
                    [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="prevCondition", value="SWR")], numShuffles))

        with plotManager.newFig(figName + "_byTimeSincePreviousSession", excludeFromCombo=excludeFromCombo, uniqueID=statsId + figName + "_byTimeSincePreviousSession") as pc:
            swrswrIdx = np.logical_and(
                dualCatsSame == "SWR", dualCatsPrev == "SWR")
            ctrlswrIdx = np.logical_and(
                dualCatsSame == "SWR", dualCatsPrev == "Ctrl")
            swrctrlIdx = np.logical_and(
                dualCatsSame == "Ctrl", dualCatsPrev == "SWR")
            ctrlctrlIdx = np.logical_and(
                dualCatsSame == "Ctrl", dualCatsPrev == "Ctrl")

            delayTimes = np.array(
                [sesh.secondsSincePrevSession for sesh in dualSessionList])
            pc.ax.scatter(delayTimes[swrswrIdx],
                          dualSessionVals[swrswrIdx], color="orange")
            pc.ax.scatter(delayTimes[swrctrlIdx],
                          dualSessionVals[swrctrlIdx], color="orange", marker="x")
            pc.ax.scatter(delayTimes[ctrlswrIdx],
                          dualSessionVals[ctrlswrIdx], color="cyan", marker="x")
            pc.ax.scatter(delayTimes[ctrlctrlIdx],
                          dualSessionVals[ctrlctrlIdx], color="cyan")

            pc.ax.set_xlabel("Time since previous session (s)")
            pc.ax.set_ylabel(self.name)
            pc.ax.legend()

            if runStats:
                pc.yvals[dataName] = dualSessionVals
                pc.xvals["delayTime"] = delayTimes
                pc.categories["condition"] = dualCatsSame
                pc.categories["prevCondition"] = dualCatsPrev
                pc.categoryColors = {"SWR": "orange", "Ctrl": "cyan"}
                pc.immediateCorrelations.append(("delayTime", dataName))

        if subFolder:
            plotManager.popOutputSubDir()
