import numpy as np
from consts import allWellNames, TRODES_SAMPLING_RATE, LFP_SAMPLING_RATE
import csv
import glob
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from itertools import groupby
from datetime import date
import sys
import os
from typing import List, Dict, Tuple, Optional
import functools
import time
import subprocess
from numpy.typing import ArrayLike
from dataclasses import dataclass, field


@dataclass
class Ripple:
    start_lfpIdx: int
    len_lfpIdx: int
    peakIdx_lfpIdx: int
    peakAmp: float
    crossThreshIdx_lfpIdx: int
    start_ts: int

    param_detectionThresh: float
    param_edgeThresh: float
    param_minLen: float
    param_maxLen: float


@dataclass
class ImportOptions:
    """
    This is global options specified possibly on command line that control how all animal data is processed
    """
    skipLFP: bool = False
    skipUSB: bool = False
    skipActivelinkLog: bool = False
    skipPrevSession: bool = True
    forceAllTrackingAuto: bool = False
    skipConfirmTrackingFileExtension: bool = True
    skipCurvature: bool = False
    runJustSpecified: bool = False
    specifiedDays: list = field(default_factory=list)
    specifiedRuns: list = field(default_factory=list)
    justExtractData: bool = False
    runInteractiveExtraction: bool = True
    confirmAfterEachSession: bool = False

    # ===========
    # consts:
    VEL_THRESH: float = 15.0  # cm/s
    MOVE_THRESH_SM_SIGMA_SECS: float = 0.8
    #  Typical observed amplitude of LFP deflection on stimulation
    DEFLECTION_THRESHOLD_HI: float = 6000.0
    DEFLECTION_THRESHOLD_LO: float = 2000.0
    MIN_ARTIFACT_DISTANCE: int = int(0.05 * LFP_SAMPLING_RATE)
    #  How much buffer time to give the ITI around behavior times
    ITI_MARGIN: float = 10.0
    #  constants for exploration bout analysis
    BOUT_VEL_SM_SIGMA_SECS: float = 1.5
    PAUSE_MAX_SPEED_CM_S: float = 8.0
    MIN_PAUSE_TIME_BETWEEN_BOUTS_SECS: float = 1.0
    MIN_EXPLORE_TIME_SECS: float = 3.0
    MIN_EXPLORE_NUM_WELLS: int = 4
    #  constants for ballisticity
    BALL_TIME_INTERVALS: List[int] = field(
        default_factory=lambda: list(range(1, 24)))
    KNOT_H_CM: float = 8.0

    # ============
    # debug :
    debugMode: bool = False
    debug_maxNumSessions: Optional[int] = None
    debug_dontSave: bool = False


@dataclass
class LoadInfo:
    """
    this is animal-specific info such as paths to raw data
    """
    # configName is the argument that determines all other values
    # It is what is specified on the command line
    configName: str
    animalName: str

    # position info:
    X_START: int
    X_FINISH: int
    Y_START: int
    Y_FINISH: int
    excludeBoxes: Optional[List[Tuple[int, int, int, int]]]

    # File names
    # First element is drive name, other elements are dir names. Actual file path str is build dynamically
    # path to bradtasksessions (or similar) directory with raw run folders
    dataDirPath: List[str]
    # path to where output file from ImportData.py is saved
    outputDirPath: List[str]
    # filename of output file from ImportData.py
    out_filename: str

    # some options about what data to include
    excluded_dates: List[str]
    excluded_sessions: List[str]
    minimum_date: Optional[str]
    maximum_date: Optional[str]

    DEFAULT_RIP_DET_TET: Optional[int]
    DEFAULT_RIP_BAS_TET: Optional[int]

    # if DeepLabCut was used, this will be not None
    DLC_dir: Optional[str] = None

    @property
    def data_dir(self):
        return os.path.join(getDrivePathByLabel(self.dataDirPath[0]), *self.dataDirPath[1:])

    @property
    def output_dir(self):
        return os.path.join(getDrivePathByLabel(self.outputDirPath[0]), *self.outputDirPath[1:])


def findDataDir(possibleDataDirs=None) -> str:
    if possibleDataDirs is None:
        possibleDataDirs = [
            getDrivePathByLabel("Harold"),
            getDrivePathByLabel("WDC10"),
            getDrivePathByLabel("WDC9"),
            getDrivePathByLabel("WDC11"),
            getDrivePathByLabel("WDC4"),
            getDrivePathByLabel("WDC8"),
            getDrivePathByLabel("WDC6"),
            "/media/fosterlab/WDC6/",
            "/home/wcroughan/data/",
        ]
    for dd in possibleDataDirs:
        if os.path.exists(dd):
            return dd

    return None


def parseCmdLineAnimalNames(default: Optional[List[str]] = None) -> List[str]:
    a = sys.argv[1:]
    names = []
    i = 0
    while i < len(a):
        s = a[i]
        if s.startswith("--"):
            i += 2
            continue

        if s.startswith("-"):
            i += 1
            continue

        if s == "new":
            names += ["B16", "B17", "B18"]
        elif s == "old":
            names += ["Martin", "B13", "B14"]
        elif s == "all":
            names += ["Martin", "B13", "B14", "B16", "B17", "B18"]
        else:
            names.append(s)
        i += 1

    if len(names) == 0:
        return default

    return names


def parseCmdLineImportOptions() -> ImportOptions:
    importOptions = ImportOptions()
    a = sys.argv[1:]
    i = 0
    while i < len(a):
        s = a[i]
        if not s.startswith("-"):
            i += 1
            continue

        if s.startswith("--"):
            flag = s[2:]
            val = a[i+1]
            if flag == "dsessions":
                importOptions.debug_maxNumSessions = int(val)
            i += 2
            continue

        flag = s[1:]
        if flag == "D":
            importOptions.debugMode = True
        if flag == "ns":
            importOptions.debug_dontSave = True

        i += 1

    return importOptions


def readWellCoordsFile(wellCoordsFile: str) -> Dict[str, Tuple[int, int]]:
    # For some reason json saving and loading turns the keys into strings, just going to change that here so
    # it's consistent
    with open(wellCoordsFile, 'r') as wcf:
        wellCoordsMap = {}
        csvReader = csv.reader(wcf)
        for dataRow in csvReader:
            try:
                wellCoordsMap[str(int(dataRow[0]))] = (
                    int(dataRow[1]), int(dataRow[2]))
            except Exception as err:
                if dataRow[1] != '':
                    print(err)

        return wellCoordsMap


def readRawPositionData(data_filename):
    try:
        with open(data_filename, 'rb') as datafile:
            dt = np.dtype([('timestamp', np.uint32), ('x1', np.uint16),
                           ('y1', np.uint16), ('x2', np.uint16), ('y2', np.uint16)])
            lineText = ""
            max_iter = 8
            iter = 0
            settings = {}
            while lineText != b'<end settings>\n':
                lineText = datafile.readline().lower()
                if lineText != b'<end settings>\n' and lineText != b'<start settings>\n':
                    ss = str(lineText).split(":")
                    if len(ss) == 2:
                        settings[str(ss[0])[2:].strip()] = str(
                            ss[1])[0:-3].strip()
                # print(lineText)
                iter += 1
                if iter > max_iter:
                    raise Exception
            return settings, np.fromfile(datafile, dtype=dt)
    except Exception as err:
        print(err)
        return None, None


def quickPosPlot(tt, xx, yy, title, irange=None):
    if True:
        if irange is not None:
            xx = xx[irange[0]:irange[1]]
            yy = yy[irange[0]:irange[1]]
            tt = tt[irange[0]:irange[1]]
        plt.plot(xx, yy)
        plt.title(title)
        plt.show()
        plt.plot(tt, xx)
        plt.plot(tt, yy)
        plt.show()
        plt.scatter(tt, xx)
        plt.scatter(tt, yy)
        plt.show()


def timeStrForTrodesTimestamp(ts):
    if ts is None:
        return "[None]"
    s = ts // TRODES_SAMPLING_RATE
    ssecs = str(s % 60)
    if len(ssecs) == 1:
        ssecs = "0" + ssecs
    s = s // 60
    smins = str(s % 60)
    if len(smins) == 1:
        smins = "0" + smins
    shrs = s // 60
    return f"{shrs}:{smins}:{ssecs}"


def getWellPosCoordinates(wellName: int) -> Tuple[float, float]:
    assert wellName in allWellNames
    y = float(wellName // 8) + 0.5
    x = float(wellName % 8 - 2) + 0.5
    return x, y


def getRipplePower(lfpData: ArrayLike, method="standard", meanPower=None,
                   stdPower=None, lfpDeflections=None, smoothStd=0.05,
                   showPlot=False, baselineLfpData: Optional[ArrayLike] = None):
    if method not in ["standard", "activelink", "causal"]:
        raise ValueError("Unknown ripple power method")

    if (meanPower is None) != (stdPower is None):
        raise ValueError(
            "meanPower and stdPower must both be provided or None")

    if method == "activelink":
        assert baselineLfpData is not None
        assert meanPower is not None
    else:
        assert baselineLfpData is None

    lfpData = lfpData.copy().astype(np.float64)
    smoothStd_frames = int(smoothStd * LFP_SAMPLING_RATE)

    lfpMask = np.zeros_like(lfpData).astype(bool)
    if lfpDeflections is not None:
        if method == "standard":
            # skipTimePointsBackward = 1
            # skipTimePointsForward = int(0.075 * LFP_SAMPLING_RATE)
            skipTimePointsBackward = smoothStd_frames * 3
            skipTimePointsForward = smoothStd_frames * 3
        elif method == "causal":
            skipTimePointsBackward = 1
            # skipTimePointsForward = int(0.075 * LFP_SAMPLING_RATE)
            skipTimePointsForward = smoothStd_frames * 3
        elif method == "activelink":
            skipTimePointsBackward = 1
            # skipTimePointsForward = int(0.075 * LFP_SAMPLING_RATE)
            skipTimePointsForward = smoothStd_frames * 3
        else:
            assert False

        for artifactIdx in lfpDeflections:
            if artifactIdx < 0 or artifactIdx > len(lfpData):
                continue
            cleanupStart = max(0, artifactIdx - skipTimePointsBackward)
            cleanupFinish = min(len(lfpData), artifactIdx +
                                skipTimePointsForward)
            lfpMask[cleanupStart:cleanupFinish] = True

    rippleFilterBand = (150, 250)
    if method == "standard":
        rippleFilterOrder = 4
        sos = signal.butter(rippleFilterOrder, rippleFilterBand,
                            btype='band', fs=LFP_SAMPLING_RATE, output="sos")
        filteredSignal = signal.sosfiltfilt(sos, lfpData)
        smoothedRipplePower = gaussian_filter(
            np.abs(filteredSignal), smoothStd_frames)
        smoothedRipplePower[lfpMask] = np.nan
    elif method == "causal":
        rippleFilterOrder = 4
        sos = signal.butter(rippleFilterOrder, rippleFilterBand,
                            btype='band', fs=LFP_SAMPLING_RATE, output="sos")
        filteredSignal = signal.sosfilt(sos, lfpData)

        filterLenFactor = 4  # How many stds should the filter extend?
        halfSmoothingEnvelope = np.exp(-np.square(np.linspace(0, filterLenFactor / 2.0,
                                                              smoothStd_frames * filterLenFactor)))
        halfSmoothingEnvelope = halfSmoothingEnvelope / \
            np.sum(halfSmoothingEnvelope)
        smoothingEnvelope = np.concatenate(
            (np.zeros_like(halfSmoothingEnvelope), halfSmoothingEnvelope), axis=0)
        smoothedRipplePower = signal.convolve(
            np.abs(filteredSignal), smoothingEnvelope, mode='same')
        smoothedRipplePower[lfpMask] = np.nan

        if showPlot:
            plt.plot(smoothingEnvelope)
            plt.show()

            plt.plot(lfpData, label="raw")
            plt.plot(filteredSignal, label="filt")
            plt.plot(smoothedRipplePower, label="smooth")
            plt.legend()
            plt.show()
    elif method == "activelink":
        """
        1. Each tetrode's signal is filtered to ripple band
        2. within each signal window, the RMS power is calculated
        3. The power within each window is smoothed by adding 0.8 * prev window val
            (this happens sequentially, each prev val was smoothed in turn by values before it,
             the prev window power also have baseline subtracted out)
        4. The baseline power is subtracted from detection tetrode power
        5. power is z-scored
        """
        rippleFilterOrder = 3
        sos = signal.butter(rippleFilterOrder, rippleFilterBand, btype='bandpass',
                            output="sos", fs=LFP_SAMPLING_RATE)
        # zi = signal.sosfilt_zi(sos)
        filteredSignal = signal.sosfilt(sos, lfpData)
        filteredBaselineSignal = signal.sosfilt(sos, baselineLfpData)
        assert len(filteredSignal) == len(
            lfpData) == len(filteredBaselineSignal)

        # activelink processes data once the number of datapoints is the filter order
        originalLength = len(filteredSignal)
        if len(filteredSignal) % rippleFilterOrder != 0:
            filteredSignal = filteredSignal[:int(
                len(filteredSignal) // rippleFilterOrder)*rippleFilterOrder]
            filteredBaselineSignal = filteredBaselineSignal[:int(
                len(filteredBaselineSignal) // rippleFilterOrder)*rippleFilterOrder]
            lfpMask = lfpMask[:int(
                len(lfpMask) // rippleFilterOrder)*rippleFilterOrder]

        rs = filteredSignal.reshape((-1, rippleFilterOrder)).T
        rbs = filteredBaselineSignal.reshape((-1, rippleFilterOrder)).T
        rmsWindows = np.sqrt(np.mean(np.power(rs, 2.0), axis=0))
        rmsBaselineWindows = np.sqrt(np.mean(np.power(rbs, 2.0), axis=0))
        powerMinusBaseline = rmsWindows - rmsBaselineWindows

        smoothedRipplePower = np.empty_like(powerMinusBaseline)
        prevVal = 0.0

        for i in range(len(powerMinusBaseline)):
            smoothedRipplePower[i] = 0.8 * prevVal + powerMinusBaseline[i]
            prevVal = smoothedRipplePower[i]

        # lfpWindowMask = np.any(lfpMask.reshape((rippleFilterOrder, -1)), axis=0)
        # smoothedRipplePower[lfpWindowMask] = np.nan
        smoothedRipplePower = np.repeat(smoothedRipplePower, rippleFilterOrder)
        if len(smoothedRipplePower) < originalLength:
            z = np.zeros((1, originalLength - len(smoothedRipplePower)))
            smoothedRipplePower = np.append(smoothedRipplePower, z)
    else:
        assert False

    if meanPower is None:
        meanPower = np.nanmean(smoothedRipplePower)
        stdPower = np.nanstd(smoothedRipplePower)
    zRipplePower = (smoothedRipplePower - meanPower) / stdPower

    return smoothedRipplePower, zRipplePower, meanPower, stdPower


def detectRipples(ripplePower, tsFunc=lambda lfpIdx: None, minHeight=3.0, minLen=0.05, maxLen=0.3, edgeThresh=0.0) -> List[Ripple]:
    pks, _ = signal.find_peaks(ripplePower, height=minHeight)

    ret = []
    i = 0
    while i < len(pks):
        pkidx = pks[i]
        ii = pkidx
        while ii >= 0 and ripplePower[ii] > edgeThresh:
            ii -= 1
        ii += 1
        ripStart = ii

        length = 0
        pkAmp = 0
        pkAmpI = 0
        crossI = 0
        crossed = False
        while ii < len(ripplePower) and ripplePower[ii] > edgeThresh:
            if ripplePower[ii] > pkAmp:
                pkAmp = ripplePower[ii]
                pkAmpI = ii

            if not crossed and ripplePower[ii] > minHeight:
                crossed = True
                crossI = ii

            ii += 1
            length += 1

        assert crossed

        lensec = float(length) / LFP_SAMPLING_RATE
        if lensec >= minLen and lensec <= maxLen:
            ret.append(Ripple(start_lfpIdx=ripStart, start_ts=tsFunc(ripStart), len_lfpIdx=length, peakIdx_lfpIdx=pkAmpI,
                              peakAmp=pkAmp, crossThreshIdx_lfpIdx=crossI,
                              param_detectionThresh=minHeight, param_edgeThresh=edgeThresh, param_maxLen=maxLen,
                              param_minLen=minLen))

        while i < len(pks) and pks[i] < ii:
            i += 1

    return ret


# 2 3
# 0 1
def quadrantOfWell(well_idx):
    if well_idx > 24:
        res = 2
    else:
        res = 0

    if (well_idx - 1) % 8 >= 4:
        res += 1

    return res


def getListOfVisitedWells(nearestWells, countFirstVisitOnly):
    if countFirstVisitOnly:
        return list(set(nearestWells))
    else:
        return [k for k, g in groupby(nearestWells)]


def onWall(well):
    return well < 9 or well > 40 or well % 8 in [2, 7]


def offWall(well):
    return not onWall(well)


def posOnWall(pos, buffer=0):
    return pos[0] < 1 + buffer or pos[0] > 5 - buffer or pos[1] < 1 + buffer or pos[1] > 5 - buffer


def posOffWall(pos, buffer=0):
    return not posOnWall(pos, buffer=buffer)


def getDrivePathByLabel(driveLabel: str):
    if os.name == "posix":
        return f"/media/{driveLabel}"
    else:
        possibleLabels = [f"{chr(i)}:" for i in range(ord("c"), ord("m"))]
        for pl in possibleLabels:
            try:
                s = subprocess.check_output(["cmd", f"/c vol {pl}"])
            except subprocess.CalledProcessError:
                continue

            if s.decode().split("\r\n")[0].split(" ")[-1] == driveLabel:
                return pl + "\\"

        return None


def getLoadInfo(config: str) -> LoadInfo:
    if config == "Martin":
        # 2020-6-7: Too few interruptions to really call this an interruption session probably
        excluded_dates = ["20200528", "20200630", "20200702", "20200703", "20200531", "20200603", "20200602",
                          "20200606", "20200605", "20200601", "20200526", "20200527", "20200604", "20200608", "20200609", "20200607"]
        excluded_sessions = ["20200624_1", "20200624_2", "20200628_2"]

        return LoadInfo(configName=config, animalName="Martin",
                        X_START=200, X_FINISH=1175, Y_START=20, Y_FINISH=1275, excludeBoxes=None,
                        # dataDirPath=["WDC1", "martindata", "bradtask"], outputDirPath=["WDC6", "Martin", "processed_data"],
                        dataDirPath=["WDC1", "martindata", "bradtask"], outputDirPath=["Harold", "processed_data"],
                        out_filename="martin_bradtask.rat",
                        excluded_dates=excluded_dates, excluded_sessions=excluded_sessions, minimum_date=None,
                        maximum_date=None,
                        DEFAULT_RIP_DET_TET=37, DEFAULT_RIP_BAS_TET=None)

    if config == "B13":
        excluded_dates = ["20220209"]
        # minimum_date = "20211209"  # had one run on the 8th with probe but used high ripple threshold and a
        # different reference tetrode
        # high ripple thresh on 12/08-1, forgot to turn stim on til after first home on 12/16-2
        excluded_sessions = ["20211208_1", "20211216_2"]
        # Messed up away well order, marked down 20 when he got 12. Ended up giving him reward at 12 twice
        excluded_sessions += ["20220131_2"]
        # Made a custom foundwells field in the behaviornotes for this guy, but would need to update the rest of the
        # import code
        # (i.e. clips loading assumes alternation with home)
        excluded_sessions += ["20220222_2"]
        # video skips
        excluded_sessions += ["20220304_2"]
        excluded_sessions += ["20220307_1"]
        # Cable got messed up during the task, pulley wasn't rolling horizontally
        excluded_sessions += ["20211217_1"]

        # Tracking isn't good. Can go back and fix and add back in later
        excluded_sessions += ["20211213_2", "20211215_1",
                              "20211215_2", "20211217_2",
                              "20211220_1", "20211221_2",
                              "20220123_1",
                              "20220125_2",
                              "20220131_1",
                              ]

        return LoadInfo(configName=config, animalName="B13",
                        X_START=100, X_FINISH=1050, Y_START=20, Y_FINISH=900, excludeBoxes=None,
                        # dataDirPath=["WDC6", "B13", "bradtasksessions"], outputDirPath=["WDC6", "B13", "processed_data"],
                        # dataDirPath=["WDC6", "B13", "bradtasksessions"],
                        dataDirPath=["Harold", "labdata", "B13",
                                     "media", "WDC6", "B13", "bradtasksessions"],
                        # /media/Harold/labdata/B13/media/WDC6/B13/bradtasksessions/
                        outputDirPath=["Harold", "processed_data"],
                        out_filename="B13_bradtask.rat",
                        excluded_dates=excluded_dates, excluded_sessions=excluded_sessions, minimum_date=None,
                        maximum_date=None,
                        DEFAULT_RIP_DET_TET=7, DEFAULT_RIP_BAS_TET=2)

    if config == "B14_old":
        # This is sessions before stim was fixed, so can look at i.e. probe behavior
        excluded_sessions = []
        excluded_sessions += ["20211213_2"]

        return LoadInfo(configName=config, animalName="B14_old",
                        X_START=100, X_FINISH=1050, Y_START=20, Y_FINISH=900, excludeBoxes=None,
                        # dataDirPath=["WDC6", "B14", "bradtasksessions"], outputDirPath=["WDC6", "B14", "processed_data"],
                        # dataDirPath=["WDC6", "B14", "bradtasksessions"],
                        dataDirPath=["Harold", "labdata", "B14",
                                     "media", "WDC6", "B14", "bradtasksessions"],
                        outputDirPath=["Harold", "processed_data"],
                        out_filename="B14_old_bradtask.rat",
                        excluded_dates=[], excluded_sessions=excluded_sessions, minimum_date=None,
                        maximum_date="20220219",
                        DEFAULT_RIP_DET_TET=3, DEFAULT_RIP_BAS_TET=2)

    if config == "B14":
        # Minimum date only after adjusting stim to correct place
        # video skips
        excluded_sessions = ["20220307_2", "20220310_2"]
        # forgot to turn stim on until after first home well
        excluded_sessions += ["20220311_1"]

        return LoadInfo(configName=config, animalName="B14",
                        X_START=100, X_FINISH=1050, Y_START=20, Y_FINISH=900, excludeBoxes=None,
                        # dataDirPath=["WDC6", "B14", "bradtasksessions"], outputDirPath=["WDC6", "B14", "processed_data"],
                        # dataDirPath=["WDC6", "B14", "bradtasksessions"],
                        dataDirPath=["Harold", "labdata", "B14",
                                     "media", "WDC6", "B14", "bradtasksessions"],
                        outputDirPath=["Harold", "processed_data"],
                        out_filename="B14_bradtask.rat",
                        excluded_dates=[], excluded_sessions=excluded_sessions, minimum_date="20220220",
                        maximum_date=None,
                        DEFAULT_RIP_DET_TET=3, DEFAULT_RIP_BAS_TET=2)

    if config == "B16":
        # minimum date: Seems like the older usb videos are all skipping for some reason
        excluded_sessions = []
        # Trodes video skips
        excluded_sessions += ["20221107_2", "20221116_1"]
        # USB video skips, can add back in later
        excluded_sessions += ["20220919_1", "20220919_2"]
        # Just stayed in the corner for ten minutes then I took him out
        excluded_sessions += ["20221118_1"]
        # Used the wrong detection tetrode
        excluded_sessions += ["20221103_1"]

        return LoadInfo(configName=config, animalName="B16",
                        X_START=100, X_FINISH=1050, Y_START=20, Y_FINISH=900, excludeBoxes=None,
                        # dataDirPath=["WDC8", "B16", "bradtasksessions"], outputDirPath=["WDC8", "B16", "processed_data"],
                        # dataDirPath=["WDC8", "B16", "bradtasksessions"],
                        dataDirPath=["Harold", "labdata", "B16",
                                     "media", "WDC8", "B16", "bradtasksessions"],
                        outputDirPath=["Harold", "processed_data"],
                        out_filename="B16_bradtask.rat",
                        excluded_dates=[], excluded_sessions=excluded_sessions, minimum_date="20221101",
                        maximum_date=None,
                        DEFAULT_RIP_DET_TET=4, DEFAULT_RIP_BAS_TET=7)

    if config == "B17":
        # Trodes video skips
        excluded_sessions = ["20221104_1"]

        return LoadInfo(configName=config, animalName="B17",
                        X_START=100, X_FINISH=1050, Y_START=20, Y_FINISH=900, excludeBoxes=None,
                        # dataDirPath=["WDC8", "B17", "bradtasksessions"], outputDirPath=["WDC8", "B17", "processed_data"],
                        # dataDirPath=["WDC4", "Backup", "media", "WDC8", "B17", "bradtasksessions"],
                        # outputDirPath=["WDC4", "Backup", "media", "WDC8", "B17", "processed_data"],
                        dataDirPath=["WDC11", "Data", "B17", "media",
                                     "WDC8", "B17", "bradtasksessions"],
                        # outputDirPath=["WDC11", "Data", "B17",
                        #                "media", "WDC8", "B17", "processed_data"],
                        outputDirPath=["Harold", "processed_data"],
                        out_filename="B17_bradtask.rat",
                        excluded_dates=[], excluded_sessions=excluded_sessions, minimum_date=None,
                        maximum_date=None,
                        DEFAULT_RIP_DET_TET=6, DEFAULT_RIP_BAS_TET=5)

    if config == "B18":
        excluded_sessions = []
        # Found wells doesn't seem like it lines up perfectly with usb video, check it out and fix
        excluded_sessions += ["20221113_2"]
        # Need to improve trodes tracking
        excluded_sessions += ["20221108_1"]
        # usb video skips (add back later, just annoying for now)
        excluded_sessions += ["20220617_1"]
        # Trodes video skips
        excluded_sessions += ["20220620_1", "20220622_2"]
        # Trodes camera partially blocked during probe by pulley system. It's just wells 7-4 ish, might be able to
        # deal with it in analysis
        # At least it shouldn't effect measures at off wall wells
        # ret.excluded_sessions += ["20220621_2"]
        # Amplitude at 100uA, probably don't need to exclude but should note
        # ret.excluded_sessions += ["20220620_2"]
        # stim wire was broken, shoulda been iterruption but was no-stim control
        # (both leads were disconnected, so no stim at all made it to brain)
        excluded_sessions += ["20220624_2"]
        # Messed up the well order, so current pipeline can't handle clips. SHould still be able to analyze probe
        # behavior though if I wanted to
        excluded_sessions += ["20220621_2"]
        # Wasn't actually detecting ripples during these
        excluded_sessions += ["20221102_1", "20221102_2",
                              "20221103_1", "20221103_2", "20221105_1"]

        return LoadInfo(configName=config, animalName="B18",
                        X_START=100, X_FINISH=1050, Y_START=20, Y_FINISH=900, excludeBoxes=[(0, 0, 130, 60)],
                        # dataDirPath=["WDC8", "B18", "bradtasksessions"], outputDirPath=["WDC8", "B18", "processed_data"],
                        # dataDirPath=["WDC8", "B18", "bradtasksessions"],
                        dataDirPath=["Harold", "labdata", "B18",
                                     "media", "WDC8", "B18", "bradtasksessions"],
                        outputDirPath=["Harold", "processed_data"],
                        out_filename="B18_bradtask.rat",
                        excluded_dates=[], excluded_sessions=excluded_sessions, minimum_date=None,
                        maximum_date=None,
                        DEFAULT_RIP_DET_TET=5, DEFAULT_RIP_BAS_TET=3)

    raise ValueError(f"Unknown config val {config}")


def getUSBVideoFile(seshName, possibleDirectories, seshIdx=None, useSeshIdxDirectly=False):
    # print(seshName)
    seshDate, seshTime = seshName.split("_")
    if len(seshTime) == 1 or seshIdx is not None:
        # seshTime is actually session idx, or it's been provided directly
        if seshIdx is None:
            seshIdx = int(seshTime) - 1
        else:
            seshIdx -= 1

        if seshDate > "20221100" and seshDate < "20221122":
            # In november, trimmed videos are labeled by session name
            # print("\tusing date string directly with no dashes")
            usbDateStr = seshDate
        else:
            usbDateStr = "-".join([seshDate[0:4],
                                  seshDate[4:6], seshDate[6:8]])
        possibleUSBVids = []
        for pd in possibleDirectories:
            directName = os.path.join(
                pd, usbDateStr + "_" + str(seshIdx + 1) + ".mkv")
            if os.path.exists(directName):
                return directName

            if not useSeshIdxDirectly:
                gl = os.path.join(pd, usbDateStr + "*.mkv")
                possibleUSBVids += glob.glob(gl)

        if len(possibleUSBVids) == 0:
            print("\tCouldn't find any matching usb video files")
            return None

        possibleUSBVids = sorted(possibleUSBVids)
        return possibleUSBVids[seshIdx]
    else:
        seshTimeVal = float(seshTime[0:2]) * 3600 + \
            float(seshTime[2:4]) * 60 + float(seshTime[4:6])

        usbDateStr = "-".join([seshDate[0:4], seshDate[4:6], seshDate[6:8]])
        possibleUSBVids = []
        for pd in possibleDirectories:
            gl = os.path.join(pd, usbDateStr + "*.mkv")
            possibleUSBVids += glob.glob(gl)

        if len(possibleUSBVids) == 0:
            return None

        minDiff = 24 * 3600
        usbVidFile = None
        for uvi, uv in enumerate(sorted(possibleUSBVids)):
            fname = os.path.basename(uv)
            # fname = uv.split("/")[-1]
            print(fname)
            if " " in fname:
                timeStr = fname.split(" ")[1].split(".")[0]
            else:
                timeStr = fname.split("_")[1].split(".")[0]
            timeVals = [float(v) for v in timeStr.split("-")]
            print(timeVals)
            usbTime = timeVals[0] * 3600 + timeVals[1] * 60 + timeVals[0]

            diff = abs(usbTime - seshTimeVal)
            if diff < minDiff:
                minDiff = diff
                usbVidFile = uv
                # seshWithinDay = uvi

        return usbVidFile


def getTrodesVideoFile(seshInfoFileName, data_dir):
    seshDate, seshIdx = seshInfoFileName.split("_")

    gl = os.path.join(data_dir, seshDate + "_*", seshDate + "_*.1.h264")
    possibleTrodesVids = glob.glob(gl)

    if len(possibleTrodesVids) == 0:
        return None

    seshIdx = int(seshIdx) - 1
    possibleTrodesVids = sorted(possibleTrodesVids)
    return possibleTrodesVids[seshIdx]


def numWellsVisited(nearestWells, countReturns=False, wellSubset=None):
    g = groupby(nearestWells)
    if wellSubset is None:
        wellSubset = allWellNames
    if countReturns:
        return len([k for k, _ in g if k in wellSubset])
    else:
        return len(set([k for k, _ in g if k in wellSubset]))


def weekIdxForDateStr(datestr, d0=date(2016, 1, 4)):
    d = date(int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:8]))
    return (d - d0).days // 7


def getActivelinkLogFile(seshName, possibleDirectories):
    seshDate, seshTime = seshName.split("_")
    seshTimeVal = float(seshTime[0:2]) * 3600 + \
        float(seshTime[2:4]) * 60 + float(seshTime[4:6])

    possibleLogs = []
    for pd in possibleDirectories:
        gl = os.path.join(pd, f"replay_disruption_log_{seshDate}_*.log")
        possibleLogs += glob.glob(gl)

    if len(possibleLogs) == 0:
        return None

    minDiff = 24 * 3600
    logFile = None
    for uvi, uv in enumerate(sorted(possibleLogs)):
        fname = os.path.basename(uv)
        timeStr = fname[-10:-4]
        # print(fname)
        timeVals = [float(timeStr[0:2]), float(
            timeStr[2:4]), float(timeStr[4:6])]
        # print(timeVals)
        logTime = timeVals[0] * 3600 + timeVals[1] * 60 + timeVals[0]

        diff = abs(logTime - seshTimeVal)
        if diff < minDiff:
            minDiff = diff
            logFile = uv
            # seshWithinDay = uvi

    return logFile


def getRotatedWells(wellName: int) -> List[int]:
    """
    Gets symetric wells up to rotating and mirroring the environment
    examples (order of resulting list possibly different):
    2 -> [7, 42, 47]
    12 -> [13, 19, 27, 36, 37, 30, 22]
    """
    def c2n(x, y):
        return 8 * y + x + 2

    def flip(c):
        return 5 - c
    wy = wellName // 8
    wx = (wellName % 8) - 2

    ret = set()
    ret.add(c2n(wx, wy))
    ret.add(c2n(flip(wx), wy))
    ret.add(c2n(wx, flip(wy)))
    ret.add(c2n(flip(wx), flip(wy)))
    ret.add(c2n(wy, wx))
    ret.add(c2n(wy, flip(wx)))
    ret.add(c2n(flip(wy), wx))
    ret.add(c2n(flip(wy), flip(wx)))
    ret.remove(wellName)

    return list(ret)


def generateFoundWells(home_well: int, away_wells: List[int], lastAwayWell: Optional[int], endedOnHome: bool, foundFirstHome: bool) -> List[int]:
    if not foundFirstHome:
        return []
    elif lastAwayWell is None:
        return [home_well]

    foundWells = []
    for aw in away_wells:
        foundWells += [home_well, aw]
        if aw == lastAwayWell:
            break
    if endedOnHome:
        foundWells.append(home_well)
    return foundWells


class TimeThisFunction:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.totalTime = 0

    def __call__(self, *args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = self.func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        self.totalTime += run_time
        return value

    def resetTimer(self):
        self.totalTime = 0


def getPreferredCategoryOrder(cats):
    s = set(cats)
    if s == {"home", "away"}:
        return ["home", "away"]
    elif s == {"home", "symmetric"}:
        return ["home", "symmetric"]
    elif s == {"same", "other"}:
        return ["same", "other"]
    elif set(cats) == {"same", "later"}:
        return ["same", "later"]
    elif set(cats) == {"same", "next"}:
        return ["same", "next"]
    elif set(cats) == {"SWR", "Ctrl"}:
        # return ["SWR", "Ctrl"]
        return ["Ctrl", "SWR"]
    else:
        return sorted(set(cats))


def flagCheck(flags: List[str], flag: str, excludeFromAll=False) -> bool:
    try:
        flags.remove(flag)
        return True
    except ValueError:
        if excludeFromAll:
            return False
        return "all" in flags
